"""End-to-end run_neb checks with an analytic stand-in for GPAW, plus plot/table discovery."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixAtoms
    from ase.io import write
except ImportError:  # pragma: no cover
    Atoms = None

if Atoms is not None:
    from scripts.simulations.dft import run_dft
    from scripts.simulations.neb import run_neb

    PERIOD = 3.0  # Angstrom along x between equivalent minima
    AMPLITUDE = 0.2  # eV per adsorbate atom; barrier = 3 * AMPLITUDE
    X0 = 1.0

    class _CorrugatedSurface(Calculator):
        """Cosine corrugation along x for the last three atoms, harmonic in y and z.

        Minima at x = X0 + n * PERIOD; the slab atoms feel no force.
        """

        implemented_properties = ["energy", "forces"]

        def __init__(self, reference):
            super().__init__()
            self.reference = reference

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = self.atoms.positions
            forces = np.zeros_like(pos)
            energy = 0.0
            k = 2.0 * np.pi / PERIOD
            for i in range(len(pos) - 3, len(pos)):
                phase = k * (pos[i, 0] - X0)
                energy += 0.5 * AMPLITUDE * (1.0 - np.cos(phase))
                forces[i, 0] = -0.5 * AMPLITUDE * k * np.sin(phase)
                dyz = pos[i, 1:] - self.reference[i, 1:]
                energy += 0.5 * 2.0 * float(dyz @ dyz)
                forces[i, 1:] = -2.0 * dyz
            self.results = {"energy": float(energy), "forces": forces}


def _structure(x_shift=0.0):
    slab = [("Zn", (0.5, 0.5, 5.0)), ("O", (3.5, 3.5, 5.0)), ("Zn", (0.5, 3.5, 7.0)), ("O", (3.5, 0.5, 7.0))]
    co2 = [("O", (X0 + x_shift, 1.84, 10.0)), ("C", (X0 + x_shift, 3.0, 10.0)), ("O", (X0 + x_shift, 4.16, 10.0))]
    symbols = [s for s, _ in slab + co2]
    positions = [p for _, p in slab + co2]
    atoms = Atoms(symbols, positions=positions, cell=[6.0, 6.0, 16.0], pbc=(True, True, False))
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    return atoms


@unittest.skipIf(Atoms is None, "ASE is required")
class RunNebTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.results = Path(self._tmp.name)
        self.reference = _structure().positions.copy()
        self.calc_kwargs = []
        (self.results / "config.yaml").write_text(
            "schema_version: 1\n"
            "methods:\n"
            "  dft: {fmax_ev_per_angstrom: 0.05, ecut_ev: 500.0, kpoints: [2, 3, 1]}\n"
            "  neb:\n"
            "    fmax_ci_ev_per_angstrom: 0.05\n"
            "    pairs: [[top_metal, bridge], [bridge, top_oxygen]]\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _fake_build_calculator(self, **kwargs):
        self.calc_kwargs.append(kwargs)
        return _CorrugatedSurface(self.reference)

    def write_endpoints(self, shifts, profiles=None, energies=None):
        profiles = profiles or {}
        energies = energies or {}
        rows = []
        for site, shift in shifts.items():
            atoms = _structure(shift)
            folder = self.results / "dft" / "ZnO" / site
            folder.mkdir(parents=True, exist_ok=True)
            write(folder / f"ZnO_{site}_optimized.xyz", atoms)
            check = atoms.copy()
            check.calc = _CorrugatedSurface(self.reference)
            rows.append({
                "material": "ZnO",
                "site": site,
                "dft_status": "ok",
                "dft_profile": profiles.get(site, "base"),
                "adsorbed_energy_ev": energies.get(site, check.get_potential_energy()),
            })
        with (self.results / "dft_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def run_pairs(self, *extra):
        args = run_neb.parse_args([
            "--results-dir", str(self.results), "--interpolation", "linear", "--steps", "60", "--images", "5", *extra,
        ])
        with mock.patch.object(run_dft, "build_calculator", side_effect=self._fake_build_calculator), \
                contextlib.redirect_stdout(io.StringIO()):
            return run_neb.run_neb(args)

    def summary(self, pair):
        return json.loads((self.results / "neb" / "ZnO" / pair / "neb_run_summary.json").read_text())

    def test_converged_pairs_carry_energies_and_provenance(self):
        self.write_endpoints({"top_metal": 0.0, "bridge": PERIOD, "top_oxygen": 2 * PERIOD})
        outcomes = self.run_pairs()
        self.assertEqual([o["path"] for o in outcomes], ["top_metal_to_bridge", "bridge_to_top_oxygen"])

        summary = self.summary("top_metal_to_bridge")
        self.assertEqual(summary["status"], "converged")
        self.assertTrue(summary["climb_activated"])
        self.assertTrue(summary["converged_ci"])
        self.assertLessEqual(summary["n_steps_plain"] + summary["n_steps_ci"], 60)
        self.assertEqual(summary["highest_image_index"], 3)
        self.assertAlmostEqual(summary["forward_barrier_eV"], 3 * AMPLITUDE, places=3)
        self.assertAlmostEqual(summary["reverse_barrier_eV"], 3 * AMPLITUDE, places=3)
        self.assertFalse(summary["no_internal_saddle"])
        self.assertEqual(summary["calculator"]["profile"], "base")
        self.assertEqual(summary["calculator"]["smearing"], 0.05)
        self.assertEqual(len(summary["endpoint_files"]["initial"]["sha256"]), 64)
        self.assertEqual(summary["fixed_indices"], [0, 1])

        # Every image (endpoints included) uses the DFT settings of the run.
        self.assertEqual(len(self.calc_kwargs), 2 * (5 + 2))
        for kwargs in self.calc_kwargs:
            self.assertEqual(kwargs["ecut"], 500.0)
            self.assertEqual(kwargs["kpts"], (2, 3, 1))
            self.assertEqual(kwargs["smearing"], 0.05)
            self.assertEqual(kwargs["mode"], "pw")

        # Image files carry energies, read back without GPAW logs.
        folder = self.results / "neb" / "ZnO" / "top_metal_to_bridge"
        energies = run_neb.read_image_energies(folder)
        self.assertEqual(len(energies), 7)
        self.assertAlmostEqual(max(energies), 3 * AMPLITUDE, places=3)
        self.assertTrue((folder / "image_03.xyz").exists())

        with (self.results / "neb" / "ZnO" / "neb_paths_summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["status"] for row in rows}, {"converged"})

    def test_default_idpp_run_with_climbing_phase(self):
        self.write_endpoints({"top_metal": 0.0, "bridge": PERIOD})
        args = run_neb.parse_args([
            "--results-dir", str(self.results), "--initial-site", "top_metal", "--final-site", "bridge",
        ])
        with mock.patch.object(run_dft, "build_calculator", side_effect=self._fake_build_calculator), \
                contextlib.redirect_stdout(io.StringIO()):
            run_neb.run_neb(args)
        summary = self.summary("top_metal_to_bridge")
        self.assertEqual(summary["interpolation"], "idpp")
        self.assertEqual(summary["status"], "converged")
        self.assertEqual(summary["plain_step_budget"], 90)
        self.assertEqual(summary["ci_step_budget"], 300 - summary["n_steps_plain"])
        self.assertTrue(summary["climb_activated"])
        self.assertAlmostEqual(summary["forward_barrier_eV"], 3 * AMPLITUDE, places=2)

    def test_provenance_mismatch_aborts_and_exits_nonzero(self):
        self.write_endpoints({"top_metal": 0.0, "bridge": PERIOD}, energies={"bridge": 0.05})
        with self.assertRaises(SystemExit) as raised:
            self.run_pairs("--initial-site", "top_metal", "--final-site", "bridge")
        self.assertEqual(raised.exception.code, 1)
        summary = self.summary("top_metal_to_bridge")
        self.assertEqual(summary["status"], "aborted_provenance")
        self.assertAlmostEqual(summary["endpoints"]["final"]["energy_difference_ev"], -0.05)
        self.assertNotIn("forward_barrier_eV", summary)

    def test_unconverged_endpoint_aborts(self):
        self.write_endpoints({"top_metal": 0.0, "bridge": PERIOD + 0.2})
        with self.assertRaises(SystemExit):
            self.run_pairs("--initial-site", "top_metal", "--final-site", "bridge")
        self.assertEqual(self.summary("top_metal_to_bridge")["status"], "aborted_endpoint_forces")

    def test_identical_endpoints_are_skipped(self):
        self.write_endpoints({"top_metal": 0.0, "bridge": 0.1})
        outcomes = self.run_pairs("--initial-site", "top_metal", "--final-site", "bridge")
        self.assertEqual(outcomes[0]["status"], "skipped_identical_endpoints")
        self.assertEqual(self.calc_kwargs, [])

    def test_profile_mismatch_is_skipped_and_fails(self):
        self.write_endpoints({"top_metal": 0.0, "bridge": PERIOD}, profiles={"bridge": "robust"})
        with self.assertRaises(SystemExit):
            self.run_pairs("--initial-site", "top_metal", "--final-site", "bridge")
        self.assertEqual(self.summary("top_metal_to_bridge")["status"], "skipped_profile_mismatch")
        self.assertEqual(self.calc_kwargs, [])

    def test_different_atom_order_is_rejected(self):
        self.write_endpoints({"top_metal": 0.0, "bridge": PERIOD})
        path = self.results / "dft" / "ZnO" / "bridge" / "ZnO_bridge_optimized.xyz"
        atoms = _structure(PERIOD)
        atoms.set_chemical_symbols(["O", "Zn", "Zn", "O", "O", "C", "O"])
        write(path, atoms)
        with self.assertRaises(SystemExit):
            self.run_pairs("--initial-site", "top_metal", "--final-site", "bridge")
        self.assertEqual(self.summary("top_metal_to_bridge")["status"], "aborted_incompatible_endpoints")

    def test_plot_and_table_discover_pairs_and_skip_unconverged(self):
        from scripts.analysis import neb_energy_table
        from scripts.visualization import plot_neb

        self.write_endpoints({"top_metal": 0.0, "bridge": PERIOD, "top_oxygen": 2 * PERIOD})
        self.run_pairs()

        base = self.results / "neb" / "ZnO"
        # Mark one pair as not converged: plots must skip it.
        path = base / "bridge_to_top_oxygen" / "neb_run_summary.json"
        summary = json.loads(path.read_text())
        summary["status"] = "max_steps_reached"
        path.write_text(json.dumps(summary))

        with contextlib.redirect_stdout(io.StringIO()):
            plot_neb.main(["--base", str(base), "--dpi", "40"])
            neb_energy_table.main(["--base", str(base)])

        with (base / "comparison_plots" / "neb_summary_table.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["folder"] for row in rows], ["top_metal_to_bridge"])
        self.assertEqual(rows[0]["highest_image_index"], "3")
        self.assertTrue((base / "comparison_plots" / "individual_profiles" / "neb_profile_top_metal_to_bridge.png").exists())
        self.assertTrue((base / "comparison_plots" / "migration_barrier_comparison.png").exists())

        with (base / "comparison_plots" / "neb_image_energies.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["pair"] for row in rows}, {"top_metal_to_bridge", "bridge_to_top_oxygen"})
        self.assertEqual({row["status"] for row in rows if row["pair"] == "bridge_to_top_oxygen"}, {"max_steps_reached"})


if __name__ == "__main__":
    unittest.main()
