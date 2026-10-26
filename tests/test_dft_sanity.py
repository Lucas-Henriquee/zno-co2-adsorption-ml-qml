"""Audit D4/D8/D9/D10/D11 checks for run_dft.py, with GPAW replaced by stubs."""

import argparse
import csv
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

try:
    from ase import Atoms
    from ase.calculators.lj import LennardJones
except ImportError:  # pragma: no cover
    Atoms = None

if Atoms is not None:
    from scripts.simulations.dft import run_dft

    class _WritableLJ(LennardJones):
        """Lennard-Jones stand-in for GPAW: cheap forces and a no-op .gpw write."""

        def write(self, *args, **kwargs):
            pass

    class _FailingSCF(_WritableLJ):
        """Calculator whose SCF never converges, as a GPAW KohnShamConvergenceError would."""

        def calculate(self, *args, **kwargs):
            raise RuntimeError("SCF did not converge")

ROOT = Path(__file__).resolve().parents[1]
PROFILES = (
    {"name": "base", "mode": "pw", "smearing": 0.05, "maxiter": 10, "mixer": None},
    {"name": "robust", "mode": "pw", "smearing": 0.05, "maxiter": 10, "mixer": None},
)


def _stretched_dimer():
    # Converges with BFGS in 62 steps from this start (LJ, fmax 0.05).
    atoms = Atoms("Ar2", positions=[(0, 0, 0), (0, 0, 1.6)], cell=[10, 10, 10], pbc=True)
    atoms.center()
    return atoms


def _slab_with_co2(co2_height=2.5):
    """Zn-O 'slab' at z = 10 and a linear CO2 (O, C, O last) above it."""
    z = 10.0 + co2_height
    return Atoms(
        "ZnOOCO",
        positions=[(0.5, 1.0, 10.0), (2.2, 1.0, 10.0), (1.84, 1.0, z), (3.0, 1.0, z), (4.16, 1.0, z)],
        cell=[6.5, 5.2, 25.0], pbc=[True, True, False],
    )


@unittest.skipIf(Atoms is None, "ASE is required")
class ProfileTests(unittest.TestCase):
    def test_fallback_profiles_keep_smearing_at_most_0_05(self):
        for profile in run_dft.OPTIMIZATION_PROFILES:
            self.assertLessEqual(profile["smearing"], 0.05, profile["name"])
        names = [profile["name"] for profile in run_dft.OPTIMIZATION_PROFILES]
        self.assertEqual(names, ["base", "robust", "very_robust"])

    def test_build_calculator_passes_optional_scf_force_criterion(self):
        recorded = []
        fake_gpaw = types.SimpleNamespace(
            FermiDirac=lambda width: ("fd", width),
            GPAW=lambda **kwargs: recorded.append(kwargs) or kwargs,
            Mixer=lambda *args: ("mixer", args),
            PW=lambda ecut: ("pw", ecut),
        )
        with mock.patch.dict(sys.modules, {"gpaw": fake_gpaw}):
            run_dft.build_calculator(500.0, (2, 3, 1), Path("x.txt"))
            run_dft.build_calculator(500.0, (2, 3, 1), Path("x.txt"), scf_force_convergence=0.01)
        self.assertNotIn("convergence", recorded[0])  # default: GPAW's own criteria
        self.assertEqual(recorded[1]["convergence"], {"forces": 0.01})


@unittest.skipIf(Atoms is None, "ASE is required")
class FailureKindTests(unittest.TestCase):
    def optimize(self, calculators, steps, relax_extra_blocks=1, rank=0):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output_dir = Path(tmp.name)
        with mock.patch.object(run_dft, "build_calculator", side_effect=calculators) as build, \
                mock.patch.object(run_dft, "mpi_rank", return_value=rank):
            try:
                return run_dft.optimize_structure(
                    atoms=_stretched_dimer(), output_dir=self.output_dir, label="dimer",
                    ecut=500.0, kpts=(1, 1, 1), fmax=0.05, steps=steps, profiles=PROFILES,
                    relax_extra_blocks=relax_extra_blocks,
                )
            finally:
                self.build_calls = build.call_count

    def test_step_limit_continues_from_last_geometry_with_same_profile(self):
        # 40 steps are not enough (62 needed); one extra block of 40 is.
        _energy, _gap, atoms, profile = self.optimize([_WritableLJ(), _WritableLJ()], steps=40)
        self.assertEqual(profile, "base")
        self.assertEqual(self.build_calls, 1)
        self.assertEqual(atoms.info["relax_restarts"], 1)
        self.assertGreater(atoms.info["n_steps"], 40)  # total over both blocks
        self.assertLessEqual(atoms.info["fmax_final"], 0.05)
        self.assertGreater(atoms.info["fmax_initial"], atoms.info["fmax_final"])

    def test_step_limit_after_extra_blocks_fails_without_other_profiles(self):
        with self.assertRaisesRegex(run_dft.RelaxationNotConverged, "did not converge in 40 steps"):
            self.optimize([_WritableLJ(), _WritableLJ()], steps=20)
        self.assertEqual(self.build_calls, 1)  # robust never tried
        self.assertTrue((self.output_dir / "dimer.base.error.txt").exists())

    def test_scf_exception_moves_to_next_profile(self):
        _energy, _gap, atoms, profile = self.optimize([_FailingSCF(), _WritableLJ()], steps=200)
        self.assertEqual(profile, "robust")
        self.assertEqual(atoms.info["relax_restarts"], 0)
        self.assertIn("SCF did not converge", (self.output_dir / "dimer.base.error.txt").read_text())

    def test_only_rank_zero_writes_error_files(self):
        with self.assertRaisesRegex(RuntimeError, "All optimization profiles failed"):
            self.optimize([_FailingSCF(), _FailingSCF()], steps=10, rank=1)
        self.assertEqual(list(self.output_dir.glob("*.error.txt")), [])

    def test_mpi_rank_reads_gpaw_world_lazily(self):
        world = types.SimpleNamespace(rank=3, size=6)
        modules = {"gpaw": types.ModuleType("gpaw"), "gpaw.mpi": types.SimpleNamespace(world=world)}
        with mock.patch.dict(sys.modules, modules):
            self.assertEqual(run_dft.mpi_rank(), 3)
            self.assertFalse(run_dft.is_master())

    def test_write_summary_skipped_off_rank_zero(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run_dft, "RESULTS_DIR", Path(tmp)), \
                mock.patch.object(run_dft, "mpi_rank", return_value=2):
            run_dft.write_summary([{"material": "ZnO", "site": "bridge"}])
            self.assertFalse((Path(tmp) / "dft_summary.csv").exists())


@unittest.skipIf(Atoms is None, "ASE is required")
class SanityCheckTests(unittest.TestCase):
    def test_overlapping_start_geometry_is_rejected(self):
        atoms = _slab_with_co2()
        atoms.positions[2] = (0.5, 1.0, 11.2)  # CO2 O 1.2 A above the Zn
        with self.assertRaisesRegex(run_dft.GeometryError, r"1\.200 Ang apart"):
            run_dft.check_initial_geometry(atoms, "case", adsorbate_atoms=3)

    def test_co2_bonds_are_exempt_and_mic_is_used(self):
        atoms = _slab_with_co2()
        self.assertGreater(run_dft.check_initial_geometry(atoms, "case", adsorbate_atoms=3), 1.5)
        # Two slab atoms 1.0 A apart only through the periodic x boundary.
        slab = Atoms("ZnO", positions=[(0.2, 0, 10), (6.3, 0, 10)], cell=[6.5, 5.2, 25.0],
                     pbc=[True, True, False])
        with self.assertRaises(run_dft.GeometryError):
            run_dft.check_initial_geometry(slab, "slab")

    def test_final_state_classification_order(self):
        base = {"co_bond_1_ang": 1.18, "co_bond_2_ang": 1.18,
                "c_min_slab_distance_ang": 3.0, "oco_angle_deg": 179.0}
        self.assertEqual(run_dft.classify_final_state(base), "physisorbed")
        self.assertEqual(run_dft.classify_final_state({**base, "oco_angle_deg": 135.0}), "chemisorbed")
        self.assertEqual(run_dft.classify_final_state({**base, "c_min_slab_distance_ang": 5.0}), "desorbed")
        # Dissociation wins over every other criterion.
        broken = {**base, "co_bond_2_ang": 2.4, "c_min_slab_distance_ang": 5.0, "oco_angle_deg": 120.0}
        self.assertEqual(run_dft.classify_final_state(broken), "dissociated")
        self.assertEqual(run_dft.classify_final_state({}), "unknown")

    def test_warnings_for_large_initial_force_and_adsorption_energy(self):
        clean = Atoms("Zn")
        clean.info["fmax_initial"] = 0.3
        ads = Atoms("Zn")
        ads.info["fmax_initial"] = 7.5
        self.assertEqual(run_dft.sanity_warnings(clean, ads, -0.4), "adsorbed initial fmax 7.50 > 5.0 eV/Ang")
        text = run_dft.sanity_warnings(clean, ads, -3.4)
        self.assertEqual(text.split("; ")[1], "|Eads| 3.40 > 3.0 eV")
        ads.info["fmax_initial"] = 1.0
        self.assertEqual(run_dft.sanity_warnings(clean, ads, -0.4), "")


@unittest.skipIf(Atoms is None, "ASE is required")
class WorkflowSanityTests(unittest.TestCase):
    """run_dft_workflow with optimize_structure and the CO2 reference stubbed."""

    def run_workflow(self, sites, clean_energy=-66.0, ads_geometry=None, existing_rows=None,
                     fmax_initial=0.5, ads_energy=-89.0):
        clean = Atoms("ZnO", positions=[(0.5, 1.0, 10.0), (2.2, 1.0, 10.0)],
                      cell=[6.5, 5.2, 25.0], pbc=[True, True, False])
        ads = ads_geometry if ads_geometry is not None else _slab_with_co2()

        def fake_read(path):
            return (clean if path.name == "clean_slab.traj" else ads).copy()

        def fake_optimize(atoms, output_dir, label, ecut, kpts, fmax, steps, profiles, **kwargs):
            out = atoms.copy()
            out.info.update(n_steps=10, fmax_final=0.04, fmax_initial=fmax_initial, relax_restarts=0)
            energy = clean_energy if label == "clean_slab" else ads_energy
            return energy, 1.0, out, profiles[0]["name"]

        args = argparse.Namespace(materials=["ZnO"], sites=sites, fmax=0.05, steps=180,
                                  ecut=500.0, kpts=(2, 3, 1), co2_box_vacuum=8.0,
                                  relax_extra_blocks=1, scf_force_convergence=None)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "geo" / "TiO2").mkdir(parents=True)
            (tmp / "geo" / "TiO2" / "clean_slab.traj").touch()
            if existing_rows:
                with (tmp / "dft_summary.csv").open("w", newline="") as handle:
                    fields = sorted({key for row in existing_rows for key in row})
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(existing_rows)
            with mock.patch.object(run_dft, "RESULTS_DIR", tmp), \
                    mock.patch.object(run_dft, "DFT_RESULTS_DIR", tmp / "dft"), \
                    mock.patch.object(run_dft, "GEOMETRY_DIR", tmp / "geo"), \
                    mock.patch.object(run_dft, "ensure_directories"), \
                    mock.patch.object(run_dft, "read_geometry", side_effect=fake_read), \
                    mock.patch.object(run_dft, "apply_bottom_layer_constraint"), \
                    mock.patch.object(run_dft, "optimize_structure", side_effect=fake_optimize), \
                    mock.patch.object(run_dft, "co2_reference_energy", return_value=-22.7), \
                    mock.patch.object(run_dft, "print0") as report:
                exit_code = 0
                try:
                    run_dft.run_dft_workflow(args)
                except SystemExit as exc:
                    exit_code = exc.code
                with (tmp / "dft_summary.csv").open() as handle:
                    rows = {(row["material"], row["site"]): row for row in csv.DictReader(handle)}
        self.messages = [str(call.args[0]) for call in report.call_args_list]
        return exit_code, rows

    def test_ok_row_has_state_warnings_and_settings_columns(self):
        code, rows = self.run_workflow(["top_metal"], fmax_initial=6.0)
        self.assertEqual(code, 0)
        row = rows[("ZnO", "top_metal")]
        self.assertEqual(row["final_state"], "physisorbed")
        self.assertIn("adsorbed initial fmax 6.00", row["warnings"])
        self.assertEqual((row["ecut_ev"], row["kpts"], row["n_slab_atoms"]), ("500.0", "2x3x1", "2"))
        self.assertEqual(row["adsorbed_relax_restarts"], "0")

    def test_positive_clean_slab_energy_is_a_material_failure(self):
        code, rows = self.run_workflow(["top_metal", "bridge"], clean_energy=317.0)
        self.assertEqual(code, 1)
        for site in ("top_metal", "bridge"):
            self.assertEqual(rows[("ZnO", site)]["dft_status"], "clean_failed")
            self.assertIn("clean slab energy is positive", rows[("ZnO", site)]["error_message"])

    def test_overlapping_adsorbate_start_is_bad_initial_geometry(self):
        ads = _slab_with_co2()
        ads.positions[2] = (0.5, 1.0, 11.0)  # CO2 O 1.0 A above the Zn
        code, rows = self.run_workflow(["top_metal"], ads_geometry=ads)
        self.assertEqual(code, 1)
        self.assertEqual(rows[("ZnO", "top_metal")]["dft_status"], "bad_initial_geometry")

    def test_stale_rows_with_other_settings_are_dropped_with_a_warning(self):
        settings = {"ecut_ev": "500.0", "kpts": "2x3x1", "n_slab_atoms": "2"}
        existing = [
            {"material": "TiO2", "site": "bridge", "dft_status": "ok", **settings},
            {"material": "TiO2", "site": "top_metal", "dft_status": "ok", **settings, "kpts": "3x3x1"},
            {"material": "ZnO", "site": "top_oxygen", "dft_status": "ok"},  # pre-audit row
        ]
        code, rows = self.run_workflow(["top_metal"], existing_rows=existing)
        self.assertEqual(code, 0)
        self.assertEqual(set(rows), {("TiO2", "bridge"), ("ZnO", "top_metal")})
        self.assertTrue(any("dropped 2 row(s)" in message for message in self.messages))


@unittest.skipIf(Atoms is None, "ASE is required")
class DefaultsTests(unittest.TestCase):
    def test_cli_defaults_match_production_yaml(self):
        settings = yaml.safe_load((ROOT / "configs/simulations/zno_production.yaml").read_text())["methods"]["dft"]
        with mock.patch.object(sys, "argv", ["run_dft"]):
            args = run_dft.parse_args()
        self.assertEqual(list(args.kpts), settings["kpoints"])
        self.assertEqual(args.steps, settings["max_steps"])
        self.assertEqual(args.fmax, settings["fmax_ev_per_angstrom"])
        self.assertEqual(args.ecut, settings["ecut_ev"])
        self.assertEqual(args.co2_box_vacuum, settings["co2_box_vacuum_angstrom"])
        self.assertEqual(args.relax_extra_blocks, settings["relax_extra_blocks"])
        self.assertEqual(args.scf_force_convergence, settings["scf_force_convergence"])
        self.assertTrue(math.isclose(args.fmax, 0.05))


if __name__ == "__main__":
    unittest.main()
