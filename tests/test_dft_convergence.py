import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


def _stretched_dimer():
    atoms = Atoms("Ar2", positions=[(0, 0, 0), (0, 0, 1.6)], cell=[10, 10, 10], pbc=True)
    atoms.center()
    return atoms


@unittest.skipIf(Atoms is None, "ASE is required")
class BFGSConvergenceTests(unittest.TestCase):
    def run_optimize(self, steps):
        profile = {"name": "base", "mode": "pw", "smearing": 0.05, "maxiter": 10, "mixer": None}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run_dft, "build_calculator", return_value=_WritableLJ()):
            return run_dft.optimize_structure(
                atoms=_stretched_dimer(), output_dir=Path(tmp), label="dimer",
                ecut=500.0, kpts=(1, 1, 1), fmax=0.05, steps=steps, profiles=(profile,),
            )

    def test_step_limit_is_a_failure_not_a_result(self):
        with self.assertRaisesRegex(RuntimeError, "did not converge in 1 steps"):
            self.run_optimize(steps=1)

    def test_converged_run_records_steps_and_final_force(self):
        _energy, _gap, atoms, profile = self.run_optimize(steps=200)
        self.assertEqual(profile, "base")
        self.assertLessEqual(atoms.info["fmax_final"], 0.05)
        self.assertGreater(atoms.info["n_steps"], 1)


class _FakeGapCalc:
    """Two bands at -1 and +1 eV around a Fermi level of 0 eV: gap of 2 eV."""

    def get_ibz_k_points(self):
        return [[0.0, 0.0, 0.0]]

    def get_number_of_spins(self):
        return 1

    def get_eigenvalues(self, kpt=0, spin=0):
        import numpy as np

        return np.array([-1.0, 1.0])

    def get_fermi_level(self):
        return 0.0


@unittest.skipIf(Atoms is None, "ASE is required")
class BandGapTests(unittest.TestCase):
    def test_band_gap_uses_ase(self):
        self.assertAlmostEqual(run_dft.compute_band_gap_safe(_FakeGapCalc()), 2.0)

    def test_band_gap_failure_is_reported(self):
        import math

        with mock.patch.object(run_dft, "print0") as report:
            self.assertTrue(math.isnan(run_dft.compute_band_gap_safe(object())))
        self.assertIn("band gap not computed", report.call_args[0][0])


@unittest.skipIf(Atoms is None, "ASE is required")
class WorkflowProfileTests(unittest.TestCase):
    """run_dft_workflow with GPAW replaced by stubs that record the profiles used."""

    def run_workflow(self, ads_profile_by_site, co2_fails=False):
        import argparse
        import csv

        calls = []
        slab = Atoms("ZnO", positions=[(0, 0, 0), (0, 0, 2)], cell=[5, 5, 10])

        def fake_optimize(atoms, output_dir, label, ecut, kpts, fmax, steps, profiles):
            names = [p["name"] for p in profiles]
            calls.append((label, names))
            if label == "clean_slab":
                profile = names[0]
                energy = {"base": -66.0, "robust": -66.5, "very_robust": -67.0}[profile]
            else:
                site = label.split("_", 1)[1]
                profile = ads_profile_by_site[site]
                if profile is None or profile not in names:
                    raise RuntimeError(f"{label}: no profile converged")
                energy = {"base": -89.0, "robust": -89.5, "very_robust": -90.0}[profile]
            out = slab.copy()
            out.info.update(n_steps=10, fmax_final=0.04)
            return energy, 1.0, out, profile

        def fake_co2(output_dir, ecut, vacuum, fmax, steps, profile):
            if co2_fails:
                raise RuntimeError("co2 did not converge")
            return {"base": -22.7, "robust": -22.8, "very_robust": -22.9}[profile["name"]]

        args = argparse.Namespace(materials=["ZnO"], sites=list(ads_profile_by_site),
                                  fmax=0.05, steps=180, ecut=500.0, kpts=(2, 3, 1),
                                  co2_box_vacuum=8.0)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run_dft, "RESULTS_DIR", Path(tmp)), \
                mock.patch.object(run_dft, "DFT_RESULTS_DIR", Path(tmp) / "dft"), \
                mock.patch.object(run_dft, "ensure_directories"), \
                mock.patch.object(run_dft, "read_geometry", return_value=slab.copy()), \
                mock.patch.object(run_dft, "apply_bottom_layer_constraint"), \
                mock.patch.object(run_dft, "adsorption_geometry_metrics", return_value={}), \
                mock.patch.object(run_dft, "optimize_structure", side_effect=fake_optimize), \
                mock.patch.object(run_dft, "co2_reference_energy", side_effect=fake_co2):
            exit_code = 0
            try:
                run_dft.run_dft_workflow(args)
            except SystemExit as exc:
                exit_code = exc.code
            with (Path(tmp) / "dft_summary.csv").open() as handle:
                rows = {row["site"]: row for row in csv.DictReader(handle)}
        return exit_code, rows, calls

    def test_adsorbate_falls_back_and_references_follow_its_profile(self):
        code, rows, calls = self.run_workflow({"top_metal": "robust"})
        self.assertEqual(code, 0)
        row = rows["top_metal"]
        self.assertEqual((row["dft_status"], row["dft_profile"]), ("ok", "robust"))
        # Slab and CO2 recomputed with the same profile as the adsorbate.
        self.assertAlmostEqual(float(row["clean_slab_energy_ev"]), -66.5)
        self.assertAlmostEqual(float(row["co2_reference_energy_ev"]), -22.8)
        self.assertAlmostEqual(float(row["adsorption_energy_ev"]), -89.5 + 66.5 + 22.8)
        self.assertIn(("ZnO_top_metal", ["base", "robust", "very_robust"]), calls)

    def test_failed_case_exits_nonzero(self):
        code, rows, _ = self.run_workflow({"top_metal": "base", "bridge": None})
        self.assertEqual(code, 1)
        self.assertEqual(rows["top_metal"]["dft_status"], "ok")
        self.assertEqual(rows["bridge"]["dft_status"], "ads_failed")

    def test_co2_reference_failure_has_its_own_status(self):
        code, rows, _ = self.run_workflow({"top_metal": "base"}, co2_fails=True)
        self.assertEqual(code, 1)
        self.assertEqual(rows["top_metal"]["dft_status"], "co2_ref_failed")
        self.assertAlmostEqual(float(rows["top_metal"]["clean_slab_energy_ev"]), -66.0)


if __name__ == "__main__":
    unittest.main()
