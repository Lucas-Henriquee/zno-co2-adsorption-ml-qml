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


if __name__ == "__main__":
    unittest.main()
