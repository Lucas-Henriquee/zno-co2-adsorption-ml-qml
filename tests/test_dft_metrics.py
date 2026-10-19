import math
import unittest
from unittest import mock

import numpy as np

try:
    from ase import Atoms
except ImportError:  # pragma: no cover
    Atoms = None

if Atoms is not None:
    from scripts.simulations.dft import run_dft


class _GapCalc:
    """Bands at -1 and +1 eV around E_F = 0; fixed_density returns a 0.6 eV calculator."""

    def __init__(self, gap=2.0):
        self.gap = gap
        self.fixed_density_kwargs = None

    def get_ibz_k_points(self):
        return [[0.0, 0.0, 0.0]]

    def get_number_of_spins(self):
        return 1

    def get_eigenvalues(self, kpt=0, spin=0):
        return np.array([-self.gap / 2, self.gap / 2])

    def get_fermi_level(self):
        return 0.0

    def get_number_of_electrons(self):
        return 144

    def fixed_density(self, **kwargs):
        self.fixed_density_kwargs = kwargs
        return _GapCalc(gap=0.6)


@unittest.skipIf(Atoms is None, "ASE is required")
class GammaGapTests(unittest.TestCase):
    def test_even_mesh_uses_gamma_centred_fixed_density(self):
        calc = _GapCalc()
        gap = run_dft.compute_band_gap_gamma(calc, (2, 3, 1), "gap.txt")
        self.assertAlmostEqual(gap, 0.6)
        kwargs = calc.fixed_density_kwargs
        self.assertEqual(kwargs["kpts"], {"size": (2, 3, 1), "gamma": True})
        self.assertEqual(kwargs["nbands"], 72 + 12)  # 144 electrons -> 72 occupied bands

    def test_odd_mesh_already_contains_gamma(self):
        calc = _GapCalc()
        self.assertAlmostEqual(run_dft.compute_band_gap_gamma(calc, (3, 3, 1), "gap.txt"), 2.0)
        self.assertIsNone(calc.fixed_density_kwargs)

    def test_failure_is_reported_as_nan(self):
        with mock.patch.object(run_dft, "print0") as report:
            self.assertTrue(math.isnan(run_dft.compute_band_gap_gamma(object(), (2, 3, 1), "x")))
        self.assertIn("Gamma-grid band gap not computed", report.call_args[0][0])


@unittest.skipIf(Atoms is None, "ASE is required")
class AdsorptionMetricsTests(unittest.TestCase):
    def system(self):
        # Zn-O "slab" at z = 10 and a CO2 lying along x, 2.5 A above, with one O wrapped
        # across the x boundary of a 3.25 A cell.
        atoms = Atoms(
            "ZnOOCO",
            positions=[(0.5, 1.0, 10.0), (2.0, 1.0, 10.0),
                       (1.84, 1.0, 12.5), (3.0, 1.0, 12.5), (4.16, 1.0, 12.5)],
            cell=[3.25, 5.0, 25.0], pbc=[True, True, False],
        )
        atoms.wrap()
        return atoms

    def test_co2_geometry_uses_minimum_image(self):
        metrics = run_dft.adsorption_geometry_metrics(self.system())
        self.assertAlmostEqual(metrics["co_bond_1_ang"], 1.16, places=6)
        self.assertAlmostEqual(metrics["co_bond_2_ang"], 1.16, places=6)
        self.assertAlmostEqual(metrics["oco_angle_deg"], 180.0, places=4)

    def test_height_and_minimum_distances_are_reported_separately(self):
        metrics = run_dft.adsorption_geometry_metrics(self.system())
        self.assertAlmostEqual(metrics["c_surface_distance_ang"], 2.5, places=6)
        # Nearest slab atom to C (x = 3.00) is the periodic image of Zn (x = 0.50 + 3.25),
        # 0.75 A away in x, closer than the O at x = 2.00: sqrt(0.75^2 + 2.5^2).
        self.assertAlmostEqual(metrics["c_min_slab_distance_ang"], np.hypot(0.75, 2.5), places=6)
        # Wrapped O at x = 4.16 - 3.25 = 0.91 is 0.41 A from the Zn at x = 0.5.
        self.assertAlmostEqual(metrics["o_min_metal_distance_ang"], np.hypot(0.41, 2.5), places=6)


if __name__ == "__main__":
    unittest.main()
