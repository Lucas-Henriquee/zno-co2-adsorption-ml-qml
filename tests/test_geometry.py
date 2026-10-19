import unittest

import numpy as np

try:
    from ase.neighborlist import neighbor_list
except ImportError:  # pragma: no cover
    neighbor_list = None

if neighbor_list is not None:
    from scripts.simulations.dft import build_geometries as geometry


@unittest.skipIf(neighbor_list is None, "ASE is required")
class ZnOGeometryTests(unittest.TestCase):
    def test_bulk_is_tetrahedral_wurtzite(self):
        bulk = geometry.build_bulk("ZnO")
        i, _ = neighbor_list("ij", bulk, 2.2)
        distances = bulk.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        self.assertGreater(distances.min(), 1.9)  # the (0, 0, z) basis gave 0.61 A
        self.assertEqual(np.bincount(i, minlength=len(bulk)).tolist(), [4, 4, 4, 4])

    def test_slab_is_2x2_with_four_planes(self):
        slab = geometry.build_slab("ZnO")
        self.assertEqual(len(slab), 32)
        np.testing.assert_allclose(slab.cell.lengths()[:2], [6.498, 10.412], atol=1e-3)
        self.assertEqual(len(np.unique(np.round(slab.positions[:, 2], 3))), 4)

    def test_bridge_uses_surface_dimer(self):
        slab = geometry.build_slab("ZnO")
        top = geometry._top_atom_indices(slab)
        metals = [i for i in top if slab[i].symbol == "Zn"]
        oxygens = [i for i in top if slab[i].symbol == "O"]
        pair = geometry._nearest_metal_oxygen_pair(slab, metals, oxygens)
        self.assertAlmostEqual(slab.positions[pair[0], 2], slab.positions[pair[1], 2], places=6)
        self.assertAlmostEqual(slab.get_distance(*pair, mic=True), 1.991, places=3)

    def test_co2_does_not_touch_its_periodic_copy(self):
        slab = geometry.build_slab("ZnO")
        sites = geometry.select_adsorption_sites(slab, "Zn")
        for site in geometry.ADSORPTION_SITES:
            with self.subTest(site=site):
                system = geometry.add_co2_to_site(slab, site, sites[site])
                molecule = system.positions[len(slab):]
                shifts = [i * system.cell[0] + j * system.cell[1]
                          for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
                gap = min(np.linalg.norm(molecule[:, None] - (molecule[None] + s), axis=2).min()
                          for s in shifts)
                self.assertGreaterEqual(gap, 3.0)  # the 1x1 cell gave 0.93 A
                distances = system.get_all_distances(mic=True)[len(slab):, :len(slab)]
                self.assertGreaterEqual(distances.min(), 1.8)


if __name__ == "__main__":
    unittest.main()
