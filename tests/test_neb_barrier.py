"""Barrier definition, pair selection and phase step accounting of run_neb (no GPAW)."""

import math
import unittest
import unittest.mock
from pathlib import Path

try:
    import ase  # noqa: F401
except ImportError:  # pragma: no cover
    ase = None

if ase is not None:
    from scripts.simulations.neb import run_neb


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(ase is None, "ASE is required")
class BarrierMetricsTests(unittest.TestCase):
    def test_barrier_excludes_endpoints(self):
        # Endothermic path: the final endpoint is the global maximum.
        metrics = run_neb.barrier_metrics([0.0, 0.3, 0.5, 0.4, 0.2, 0.9, 1.2])
        self.assertEqual(metrics["highest_image_index"], 5)
        self.assertAlmostEqual(metrics["forward_barrier_eV"], 0.9)
        self.assertAlmostEqual(metrics["reverse_barrier_eV"], 0.9 - 1.2)
        self.assertAlmostEqual(metrics["reaction_energy_eV"], 1.2)
        self.assertTrue(metrics["no_internal_saddle"])

    def test_saddle_and_reverse_barrier(self):
        metrics = run_neb.barrier_metrics([-10.0, -9.8, -9.5, -9.7, -9.9, -9.95, -10.2])
        self.assertEqual(metrics["highest_image_index"], 2)
        self.assertAlmostEqual(metrics["forward_barrier_eV"], 0.5)
        self.assertAlmostEqual(metrics["reverse_barrier_eV"], 0.7)
        self.assertAlmostEqual(metrics["reaction_energy_eV"], -0.2)
        self.assertFalse(metrics["no_internal_saddle"])
        self.assertAlmostEqual(metrics["relative_energies"][0], 0.0)

    def test_monotonic_exothermic_profile_is_flagged(self):
        metrics = run_neb.barrier_metrics([0.0, -0.1, -0.2, -0.3])
        self.assertTrue(metrics["no_internal_saddle"])
        self.assertAlmostEqual(metrics["forward_barrier_eV"], -0.1)

    def test_invalid_energies(self):
        self.assertIsNone(run_neb.barrier_metrics([0.0, 1.0]))
        self.assertIsNone(run_neb.barrier_metrics([0.0, math.nan, 0.1]))


class _FakeNEB:
    climb = False


class _FakeOptimizer:
    """Mimics ASE: run(steps) allows `steps` more on top of the accumulated nsteps."""

    def __init__(self, converge_after):
        self.nsteps = 0
        self.calls = []
        self.converge_after = converge_after  # per call: steps needed, or None

    def run(self, fmax, steps):
        needed = self.converge_after[len(self.calls)]
        self.calls.append((fmax, steps))
        if needed is not None and needed <= steps:
            self.nsteps += needed
            return True
        self.nsteps += steps
        return False


@unittest.skipIf(ase is None, "ASE is required")
class PhaseTests(unittest.TestCase):
    def run_phases(self, converge_after, climb=True):
        neb = _FakeNEB()
        opt = _FakeOptimizer(converge_after)
        record = run_neb.run_neb_phases(
            neb, opt, fmax=0.08, fmax_ci=0.05, steps=300, climb=climb, climb_start_fraction=0.30,
        )
        return neb, opt, record

    def test_ci_phase_runs_even_if_plain_phase_converged(self):
        neb, opt, record = self.run_phases([20, 15])
        self.assertTrue(neb.climb)
        self.assertEqual(opt.calls, [(0.08, 90), (0.05, 280)])
        self.assertTrue(record["climb_activated"])
        self.assertTrue(record["converged_plain"])
        self.assertTrue(record["converged_ci"])
        self.assertEqual(record["n_steps_plain"], 20)
        self.assertEqual(record["n_steps_ci"], 15)
        self.assertTrue(record["converged"])

    def test_total_budget_is_respected(self):
        _, opt, record = self.run_phases([None, None])
        self.assertEqual(opt.calls, [(0.08, 90), (0.05, 210)])
        self.assertEqual(record["n_steps_plain"] + record["n_steps_ci"], 300)
        self.assertEqual(opt.nsteps, 300)
        self.assertFalse(record["converged"])

    def test_pair_converged_only_if_ci_converged(self):
        _, _, record = self.run_phases([10, None])
        self.assertTrue(record["converged_plain"])
        self.assertFalse(record["converged_ci"])
        self.assertFalse(record["converged"])

    def test_without_climb_single_phase_uses_full_budget(self):
        neb, opt, record = self.run_phases([None], climb=False)
        self.assertFalse(neb.climb)
        self.assertEqual(opt.calls, [(0.08, 300)])
        self.assertFalse(record["climb_activated"])
        self.assertIsNone(record["converged_ci"])


@unittest.skipIf(ase is None, "ASE is required")
class PairSelectionTests(unittest.TestCase):
    def test_defaults_match_yaml(self):
        import yaml

        config = yaml.safe_load((ROOT / "configs/simulations/zno_production.yaml").read_text())
        neb = config["methods"]["neb"]
        args = run_neb.parse_args([])
        self.assertEqual(args.ecut, neb["ecut_ev"])
        self.assertEqual(list(args.kpts), neb["kpoints"])
        self.assertEqual(list(args.kpts), config["methods"]["dft"]["kpoints"])
        self.assertEqual(args.images, neb["internal_images"])
        self.assertEqual(args.steps, neb["max_steps"])
        self.assertEqual(args.fmax, neb["fmax_ev_per_angstrom"])
        self.assertEqual(args.interpolation, neb["interpolation"])
        self.assertEqual(args.optimizer, neb["optimizer"])
        self.assertEqual(args.climb_start_fraction, neb["climb_start_fraction"])
        self.assertTrue(args.climb)
        run_neb.resolve_settings(args, config)
        self.assertEqual(args.fmax_ci, neb["fmax_ci_ev_per_angstrom"])
        self.assertEqual(args.dft_fmax, config["methods"]["dft"]["fmax_ev_per_angstrom"])

    def test_default_pairs_are_adjacent_hops(self):
        args = run_neb.parse_args([])
        self.assertEqual(run_neb.get_pairs(args, {}), [("top_metal", "bridge"), ("bridge", "top_oxygen")])

    def test_yaml_pairs_and_site_filter(self):
        config = {"methods": {"neb": {"pairs": [["bridge", "top_oxygen"], "top_metal:top_oxygen"]}}}
        args = run_neb.parse_args(["--sites", "bridge", "top_oxygen"])
        self.assertEqual(run_neb.get_pairs(args, config), [("bridge", "top_oxygen")])

    def test_cli_pairs_override_yaml(self):
        config = {"methods": {"neb": {"pairs": [["bridge", "top_oxygen"]]}}}
        args = run_neb.parse_args(["--pairs", "top_metal:bridge"])
        self.assertEqual(run_neb.get_pairs(args, config), [("top_metal", "bridge")])

    def test_invalid_pair(self):
        args = run_neb.parse_args(["--pairs", "bridge:bridge"])
        with self.assertRaises(SystemExit):
            run_neb.get_pairs(args, {})

    def test_dft_settings_mismatch_aborts(self):
        args = run_neb.parse_args(["--kpts", "3", "3", "1"])
        with self.assertRaisesRegex(SystemExit, "kpts"):
            run_neb.resolve_settings(args, {"methods": {"dft": {"ecut_ev": 500.0, "kpoints": [2, 3, 1]}}})

    def test_parallel_layout_requires_divisible_ranks(self):
        with self.assertRaisesRegex(SystemExit, "6 % 5 != 0"):
            run_neb.check_parallel_layout(5, 6, True)
        run_neb.check_parallel_layout(3, 6, True)
        run_neb.check_parallel_layout(5, 6, False)
        run_neb.check_parallel_layout(5, 1, True)

    def test_run_neb_fails_early_on_bad_layout(self):
        class _World:
            size = 6
            rank = 0

        args = run_neb.parse_args(["--images", "5", "--parallel-images", "--results-dir", "/nonexistent"])
        with unittest.mock.patch.object(run_neb, "get_world", return_value=_World()), \
                unittest.mock.patch.object(run_neb, "read_dft_summary") as read_summary:
            with self.assertRaisesRegex(SystemExit, "divisible"):
                run_neb.run_neb(args)
        read_summary.assert_not_called()

    def test_ci_phase_needs_budget(self):
        args = run_neb.parse_args(["--climb-start-fraction", "1.0"])
        with self.assertRaisesRegex(SystemExit, "climbing-image"):
            run_neb.resolve_settings(args, {})


if __name__ == "__main__":
    unittest.main()
