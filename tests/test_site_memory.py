"""Site-memory analysis and the numpy ZZ kernel (no qiskit, no GPAW)."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import site_memory as sm
from scripts.models.qml.zz_kernel import entanglement_pairs, zz_kernel, zz_states

SITES = ["top_metal", "top_oxygen", "bridge"]
START_HEIGHT = {"top_metal": 2.7, "top_oxygen": 3.6, "bridge": 4.0}


def write_run(root: Path, seeds, temperature=300.0, memory_fs=100.0, eq_steps=100, prod_steps=200,
              interval=20, rng_seed=0):
    """Frames whose geometry keeps the site for memory_fs, then is pure noise around 3.8 A."""
    rng = np.random.default_rng(rng_seed)
    (root).mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(
        "methods:\n  md:\n"
        f"    temperature_kelvin: {temperature}\n    timestep_fs: 1.0\n"
        f"    log_interval: {interval}\n    equilibration_steps: {eq_steps}\n", encoding="utf-8")
    for site in SITES:
        rows = []
        for seed in seeds:
            for phase, offset, steps in (("equilibration", 0, eq_steps), ("production", eq_steps, prod_steps)):
                for index in range(steps // interval + 1):
                    step = offset + index * interval
                    t = float(step)
                    height = START_HEIGHT[site] if t < memory_fs else 3.8
                    height += rng.normal(0, 0.02 if t < memory_fs else 0.4)
                    rows.append({"seed": seed, "phase": phase, "frame_index": index, "step": step,
                                 "time_fs": index * interval, "nearest_site": site,
                                 "nearest_site_lateral_ang": 0.0, "start_site_lateral_ang": 0.0,
                                 "c_height_top_plane_ang": height,
                                 "c_min_slab_distance_ang": height + 0.2 + rng.normal(0, 0.02),
                                 "c_o_min_slab_distance_ang": height - 0.1 + rng.normal(0, 0.02),
                                 "desorbed": 0, "msd_lateral_ang2": 0.0, "site_assignment": site})
        folder = root / "md" / "ZnO" / site
        folder.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(folder / sm.FRAME_FILE, index=False)


class ZZKernelTests(unittest.TestCase):
    def test_kernel_is_a_valid_fidelity_kernel(self):
        X = np.random.default_rng(1).uniform(0, np.pi, (12, 3))
        for reps in (1, 2):
            for entanglement in ("linear", "circular", "full"):
                K = zz_kernel(X, reps=reps, entanglement=entanglement)
                np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
                np.testing.assert_allclose(K, K.T, atol=1e-12)
                self.assertGreater(np.linalg.eigvalsh(K).min(), -1e-10)
                self.assertTrue(((K >= -1e-12) & (K <= 1 + 1e-12)).all())

    def test_states_are_normalised_and_match_a_hand_built_circuit(self):
        x = np.array([0.3, 1.1])
        # H on both qubits, phases P(2 x_i), then CX P(2 (pi - x0)(pi - x1)) CX on (0, 1).
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        state = np.kron(H, H) @ np.array([1, 0, 0, 0], dtype=complex)
        for k in range(4):
            b0, b1 = k & 1, (k >> 1) & 1
            phase = 2 * x[0] * b0 + 2 * x[1] * b1 + 2 * (np.pi - x[0]) * (np.pi - x[1]) * (b0 ^ b1)
            state[k] *= np.exp(1j * phase)
        np.testing.assert_allclose(zz_states(x[None, :])[0], state, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(zz_states(x[None, :])[0])), 1.0)

    def test_single_qubit_phase_has_period_pi(self):
        # The reason the site-memory features are scaled to [0, pi/2], not [0, pi].
        self.assertAlmostEqual(float(zz_kernel(np.array([[0.0]]), np.array([[np.pi]]))[0, 0]), 1.0)
        self.assertLess(float(zz_kernel(np.array([[0.0]]), np.array([[np.pi / 2]]))[0, 0]), 1e-12)

    def test_entanglement_pairs(self):
        self.assertEqual(entanglement_pairs(3, "linear"), [(0, 1), (1, 2)])
        self.assertEqual(sorted(entanglement_pairs(3, "circular")), [(0, 1), (1, 2), (2, 0)])
        self.assertEqual(entanglement_pairs(3, "full"), [(0, 1), (0, 2), (1, 2)])

    def test_evaluate_grouped_accepts_zz_exact(self):
        from scripts.models import evaluate_grouped as eg

        kernel = eg.resolve_kernel("zz_exact", 3)
        X = np.random.default_rng(2).uniform(0, 1, (5, 3))
        np.testing.assert_allclose(kernel(X), zz_kernel(X))
        with self.assertRaises(ValueError):
            eg.resolve_kernel("zz_exact", 3, n_qubits=6)


class SiteMemoryDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_time_axis_counts_the_end_of_equilibration_once(self):
        write_run(self.root / "run", [1, 2])
        frames, _ = sm.load_frames([self.root / "run"], "ZnO", SITES, 1.0)
        one = frames[(frames["site"] == "bridge") & (frames["seed"] == 1)]
        self.assertTrue(one["t_fs"].is_unique)
        self.assertEqual(one.loc[one["t_fs"] == 100.0, "phase"].tolist(), ["equilibration"])
        self.assertEqual(float(one["t_fs"].max()), 300.0)

    def test_one_row_per_trajectory_per_window_without_leaking_columns(self):
        write_run(self.root / "run", [1, 2, 3])
        frames, _ = sm.load_frames([self.root / "run"], "ZnO", SITES, 1.0)
        edges = sm.window_edges(frames, 40.0, None)
        dataset = sm.window_dataset(frames, sm.FEATURE_SETS["geometry3"], edges, SITES)
        self.assertFalse(dataset.duplicated(["window", "site", "seed"]).any())
        self.assertEqual(len(dataset), len(edges) * 9)
        self.assertFalse(set(sm.LEAKING_COLUMNS) & set(dataset.columns))
        with self.assertRaises(ValueError):
            sm.window_dataset(frames, ["nearest_site_lateral_ang"], edges, SITES)

    def test_runs_with_other_md_settings_are_not_combined(self):
        write_run(self.root / "a", [1, 2], temperature=300.0)
        write_run(self.root / "b", [3, 4], temperature=100.0)
        with self.assertRaisesRegex(SystemExit, "temperature_kelvin"):
            sm.common_settings([self.root / "a", self.root / "b"])

    def test_the_same_seed_in_two_runs_is_refused(self):
        write_run(self.root / "a", [1, 2])
        write_run(self.root / "b", [2, 3])
        with self.assertRaisesRegex(SystemExit, "more than one run"):
            sm.load_frames([self.root / "a", self.root / "b"], "ZnO", SITES, 1.0)

    def test_missing_frame_analysis_names_the_command(self):
        (self.root / "run").mkdir()
        (self.root / "run" / "config.yaml").write_text(
            "methods:\n  md:\n    temperature_kelvin: 300\n    timestep_fs: 1\n    log_interval: 20\n"
            "    equilibration_steps: 100\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "md_trajectory_analysis"):
            sm.load_frames([self.root / "run"], "ZnO", SITES, 1.0)


class SiteMemoryStatisticsTests(unittest.TestCase):
    def test_within_seed_relabelings_are_exact_and_keep_one_site_per_seed(self):
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        seeds = np.repeat([1, 2, 3], 3)
        labelings, exact = sm.within_seed_relabelings(y, seeds, 1000, np.random.default_rng(0))
        self.assertTrue(exact)
        self.assertEqual(len(labelings), 6 ** 3 - 1)
        for labels in labelings:
            for seed in (1, 2, 3):
                self.assertEqual(sorted(labels[seeds == seed]), [0, 1, 2])

    def test_many_seeds_are_sampled(self):
        y = np.tile([0, 1, 2], 6)
        seeds = np.repeat(np.arange(6), 3)
        labelings, exact = sm.within_seed_relabelings(y, seeds, 50, np.random.default_rng(0))
        self.assertFalse(exact)
        self.assertEqual(len(labelings), 50)

    def test_memory_time_is_the_first_significant_run(self):
        windows = pd.DataFrame({"model": "svm_rbf", "t_start_fs": [0, 40, 80, 120],
                                "t_end_fs": [40, 80, 120, 160], "p_value": [0.02, 0.03, 0.4, 0.01]})
        self.assertEqual(sm.memory_time(windows, "svm_rbf", 0.05), 80.0)
        windows.loc[0, "p_value"] = 0.5
        self.assertEqual(sm.memory_time(windows, "svm_rbf", 0.05), 0.0)

    def test_scaling_uses_the_training_rows_only(self):
        train = np.array([[1.0, 5.0], [3.0, 5.0]])
        test = np.array([[5.0, 7.0]])
        a, b = sm.scale_features(train, test)
        np.testing.assert_allclose(a, [[0, 0], [np.pi / 2, 0]])
        # Column 0: (5 - 1) / 2 * pi/2 = pi; constant column 1: span 1, (7 - 5) * pi/2 = pi.
        np.testing.assert_allclose(b, [[np.pi, np.pi]])


class QiskitKernelTests(unittest.TestCase):
    def test_statevector_is_the_default_and_needs_qiskit(self):
        args = sm.parse_args(["--runs", "x", "--output-dir", "y"])
        self.assertEqual(args.qsvm_kernel, "statevector")
        try:
            import qiskit_machine_learning  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(SystemExit, "qiskit"):
                sm.make_qsvm_kernel(args, 3)
        else:
            X = np.random.default_rng(4).uniform(0, np.pi / 2, (6, 3))
            np.testing.assert_allclose(sm.make_qsvm_kernel(args, 3)(X), zz_kernel(X), atol=1e-8)


class SiteMemoryEndToEndTests(unittest.TestCase):
    def test_memory_is_detected_and_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "run", [1, 2, 3], memory_fs=100.0)
            out = root / "out"
            sm.main(["--runs", str(root / "run"), "--output-dir", str(out), "--window-fs", "40",
                     "--models", "majority", "svm_rbf", "qsvm", "--qsvm-kernel", "zz_exact",
                     "--n-permutations", "300"])
            windows = pd.read_csv(out / "site_memory_windows.csv")
            summary = json.loads((out / "site_memory_summary.json").read_text())
            for name in ("site_memory_predictions.csv", "site_memory_features.csv", "site_memory_curve.png"):
                self.assertTrue((out / name).is_file())
            first = windows[(windows["window"] == 0) & (windows["model"] == "svm_rbf")].iloc[0]
            self.assertEqual(int(first["n_correct"]), 9)
            self.assertAlmostEqual(float(first["p_value"]), 6 / 216)  # 3! renamings of 216 labelings
            self.assertTrue(bool(first["exact_null"]))
            self.assertGreaterEqual(summary["memory_time_fs"]["svm_rbf"], 80.0)
            self.assertLessEqual(summary["memory_time_fs"]["svm_rbf"], 160.0)
            majority = windows[windows["model"] == "majority"]
            self.assertTrue(np.allclose(majority["balanced_accuracy"], 1 / 3))
            self.assertTrue(majority["p_value"].isna().all())
            self.assertEqual(summary["seeds"], [1, 2, 3])
            self.assertTrue(math.isclose(summary["chance_level"], 1 / 3))


if __name__ == "__main__":
    unittest.main()
