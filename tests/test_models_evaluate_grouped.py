"""Grouped (trajectory-level) evaluation protocol shared by SVM, QSVM and baselines.

Runs without qiskit: the QSVM path is exercised with the classically computable
product-angle fidelity kernel and with a 'module:callable' plugin kernel.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.models import evaluate_grouped as eg
from scripts.models import grouped_split as gs
from scripts.models.benchmarks import run_benchmarks
from scripts.models.ml import classical_svm_baseline as svm
from scripts.models.qml_ibm import qsvm_ibm_hardware as ibm
from scripts.workflows import run_models

SITES = ("top_metal", "top_oxygen", "bridge")
FEATURES = svm.FEATURE_SETS["md3"]
LABEL, GROUP = "label_site_id", "group"


def grouped_dataset(groups_per_class: int = 3, windows: int = 5, seed: int = 0,
                    separation: float = 1.0) -> pd.DataFrame:
    """3 classes x n trajectories; each trajectory has its own offset (pseudo-replicates)."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, site in enumerate(SITES):
        for trajectory in range(1, groups_per_class + 1):
            offset = rng.normal(scale=0.3, size=3)
            for window in range(windows):
                rows.append({
                    "group": f"ZnO/{site}/seed_{trajectory}",
                    "md_window_id": window,
                    FEATURES[0]: 2.5 + separation * label + offset[0] + rng.normal(scale=0.05),
                    FEATURES[1]: 0.1 + 0.05 * offset[1] + 0.01 * rng.random(),
                    FEATURES[2]: 175 - 2 * separation * label + offset[2] + rng.normal(scale=0.3),
                    "label_site": site,
                    "label_site_id": label,
                })
    return pd.DataFrame(rows)


def settings(**overrides) -> eg.ProtocolSettings:
    values = dict(features=list(FEATURES), label=LABEL, group=GROUP, outer="both",
                  preprocessing="minmax_pi", tuning="fixed", n_permutations=0,
                  permutation_models=None, random_state=0)
    values.update(overrides)
    return eg.ProtocolSettings(**values)


class OuterSplitTests(unittest.TestCase):
    def setUp(self):
        self.df = grouped_dataset()
        self.group_labels = self.df.groupby(GROUP)[LABEL].first()

    def test_27_one_per_class_splits_and_9_loto(self):
        splits = eg.outer_splits(self.group_labels, "one_per_class")
        self.assertEqual(len(splits), 27)
        self.assertEqual(len(set(splits)), 27)
        for test_groups in splits:
            self.assertEqual(sorted(self.group_labels[list(test_groups)].tolist()), [0, 1, 2])
        self.assertEqual(len(eg.outer_splits(self.group_labels, "loto")), 9)

    def test_general_product_of_groups_per_class(self):
        df = pd.concat([grouped_dataset(2), grouped_dataset(4).query("label_site_id == 2")])
        labels = df.groupby(GROUP)[LABEL].first()
        self.assertEqual(len(eg.outer_splits(labels, "one_per_class")), 2 * 2 * 4)

    def test_one_trajectory_per_class_is_refused(self):
        labels = grouped_dataset(1).groupby(GROUP)[LABEL].first()
        with self.assertRaises(gs.InsufficientGroupsError):
            eg.outer_splits(labels, "one_per_class")

    def test_no_group_in_train_and_test(self):
        specs = eg.build_model_specs(["nearest_centroid"])
        per_split, _ = eg.run_protocol(self.df, specs, settings())
        self.assertEqual(per_split.groupby("scheme").size().to_dict(), {"one_per_class": 27, "loto": 9})
        for _, row in per_split.iterrows():
            test = set(row["test_groups"].split("|"))
            train = set(self.df[GROUP]) - test
            self.assertFalse(test & train)
            self.assertEqual(row["n_test_rows"], int(self.df[GROUP].isin(test).sum()))
            self.assertEqual(row["n_train_groups"], len(train))

    def test_inner_folds_stay_inside_training_trajectories(self):
        y = self.df[LABEL].to_numpy()
        groups = self.df[GROUP].to_numpy()
        test_groups = eg.outer_splits(self.group_labels, "one_per_class")[0]
        train_idx = np.flatnonzero(~np.isin(groups, test_groups))
        folds = eg.inner_folds(y, groups, train_idx, 0)
        self.assertEqual(len(folds), 2)  # 2 training trajectories per class
        for tr, va in folds:
            self.assertTrue(set(tr) <= set(train_idx) and set(va) <= set(train_idx))
            self.assertFalse(set(groups[tr]) & set(groups[va]))
            self.assertFalse(set(groups[va]) & set(test_groups))


class LeakageTests(unittest.TestCase):
    def test_preprocessing_is_fitted_on_training_rows_only(self):
        df = grouped_dataset(windows=3)
        df["row_marker"] = np.arange(len(df), dtype=float)
        group_of = dict(zip(df["row_marker"], df[GROUP]))
        records = []

        class Spy:
            def fit_transform(self, X):
                self.fitted = set(np.asarray(X)[:, 0])
                return np.asarray(X, dtype=float)

            def transform(self, X):
                records.append((self.fitted, set(np.asarray(X)[:, 0])))
                return np.asarray(X, dtype=float)

        with patch.object(eg, "make_preprocessor", lambda name: Spy()):
            specs = eg.build_model_specs(["svm_rbf"], param_grid={"svm_rbf": {"C": [1.0, 10.0], "gamma": ["scale"]}})
            eg.run_protocol(df, specs, settings(features=["row_marker", *FEATURES], tuning="nested"))
        self.assertTrue(records)
        for fitted, evaluated in records:
            fitted_groups = {group_of[v] for v in fitted}
            evaluated_groups = {group_of[v] for v in evaluated}
            self.assertFalse(fitted_groups & evaluated_groups)

    def test_tuning_never_sees_the_outer_test_fold(self):
        df = grouped_dataset(windows=3)
        groups = df[GROUP].to_numpy()
        calls = []
        original = eg.fit_predict

        def spy(spec, params, cache, y, train_idx, eval_idx):
            calls.append((set(groups[train_idx]), set(groups[eval_idx])))
            return original(spec, params, cache, y, train_idx, eval_idx)

        specs = eg.build_model_specs(["logistic"])
        with patch.object(eg, "fit_predict", spy):
            per_split, _ = eg.run_protocol(df, specs, settings(tuning="nested", outer="one_per_class"))
        tests = [set(t.split("|")) for t in per_split["test_groups"]]
        k = 0
        for train_groups, eval_groups in calls:
            self.assertFalse(train_groups & tests[k])
            if eval_groups == tests[k]:
                k += 1  # final outer fit of this split
            else:
                self.assertFalse(eval_groups & tests[k])
        self.assertEqual(k, 27)
        # Inner calls happened: 3 C values x 2 inner folds per split.
        self.assertEqual(len(calls), 27 * (3 * 2 + 1))


class PermutationTests(unittest.TestCase):
    def setUp(self):
        self.df = grouped_dataset()
        self.group_labels = self.df.groupby(GROUP)[LABEL].first()

    def test_group_relabelings_keep_class_sizes_and_are_exhaustive(self):
        self.assertEqual(eg.count_distinct_relabelings(self.group_labels), 1680)
        self.assertEqual(eg.count_distinct_partitions(self.group_labels), 280)
        labelings, exact = eg.group_relabelings(self.group_labels, 1000, np.random.default_rng(0))
        self.assertTrue(exact)
        self.assertEqual(len(labelings), 279)
        keys = set()
        for labeling in labelings:
            self.assertEqual(sorted(labeling.value_counts().tolist()), [3, 3, 3])
            self.assertEqual(list(labeling.index), list(self.group_labels.index))
            keys.add(eg._partition_key(list(labeling.index), tuple(labeling)))
        self.assertEqual(len(keys), 279)
        self.assertNotIn(eg._partition_key(list(self.group_labels.index), tuple(self.group_labels)), keys)

    def test_sampled_relabelings_when_enumeration_exceeds_budget(self):
        labelings, exact = eg.group_relabelings(self.group_labels, 50, np.random.default_rng(0))
        self.assertFalse(exact)
        self.assertEqual(len(labelings), 50)
        for labeling in labelings:
            self.assertEqual(sorted(labeling.value_counts().tolist()), [3, 3, 3])

    def test_permute_group_labels_never_splits_a_group(self):
        for seed in range(10):
            permuted = gs.permute_group_labels(self.df, LABEL, GROUP, seed)
            self.assertTrue((permuted.groupby(GROUP)[LABEL].nunique() == 1).all())
            self.assertEqual(permuted.groupby(LABEL)[GROUP].nunique().tolist(), [3, 3, 3])
            self.assertEqual(permuted[LABEL].value_counts().sort_index().tolist(),
                             self.df[LABEL].value_counts().sort_index().tolist())

    def test_p_value_formula(self):
        self.assertAlmostEqual(eg.permutation_p_value(0.9, [0.1, 0.95, 0.9, 0.2]), 3 / 5)
        self.assertAlmostEqual(eg.permutation_p_value(1.0, np.zeros(279)), 1 / 280)
        self.assertAlmostEqual(eg.permutation_p_value(0.0, np.zeros(9)), 1.0)

    def test_permutation_test_reruns_protocol_and_detects_signal(self):
        specs = eg.build_model_specs(["nearest_centroid"])
        result = eg.evaluate(self.df, specs, settings(n_permutations=1000))
        perm = result["summary"]["permutation"]["nearest_centroid"]
        self.assertTrue(perm["exact_enumeration"])
        self.assertEqual(perm["n_permutations"], 279)
        self.assertGreater(perm["observed"], 0.7)
        self.assertLess(perm["null_mean"], 0.6)
        self.assertLess(perm["p_value"], 0.05)
        self.assertEqual(len(result["null"]), 279)

    def test_parallel_permutations_match_serial(self):
        specs = eg.build_model_specs(["nearest_centroid"])
        serial = eg.evaluate(self.df, specs, settings(n_permutations=20, outer="one_per_class"))
        parallel = eg.evaluate(self.df, specs, settings(n_permutations=20, outer="one_per_class", n_jobs=2))
        pd.testing.assert_frame_equal(serial["null"].reset_index(drop=True),
                                      parallel["null"].reset_index(drop=True))


class StatisticsTests(unittest.TestCase):
    def test_clopper_pearson_reference_values(self):
        lower, upper = eg.clopper_pearson(9, 9)
        self.assertAlmostEqual(lower, 0.025 ** (1 / 9), places=6)
        self.assertAlmostEqual(lower, 0.6637, places=3)
        self.assertEqual(upper, 1.0)
        lower, upper = eg.clopper_pearson(6, 9)
        self.assertAlmostEqual(lower, 0.2993, places=3)
        self.assertAlmostEqual(upper, 0.9251, places=3)
        self.assertEqual(eg.clopper_pearson(0, 9)[0], 0.0)

    def test_majority_baseline_is_at_chance(self):
        specs = eg.build_model_specs(["majority"])
        result = eg.evaluate(grouped_dataset(), specs, settings())
        entry = result["summary"]["models"]["majority"]["one_per_class"]
        self.assertAlmostEqual(entry["trajectory_balanced_accuracy_mean"], 1 / 3)
        self.assertAlmostEqual(entry["window_balanced_accuracy_mean"], 1 / 3)
        self.assertAlmostEqual(result["summary"]["chance_level"], 1 / 3)

    def test_balanced_accuracy_matches_sklearn(self):
        from sklearn.metrics import balanced_accuracy_score
        rng = np.random.default_rng(0)
        for _ in range(20):
            y_true, y_pred = rng.integers(0, 3, 15), rng.integers(0, 3, 15)
            self.assertAlmostEqual(eg.balanced_accuracy(y_true, y_pred),
                                   balanced_accuracy_score(y_true, y_pred))

    def test_trajectory_vote_tie_counts_as_wrong(self):
        votes = eg.trajectory_votes(np.array(["a", "a", "b", "b", "b"]), np.array([0, 1, 1, 1, 0]))
        self.assertIs(votes["a"], eg.TIE_VOTE)
        self.assertEqual(votes["b"], 1)
        ba, correct, n = eg.trajectory_balanced_accuracy({"a": 0, "b": 1}, votes)
        self.assertEqual((ba, correct, n), (0.5, 1, 2))

    def test_paired_tests(self):
        diff = np.array([1 / 3, 0, 0, 1 / 3, -1 / 3, 1 / 3])
        sign = eg.sign_test(diff)
        self.assertEqual((sign["n_nonzero"], sign["n_positive"]), (4, 3))
        self.assertAlmostEqual(sign["p_two_sided"], 0.625)
        flip = eg.paired_sign_flip_test(diff, np.random.default_rng(0))
        self.assertTrue(flip["exact"])
        self.assertAlmostEqual(flip["p_two_sided"], 10 / 16)
        mcnemar = eg.mcnemar_exact(np.array([True] * 9), np.array([True] * 6 + [False] * 3))
        self.assertEqual((mcnemar["a_correct_b_wrong"], mcnemar["a_wrong_b_correct"]), (3, 0))
        self.assertAlmostEqual(mcnemar["p_two_sided"], 0.25)


class KernelAndQsvmTests(unittest.TestCase):
    def test_product_angle_kernel_is_a_fidelity(self):
        X = np.random.default_rng(0).uniform(0, np.pi, size=(5, 3))
        K = eg.product_angle_kernel(X)
        np.testing.assert_allclose(np.diag(K), 1.0)
        np.testing.assert_allclose(K, K.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-10)
        self.assertEqual(eg.product_angle_kernel(X[:2], X).shape, (2, 5))

    def test_plugin_kernel_and_qsvm_vs_svm_comparison(self):
        kernel = eg.resolve_kernel("scripts.models.evaluate_grouped:product_angle_kernel", 3)
        specs = eg.build_model_specs(["svm_rbf", "qsvm"], kernel=kernel)
        result = eg.evaluate(grouped_dataset(), specs, settings(compare=[("qsvm", "svm_rbf")]))
        comparison = result["summary"]["comparisons"]["qsvm_vs_svm_rbf"]
        self.assertIn("Low power", comparison["power_note"])
        self.assertIn("loto_mcnemar_exact", comparison)
        self.assertEqual(len(result["differences"]), 2 * (27 + 9))

    def test_qiskit_is_imported_lazily(self):
        try:
            import qiskit  # noqa: F401
        except ImportError:
            self.assertNotIn("qiskit", sys.modules)
            with self.assertRaises(ImportError):
                eg.resolve_kernel("statevector", 3)
        else:
            self.skipTest("qiskit is installed; laziness is only checked without it")


class CommandLineTests(unittest.TestCase):
    def test_end_to_end_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "windows.csv"
            grouped_dataset(windows=4).to_csv(csv_path, index=False)
            argv = ["--csv", str(csv_path), "--models", "majority", "nearest_centroid", "svm_rbf", "qsvm",
                    "--qsvm-kernel", "product_angle", "--tuning", "fixed", "--preprocessing", "minmax_pi",
                    "--n-permutations", "5", "--output-dir", str(root / "out"), "--run-name", "check"]
            with patch("builtins.print"):
                run_dir = eg.main(argv)
            for name in ("per_split.csv", "trajectory_predictions.csv", "permutation_null.csv",
                         "paired_differences.csv", "summary.json", "report.md"):
                self.assertTrue((run_dir / name).is_file(), name)
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["data"]["sha256"], eg.file_sha256(csv_path))
            self.assertEqual(summary["data"]["groups_per_class"], {"0": 3, "1": 3, "2": 3})
            self.assertEqual(summary["protocol"]["n_outer_splits"], {"one_per_class": 27, "loto": 9})
            self.assertEqual(summary["protocol"]["permutation_models"], ["svm_rbf", "qsvm"])
            self.assertEqual(sorted(summary["permutation"]), ["qsvm", "svm_rbf"])
            self.assertEqual(summary["permutation"]["qsvm"]["n_permutations"], 5)
            self.assertIn("qsvm_vs_svm_rbf", summary["comparisons"])
            loto = summary["models"]["svm_rbf"]["loto_pooled"]
            self.assertEqual(loto["trajectories"], 9)
            per_split = pd.read_csv(run_dir / "per_split.csv")
            self.assertEqual(len(per_split), 4 * (27 + 9))
            self.assertIn("Clopper-Pearson", (run_dir / "report.md").read_text())


class ScriptFixTests(unittest.TestCase):
    def test_split_file_with_mismatched_groups_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "windows.csv"
            grouped_dataset().to_csv(csv_path, index=False)
            df = gs.load_grouped_dataset(csv_path, FEATURES, LABEL, GROUP)
            train, test = gs.grouped_train_test_split(df, LABEL, GROUP, 0.3, 0, 0)
            split = gs.split_table(train, test, LABEL, GROUP)
            good = Path(tmp) / "good.csv"
            split.to_csv(good, index=False)
            gs.read_split_file(good, df, LABEL, GROUP)  # accepted

            bad = split.copy()
            bad.loc[0, "group"] = "ZnO/other/seed_9"
            bad_path = Path(tmp) / "bad.csv"
            bad.to_csv(bad_path, index=False)
            with self.assertRaisesRegex(ValueError, "does not match the dataset"):
                gs.read_split_file(bad_path, df, LABEL, GROUP)

            missing = Path(tmp) / "missing.csv"
            split.drop(columns=["group"]).to_csv(missing, index=False)
            with self.assertRaisesRegex(ValueError, "must contain columns"):
                gs.read_split_file(missing, df, LABEL, GROUP)

    def test_shuffle_label_permutes_training_groups_and_keeps_test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "windows.csv"
            data = grouped_dataset(groups_per_class=4)
            data.to_csv(csv_path, index=False)
            argv = ["svm", "--csv", str(csv_path), "--cv", "0", "--shuffle-label",
                    "--output-dir", str(root / "svm"), "--run-name", "shuffled"]
            with patch.object(sys, "argv", argv), patch("builtins.print"):
                svm.main()
            run_dir = root / "svm" / "shuffled"
            predictions = pd.read_csv(run_dir / "predictions.csv")
            truth = data[LABEL].to_numpy()[predictions["row_id"].to_numpy()]
            np.testing.assert_array_equal(predictions["true_label"].to_numpy(), truth)
            split = pd.read_csv(run_dir / "split.csv")
            train = split[split["split"] == "train"]
            self.assertTrue((train.groupby("group")["true_label"].nunique() == 1).all())
            self.assertTrue(json.loads((run_dir / "metrics.json").read_text())["shuffle_label"])

    def test_run_benchmarks_defaults(self):
        with patch.object(sys, "argv", ["run_benchmarks"]):
            args = run_benchmarks.parse_args()
        self.assertEqual(args.max_samples, 0)
        self.assertEqual(args.ibm_max_samples, 12)
        self.assertEqual(args.svm_cv, args.qsvm_cv)
        self.assertEqual(args.svm_scalers, args.qsvm_scalers)
        self.assertEqual(args.ibm_kernel_diagonal, "ideal")

    def test_select_by_cv_ignores_test_scores(self):
        summary = pd.DataFrame({
            "requested_model": ["svm", "svm", "qsvm"],
            "feature_set": ["md3"] * 3,
            "cv_balanced_accuracy_mean": [0.9, 0.5, None],
            "test_balanced_accuracy": [0.2, 1.0, 1.0],
        })
        selected = run_benchmarks.select_by_cv(summary)
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected.iloc[0]["test_balanced_accuracy"], 0.2)

    def test_ibm_kernel_diagonal_modes(self):
        K_train = np.array([[1.0, 0.4], [0.4, 1.0]])
        K_test = np.array([[0.3, 0.5]])
        same = ibm.apply_kernel_diagonal(K_train, K_test, "ideal")
        np.testing.assert_array_equal(same[0], K_train)
        np.testing.assert_array_equal(same[1], K_test)
        measured, test = ibm.apply_kernel_diagonal(K_train, K_test, "measured", np.array([0.8, 0.9]))
        np.testing.assert_allclose(np.diag(measured), [0.8, 0.9])
        np.testing.assert_array_equal(test, K_test)
        norm_train, norm_test = ibm.apply_kernel_diagonal(
            K_train, K_test, "measured_normalized", np.array([0.8, 0.9]), np.array([0.5]))
        np.testing.assert_allclose(np.diag(norm_train), 1.0)
        self.assertAlmostEqual(norm_train[0, 1], 0.4 / np.sqrt(0.8 * 0.9))
        np.testing.assert_allclose(norm_test, [[0.3 / np.sqrt(0.5 * 0.8), 0.5 / np.sqrt(0.5 * 0.9)]])
        self.assertEqual(ibm.estimate_kernel_entries(4, 2, "measured_normalized")["self_fidelity_circuits"], 6)
        self.assertEqual(ibm.estimate_kernel_entries(4, 2)["approx_total_fidelity_entries"], 6 + 8)

    def test_run_models_builds_grouped_protocol_command(self):
        config = run_models.load_config(Path("configs/models/zno_local.yaml"))
        self.assertEqual(run_models.evaluation_protocol(config, None), "grouped_27")
        self.assertEqual(run_models.evaluation_protocol(config, "holdout"), "holdout")
        command = run_models.build_evaluation_command(config, ["svm", "qsvm"], Path("d.csv"), Path("/tmp/run"))
        self.assertIn(run_models.EVALUATION_MODULE, command)
        models = command[command.index("--models") + 1:command.index("--outer")]
        self.assertIn("qsvm", models)
        self.assertIn("qsvm:svm_rbf", command)
        svm_only = run_models.build_evaluation_command(config, ["svm"], Path("d.csv"), Path("/tmp/run"))
        self.assertNotIn("qsvm", svm_only[svm_only.index("--models") + 1:svm_only.index("--outer")])
        self.assertNotIn("qsvm:svm_rbf", svm_only)
        benchmark = run_models.build_command(config, ["svm", "qsvm"], Path("d.csv"), Path("/tmp/run"),
                                             False, False, None)
        self.assertEqual(benchmark[benchmark.index("--max-samples") + 1], "0")
        self.assertEqual(benchmark[benchmark.index("--ibm-max-samples") + 1], "12")
        self.assertEqual(benchmark[benchmark.index("--svm-cv") + 1], benchmark[benchmark.index("--qsvm-cv") + 1])
        ibm_plan = run_models.load_config(Path("configs/models/zno_ibm_plan.yaml"))
        self.assertEqual(run_models.evaluation_protocol(ibm_plan, None), "holdout")


if __name__ == "__main__":
    unittest.main()
