"""Grouped train/test splits and CV for the SVM, QSVM, IBM QSVM and benchmark scripts.

qiskit is not required: the quantum scripts import it lazily, so their data
loading and splitting are tested here without building circuits.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.models import grouped_split as gs
from scripts.models.benchmarks import run_benchmarks
from scripts.models.ml import classical_svm_baseline as svm
from scripts.models.qml import qsvm_baseline as qsvm
from scripts.models.qml_ibm import qsvm_ibm_hardware as ibm

SITES = ("top_metal", "top_oxygen", "bridge")
FEATURES = svm.FEATURE_SETS["md3"]


def window_dataset(seeds_per_site: int, windows: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, site in enumerate(SITES):
        for trajectory in range(1, seeds_per_site + 1):
            offset = rng.normal(scale=0.2)
            for window in range(windows):
                rows.append({
                    "material": "ZnO",
                    "site": site,
                    "group": f"ZnO/{site}/seed_{trajectory}",
                    "md_window_id": window,
                    FEATURES[0]: 2.5 + label + offset + rng.normal(scale=0.05),
                    FEATURES[1]: 0.1 + 0.02 * rng.random(),
                    FEATURES[2]: 175 - 2 * label + rng.normal(),
                    "label_site": site,
                    "label_site_id": label,
                })
    return pd.DataFrame(rows)


def write_dataset(directory: Path, seeds_per_site: int, **kwargs) -> Path:
    path = directory / "windows.csv"
    window_dataset(seeds_per_site, **kwargs).to_csv(path, index=False)
    return path


def split_args(csv_path: Path, **overrides) -> SimpleNamespace:
    values = dict(csv=csv_path, label="label_site_id", group_column="group", test_size=0.3,
                  max_samples=0, random_state=42, split_file=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class GroupedSplitTests(unittest.TestCase):
    def test_holdout_never_shares_groups_and_keeps_every_class(self):
        df = window_dataset(seeds_per_site=4).assign(_row_id=lambda d: np.arange(len(d)))
        for seed in range(20):
            train, test = gs.grouped_train_test_split(df, "label_site_id", "group", 0.3, 0, seed)
            self.assertFalse(set(train["group"]) & set(test["group"]))
            self.assertEqual(set(test["label_site_id"]), {0, 1, 2})
            self.assertEqual(set(train["label_site_id"]), {0, 1, 2})
            self.assertEqual(len(train) + len(test), len(df))

    def test_subsample_spreads_over_groups(self):
        df = window_dataset(seeds_per_site=3).assign(_row_id=lambda d: np.arange(len(d)))
        sampled = gs.group_balanced_subsample(df, "label_site_id", "group", 12, 0)
        self.assertEqual(len(sampled), 12)
        self.assertTrue((sampled.groupby("label_site_id")["group"].nunique() == 3).all())

    def test_one_trajectory_per_site_fails_with_clear_message(self):
        df = window_dataset(seeds_per_site=1).assign(_row_id=lambda d: np.arange(len(d)))
        with self.assertRaises(gs.InsufficientGroupsError) as context:
            gs.grouped_train_test_split(df, "label_site_id", "group", 0.3, 0, 0)
        message = str(context.exception)
        self.assertIn("at least 2 independent trajectories", message)
        self.assertIn("seed", message)

    def test_cv_folds_never_share_groups(self):
        df = window_dataset(seeds_per_site=5).assign(_row_id=lambda d: np.arange(len(d)))
        train, _ = gs.grouped_train_test_split(df, "label_site_id", "group", 0.3, 0, 1)
        splitter, folds = gs.grouped_cv_splitter(train, "label_site_id", "group", 5, 1)
        self.assertEqual(folds, int(train.groupby("label_site_id")["group"].nunique().min()))
        for tr, va in splitter.split(train[FEATURES], train["label_site_id"], train["group"]):
            self.assertFalse(set(train["group"].iloc[tr]) & set(train["group"].iloc[va]))

    def test_cv_with_one_training_group_per_class_fails(self):
        df = window_dataset(seeds_per_site=2).assign(_row_id=lambda d: np.arange(len(d)))
        train, _ = gs.grouped_train_test_split(df, "label_site_id", "group", 0.3, 0, 1)
        with self.assertRaisesRegex(gs.InsufficientGroupsError, "at least 3 trajectories per site"):
            gs.grouped_cv_splitter(train, "label_site_id", "group", 5, 1)
        self.assertEqual(gs.grouped_cv_splitter(train, "label_site_id", "group", 0, 1), (None, 0))

    def test_dataset_without_group_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            window_dataset(3).drop(columns=["group"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "no 'group' column"):
                gs.load_grouped_dataset(path, FEATURES, "label_site_id", "group")

    def test_leaking_split_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_dataset(Path(tmp), 3)
            df = gs.load_grouped_dataset(csv_path, FEATURES, "label_site_id", "group")
            split = pd.DataFrame({"row_id": df["_row_id"],
                                  "split": np.where(df["_row_id"] % 2 == 0, "train", "test")})
            split_path = Path(tmp) / "split.csv"
            split.to_csv(split_path, index=False)
            with self.assertRaisesRegex(ValueError, "both train and test"):
                gs.read_split_file(split_path, df, "label_site_id", "group")


class ModelScriptTests(unittest.TestCase):
    def test_every_script_splits_by_group_without_qiskit(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_dataset(Path(tmp), 3)
            for module in (svm, qsvm, ibm):
                with self.subTest(module=module.__name__):
                    args = split_args(csv_path)
                    df = module.load_base_data(args, FEATURES)
                    train, test = module.split_data(args, df)
                    self.assertFalse(set(train["group"]) & set(test["group"]))

    def test_every_script_refuses_one_trajectory_per_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_dataset(Path(tmp), 1)
            for module in (svm, qsvm, ibm):
                with self.subTest(module=module.__name__):
                    args = split_args(csv_path)
                    df = module.load_base_data(args, FEATURES)
                    with self.assertRaises(gs.InsufficientGroupsError):
                        module.split_data(args, df)

    def test_classical_svm_end_to_end_uses_grouped_split_and_cv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = write_dataset(root, 4)
            argv = ["svm", "--csv", str(csv_path), "--feature-set", "md3", "--cv", "3",
                    "--output-dir", str(root / "svm"), "--run-name", "check"]
            with patch.object(sys, "argv", argv), patch("builtins.print"):
                svm.main()
            run_dir = root / "svm" / "check"
            split = pd.read_csv(run_dir / "split.csv")
            train_groups = set(split.loc[split["split"] == "train", "group"])
            test_groups = set(split.loc[split["split"] == "test", "group"])
            self.assertFalse(train_groups & test_groups)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(metrics["split_strategy"], "grouped")
            self.assertEqual(metrics["grouping"]["test_groups"], len(test_groups))
            self.assertGreaterEqual(metrics["cv_folds"], 2)

    def test_benchmark_shared_split_is_grouped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = write_dataset(root, 3)
            path = run_benchmarks.create_split(csv_path, "label_site_id", "md3", 12, 0.3, 42, root / "splits")
            split = pd.read_csv(path)
            self.assertEqual(len(split), 12)
            train_groups = set(split.loc[split["split"] == "train", "group"])
            test_groups = set(split.loc[split["split"] == "test", "group"])
            self.assertFalse(train_groups & test_groups)

            (root / "single").mkdir()
            single = write_dataset(root / "single", 1)
            with self.assertRaises(gs.InsufficientGroupsError):
                run_benchmarks.create_split(single, "label_site_id", "md3", 0, 0.3, 42, root / "splits")

    def test_feature_sets_match_across_scripts(self):
        for module in (qsvm, ibm, run_benchmarks):
            self.assertEqual(module.FEATURE_SETS, svm.FEATURE_SETS)


if __name__ == "__main__":
    unittest.main()
