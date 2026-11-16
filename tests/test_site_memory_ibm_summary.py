"""site_memory_ibm_summary: tables from saved hardware runs, repeats, skipped plan and reuse folders."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts.analysis import site_memory_ibm as smi
from scripts.analysis import site_memory_ibm_summary as summary
from scripts.models.qml.zz_kernel import zz_kernel
from scripts.models.qml_ibm import qsvm_ibm_hardware as hw
from site_memory_fixtures import write_run

COMMON = ["--window-starts-fs", "0", "200", "--ideal-kernel", "zz_exact", "--n-permutations", "50"]


def noisy_train(scale):
    def train(X, *args, **kwargs):
        raw = zz_kernel(X) - scale * np.cos(np.add.outer(np.arange(len(X)), np.arange(len(X))) ** 2)
        np.fill_diagonal(raw, 1.0)
        stats = {"n_circuits": len(X) * (len(X) - 1) // 2, "depth_mean": 28.5, "two_qubit_gates_mean": 6.0}
        return raw, raw, stats, {}, [{"job_id": f"job-{scale}", "quantum_seconds": 12.0}]
    return train


class IbmSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_run(self.root / "run", [1, 2, 3], memory_fs=100.0)
        self.run = str(self.root / "run")
        self.out = self.root / "site_memory"

    def tearDown(self):
        self.tmp.cleanup()

    def hardware_run(self, name, scale, extra=()):
        patches = [
            mock.patch.object(hw, "create_runtime_service", return_value=object()),
            mock.patch.object(hw, "pick_backend", return_value=type("B", (), {"name": "ibm_marrakesh"})()),
            mock.patch.object(hw, "build_hardware_executor", return_value=(object(), object())),
            mock.patch.object(hw, "evaluate_hardware_kernel_train", side_effect=noisy_train(scale)),
            mock.patch("scripts.models.qml.qsvm_baseline.build_feature_map", return_value=(None, "fake")),
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            smi.main(["--runs", self.run, "--output-dir", str(self.out / name), "--submit", *COMMON, *extra])

    def test_summary_repeats_and_skipped_folders(self):
        self.hardware_run("ibm_a", 0.01)
        self.hardware_run("ibm_a_rep", 0.02)
        self.hardware_run("ibm_s256", 0.01, ["--shots", "256"])
        smi.main(["--runs", self.run, "--output-dir", str(self.out / "ibm_plan"), *COMMON])  # plan only
        with mock.patch.object(hw, "create_runtime_service", side_effect=AssertionError("network")):
            smi.main(["--runs", self.run, "--output-dir", str(self.out / "ibm_a_reuse"), "--reuse-hardware",
                      str(self.out / "ibm_a" / smi.HARDWARE_DIR), *COMMON])
            with mock.patch("builtins.print"):
                summary.main(["--root", str(self.out), "--output-dir", str(self.root / "summary")])

        table = pd.read_csv(self.root / "summary" / "ibm_summary.csv")
        self.assertEqual(sorted(set(table["folder"])), ["ibm_a", "ibm_a_rep", "ibm_s256"])
        self.assertEqual(len(table), 3 * 2)
        self.assertEqual(set(table["bandwidth"]), {"A"})
        self.assertEqual(set(table["backend"]), {"ibm_marrakesh"})
        self.assertTrue((table["quantum_seconds"] == 12.0).all())
        self.assertTrue((table["depth_mean"] == 28.5).all())
        early = table[(table["t_start_fs"] == 0) & (table["folder"] == "ibm_a")].iloc[0]
        self.assertEqual(int(early["svm_rbf_correct"]), 9)

        repeats = pd.read_csv(self.root / "summary" / "ibm_repeats.csv")
        self.assertEqual(len(repeats), 2)  # ibm_a vs ibm_a_rep, one row per window
        self.assertEqual(set(repeats["folder_a"]) | set(repeats["folder_b"]), {"ibm_a", "ibm_a_rep"})
        self.assertTrue((repeats["kernel_run_to_run_mae"] > 0).all())
        self.assertTrue((repeats["same_predictions"] <= repeats["n_trajectories"]).all())

        totals = json.loads((self.root / "summary" / "ibm_summary.json").read_text())
        self.assertEqual(totals["quantum_seconds"], 12.0 * 6)
        self.assertEqual(totals["circuits"], 36 * 6)
        self.assertTrue((self.root / "summary" / "ibm_summary.png").is_file())

    def test_no_hardware_folder_is_an_error(self):
        smi.main(["--runs", self.run, "--output-dir", str(self.out / "ibm_plan"), *COMMON])
        with self.assertRaisesRegex(SystemExit, "no folder with hardware kernels"):
            summary.main(["--root", str(self.out), "--output-dir", str(self.root / "summary")])

    def test_folders_of_different_data_are_refused(self):
        self.hardware_run("ibm_a", 0.01)
        write_run(self.root / "other", [1, 2, 3], memory_fs=100.0, rng_seed=7)
        self.run = str(self.root / "other")
        self.hardware_run("ibm_b", 0.01)
        with self.assertRaisesRegex(SystemExit, "different MD data"):
            summary.main(["--root", str(self.out), "--output-dir", str(self.root / "summary")])


if __name__ == "__main__":
    unittest.main()
