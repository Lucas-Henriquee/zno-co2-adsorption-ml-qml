"""site_memory_ibm: plan without network, hardware path with a fake backend, reuse of saved kernels."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts.analysis import site_memory_ibm as smi
from scripts.models.qml.zz_kernel import zz_kernel
from scripts.models.qml_ibm import qsvm_ibm_hardware as hw
from tests.test_site_memory import write_run

BASE = ["--window-starts-fs", "0", "200", "--ideal-kernel", "zz_exact", "--n-permutations", "300"]


def fake_train(X, *args, **kwargs):
    """Ideal kernel plus a small symmetric perturbation, as a noisy QPU would return."""
    raw = zz_kernel(X)
    noise = 0.01 * np.cos(np.add.outer(np.arange(len(X)), np.arange(len(X))))
    raw = raw + noise - np.diag(np.diag(noise))
    np.fill_diagonal(raw, 1.0)
    return raw, raw, {"n_circuits": len(X) * (len(X) - 1) // 2}, {}, [{"job_id": "fake", "quantum_seconds": 1.5}]


def fake_cross(A, B, *args, **kwargs):
    return zz_kernel(A, B) - 0.01, {"n_circuits": len(A) * len(B)}, {}, [{"job_id": "fake", "quantum_seconds": 2.0}]


class FakeBackend:
    name = "fake_backend"


class SiteMemoryIBMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_run(self.root / "run", [1, 2, 3], memory_fs=100.0)
        self.run = str(self.root / "run")

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_only_never_contacts_ibm(self):
        out = self.root / "plan"
        with mock.patch.object(hw, "create_runtime_service", side_effect=AssertionError("network")):
            smi.main(["--runs", self.run, "--output-dir", str(out), *BASE])
        plan = json.loads((out / "site_memory_ibm_plan.json").read_text())
        self.assertEqual(plan["total_circuits"], 2 * 36)  # fixed scaling: 9 rows -> 36 pairs per window
        windows = pd.read_csv(out / "site_memory_ibm_windows.csv")
        self.assertEqual(set(windows["model"]), {"svm_rbf", "qsvm_ideal"})
        early = windows[(windows["t_start_fs"] == 0) & (windows["model"] == "svm_rbf")].iloc[0]
        self.assertEqual(int(early["n_correct"]), 9)

    def test_train_scaling_needs_one_kernel_per_split(self):
        out = self.root / "plan_train"
        smi.main(["--runs", self.run, "--output-dir", str(out), "--scaling", "train", *BASE])
        plan = json.loads((out / "site_memory_ibm_plan.json").read_text())
        self.assertEqual(plan["windows"][0]["circuits"], 3 * (15 + 3 * 6))

    def test_unknown_window_start_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "no window starts at 30"):
            smi.main(["--runs", self.run, "--output-dir", str(self.root / "x"), "--window-starts-fs", "30",
                      "--ideal-kernel", "zz_exact"])

    def test_hardware_run_and_reuse_give_the_same_result(self):
        out = self.root / "hw"
        patches = [
            mock.patch.object(hw, "create_runtime_service", return_value=object()),
            mock.patch.object(hw, "pick_backend", return_value=FakeBackend()),
            mock.patch.object(hw, "build_hardware_executor", return_value=(object(), object())),
            mock.patch.object(hw, "evaluate_hardware_kernel_train", side_effect=fake_train),
            mock.patch.object(hw, "evaluate_hardware_kernel_cross", side_effect=fake_cross),
            mock.patch("scripts.models.qml.qsvm_baseline.build_feature_map", return_value=(None, "fake")),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        smi.main(["--runs", self.run, "--output-dir", str(out), "--submit", *BASE])
        windows = pd.read_csv(out / "site_memory_ibm_windows.csv")
        self.assertEqual(set(windows["model"]), {"svm_rbf", "qsvm_ideal", "qsvm_ibm"})
        deviation = pd.read_csv(out / "site_memory_ibm_kernel_deviation.csv")
        self.assertTrue((deviation["kernel_mae"] > 0).all() and (deviation["kernel_mae"] < 0.02).all())
        summary = json.loads((out / "site_memory_ibm_summary.json").read_text())
        self.assertEqual(summary["hardware"]["backend"], "fake_backend")
        self.assertAlmostEqual(summary["hardware"]["quantum_seconds"], 2 * 1.5)

        reused = self.root / "reuse"
        for patch in patches:
            patch.stop()
        with mock.patch.object(hw, "create_runtime_service", side_effect=AssertionError("network")):
            smi.main(["--runs", self.run, "--output-dir", str(reused), "--reuse-hardware",
                      str(out / smi.HARDWARE_DIR), *BASE])
        again = pd.read_csv(reused / "site_memory_ibm_windows.csv")
        pd.testing.assert_frame_equal(windows, again)

    def test_reuse_refuses_kernels_of_other_data(self):
        folder = self.root / "saved"
        folder.mkdir()
        (folder / "hardware_record.json").write_text(json.dumps({"kernels": {
            "w0_all": {"left_sha256": "other", "right_sha256": None}}}))
        np.save(folder / "w0_all.npy", np.eye(9))
        with self.assertRaisesRegex(SystemExit, "other data"):
            smi.main(["--runs", self.run, "--output-dir", str(self.root / "x"), "--reuse-hardware", str(folder),
                      "--window-starts-fs", "0", "--ideal-kernel", "zz_exact"])

    def test_hardware_training_kernel_has_unit_diagonal_and_is_psd(self):
        job_args = smi.parse_args(["--runs", self.run, "--output-dir", "x", "--window-starts-fs", "0"])
        rows = pd.DataFrame({"window": 0, "t_start_fs": 0.0, "t_end_fs": 40.0, "seed": np.repeat([1, 2, 3], 3),
                             "label": np.tile([0, 1, 2], 3), "site": "x",
                             "c_height_top_plane_ang": np.linspace(2.5, 4.5, 9),
                             "c_min_slab_distance_ang": np.linspace(3.0, 5.0, 9),
                             "c_o_min_slab_distance_ang": np.linspace(2.8, 4.0, 9)})
        job = smi.WindowJob(rows, smi.sm.FEATURE_SETS["geometry3"], job_args)
        noisy = zz_kernel(job.Z) - 0.3
        kernels = smi.hardware_split_kernels(job, {"w0_all": noisy})
        for k_train, _ in kernels:
            self.assertGreater(np.linalg.eigvalsh(k_train).min(), -1e-10)


if __name__ == "__main__":
    unittest.main()
