"""site_memory_compare: table and figure from saved site_memory folders, with its checks."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.analysis import site_memory as sm
from scripts.analysis import site_memory_compare as compare
from site_memory_fixtures import write_run

FAST = ["--qsvm-kernel", "zz_exact", "--n-permutations", "100", "--models", "svm_rbf", "qsvm", "qsvm_bw"]


class SiteMemoryCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        write_run(root / "cold", [1, 2, 3], temperature=100.0, memory_fs=160.0)
        write_run(root / "hot", [1, 2, 3, 4], temperature=300.0, memory_fs=60.0, rng_seed=3)
        cls.out = root / "out"
        with mock.patch("builtins.print"):
            sm.main(["--runs", str(root / "cold"), "--output-dir", str(cls.out / "100k"), *FAST])
            sm.main(["--runs", str(root / "hot"), "--output-dir", str(cls.out / "300k"), *FAST])
            sm.main(["--runs", str(root / "hot"), "--output-dir", str(cls.out / "300k_full"), *FAST,
                     "--qsvm-entanglement", "full"])
            sm.main(["--runs", str(root / "hot"), "--output-dir", str(cls.out / "300k_w20"), *FAST,
                     "--window-fs", "20"])
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_compare(self, *names, output="compare"):
        with mock.patch("builtins.print"):
            compare.main(["--inputs", *[str(self.out / name) for name in names],
                          "--output-dir", str(self.root / output)])
        return self.root / output

    def test_table_figure_and_seed_warning(self):
        folder = self.run_compare("300k", "100k", "300k_full")
        table = pd.read_csv(folder / "site_memory_compare.csv")
        self.assertEqual(len(table), 3 * 3)
        reference = table[table["reference_setting"]]
        self.assertEqual(sorted(set(reference["temperature_k"])), [100.0, 300.0])
        self.assertEqual(set(table.loc[~table["reference_setting"], "qsvm_entanglement"]), {"full"})
        windows = pd.read_csv(self.out / "100k" / "site_memory_windows.csv")
        expected = sm.memory_time(windows, "svm_rbf", 0.05)
        row = table[(table["folder"] == "100k") & (table["model"] == "svm_rbf")].iloc[0]
        self.assertEqual(row["memory_time_fs"], expected)
        self.assertEqual(int(row["seeds"]), 3)
        self.assertTrue((folder / "site_memory_compare.png").is_file())
        notes = json.loads((folder / "site_memory_compare.json").read_text())
        self.assertTrue(any("numbers of seeds" in warning for warning in notes["warnings"]))

    def test_different_window_widths_are_refused(self):
        with self.assertRaisesRegex(SystemExit, "window widths"):
            self.run_compare("300k", "300k_w20", output="bad_width")

    def test_two_reference_folders_of_one_temperature_are_refused(self):
        twin = self.out / "300k_copy"
        twin.mkdir(exist_ok=True)
        for name in ("site_memory_summary.json", "site_memory_windows.csv"):
            (twin / name).write_bytes((self.out / "300k" / name).read_bytes())
        with self.assertRaisesRegex(SystemExit, "more than one reference"):
            self.run_compare("300k", "300k_copy", output="bad_twin")

    def test_not_a_site_memory_folder(self):
        with self.assertRaisesRegex(SystemExit, "not a site_memory output"):
            compare.main(["--inputs", str(self.root), "--output-dir", str(self.root / "x")])


if __name__ == "__main__":
    unittest.main()
