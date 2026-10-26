"""Locate site-level MD production tables and read them per seed."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.common.md_inputs import find_md_summaries
from scripts.datasets.prepare_multitech_dataset import load_md_summary_features, load_md_windows
from scripts.workflows.run_datasets import collect_inputs


def write_frames(directory: Path, seeds=(1, 2), n_frames=4) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for step in range(n_frames):
            rows.append({
                "seed": seed,
                "phase": "production",
                "step": step,
                "time_fs": float(step),
                "c_surface_distance_ang": 2.0 + 0.2 * step + 0.01 * seed,
                "oco_angle_deg": 170.0 + step,
            })
    path = directory / "md_production_frames.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class MDInputTests(unittest.TestCase):
    def test_site_level_production_table_is_found_and_split_by_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = write_frames(root / "md/ZnO/top_metal")
            # Per-seed folders are ignored; the site-level table holds every seed.
            write_frames(root / "md/ZnO/top_metal/seed_1", seeds=(1,))
            self.assertEqual(find_md_summaries(root), [current])
            config = {"input": {}, "methods": {"multitech": {"targets": ["md_windows"]}}}
            self.assertEqual(collect_inputs("multitech", config, root, None), [current])

            windows = load_md_windows(root, 2)
            self.assertEqual(
                sorted(windows["group"].unique()),
                ["ZnO/top_metal/seed_1", "ZnO/top_metal/seed_2"],
            )
            summary = load_md_summary_features(root)
            self.assertEqual(len(summary), 1)
            self.assertEqual(int(summary["md_n_trajectories"].iloc[0]), 2)

    def test_legacy_md_summary_is_not_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "md/ZnO/bridge"
            directory.mkdir(parents=True)
            (directory / "md_summary.csv").write_text("time_ps,c_surface_distance_ang,oco_angle_deg\n0,2,170\n")
            with self.assertRaises(FileNotFoundError):
                find_md_summaries(root)

    def test_missing_trajectory_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                find_md_summaries(Path(tmp))


if __name__ == "__main__":
    unittest.main()
