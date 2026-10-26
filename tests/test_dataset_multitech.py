"""Synthetic checks of the dataset, screening and QML export corrections (D1, M3, A1, A2, P2)."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.analysis import analyze_results
from scripts.datasets import export_qml_dataset
from scripts.datasets import prepare_multitech_dataset as pm

SITES = ("top_metal", "top_oxygen", "bridge")
SEEDS = (1, 2, 3)
PHI = 0.9  # AR(1) coefficient: tau_int = (1 + phi) / (1 - phi) = 19 frames


def ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def write_dft_summary(root: Path) -> None:
    rows = []
    for index, site in enumerate(SITES + ("carbonate",)):
        rows.append({
            "material": "ZnO",
            "site": site,
            "dft_status": "ok",
            "adsorption_energy_ev": -0.2 - 0.1 * index,
            "clean_band_gap_ev": 3.0,  # mesh gap, must not be used
            "adsorbed_band_gap_ev": 3.5,
            "clean_band_gap_gamma_ev": 1.0,
            "adsorbed_band_gap_gamma_ev": 1.1 + 0.1 * index,
            "co_bond_1_ang": 1.18,
            "co_bond_2_ang": 1.18,
            "oco_angle_deg": 178.0 - index,
            "c_surface_distance_ang": 2.5 + index,
            "c_min_slab_distance_ang": 2.6 + index,
            "o_min_metal_distance_ang": 2.2 + index,
        })
    rows.append({**rows[0], "site": "top_oxygen", "dft_status": "failed",
                 "adsorption_energy_ev": -9.0})
    pd.DataFrame(rows).to_csv(root / "dft_summary.csv", index=False)


def write_tddft_summary(root: Path) -> None:
    rows = [{
        "material": "ZnO",
        "site": site,
        "tddft_status": "ok",
        "co2_isolated_first_excitation_ev": 8.0 + index,
        "co2_isolated_onset_ev": 8.1 + index,
        "co2_isolated_peak_ev": 9.0 + index,
        "co2_isolated_total_f": 0.5,
        "trk_sum": 15.0,
    } for index, site in enumerate(SITES)]
    pd.DataFrame(rows).to_csv(root / "tddft_summary.csv", index=False)


def write_co2rr_summary(root: Path) -> None:
    rows = []
    for index, site in enumerate(SITES):
        prefix = "dE_" if site == "bridge" else "dG_"
        row = {"material": "ZnO", "site": site, "pathway_status": "ok",
               "energy_basis": "electronic_energies_only" if site == "bridge" else "free_energy_CHE",
               "limiting_step": "CO2_to_COOH", "limiting_potential_v": -0.5 - index}
        for step in ("CO2_adsorption", "CO2_to_COOH", "COOH_to_CO", "CO_desorption", "overall"):
            row[f"{prefix}{step}_ev"] = 0.1 * (index + 1)
        rows.append(row)
    rows.append({**rows[0], "site": "top_metal", "pathway_status": "inconsistent"})
    pd.DataFrame(rows).to_csv(root / "co2rr_pathway_summary.csv", index=False)


def write_md_frames(root: Path, n_production: int = 400, n_equilibration: int = 50) -> None:
    rng = np.random.default_rng(0)
    for site_index, site in enumerate(SITES):
        rows = []
        for seed in SEEDS:
            for step in range(n_equilibration):
                rows.append({"seed": seed, "phase": "equilibration", "step": step,
                             "time_fs": float(step), "c_surface_distance_ang": 99.0,
                             "c_min_slab_distance_ang": 99.0, "o_min_metal_distance_ang": 99.0,
                             "co_bond_avg_ang": 9.0, "oco_angle_deg": 0.0})
            noise = ar1(n_production, PHI, rng)
            for step in range(n_production):
                rows.append({
                    "seed": seed,
                    "phase": "production",
                    "step": step,
                    "time_fs": 2.0 * step,
                    "c_surface_distance_ang": 2.5 + site_index + 0.05 * noise[step],
                    "c_min_slab_distance_ang": 2.6 + site_index + 0.05 * noise[step],
                    "o_min_metal_distance_ang": 2.2 + site_index,
                    "co_bond_avg_ang": 1.18,
                    "oco_angle_deg": 175.0 - site_index + noise[step],
                })
        directory = root / "md" / "ZnO" / site
        directory.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(directory / "md_production_frames.csv", index=False)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_dft_summary(self.root)
        write_tddft_summary(self.root)
        write_co2rr_summary(self.root)
        write_md_frames(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_summary_has_one_row_per_successful_calculation(self):
        summary = pm.build_summary_dataset(self.root)
        self.assertEqual(sorted(summary["site"]), sorted(SITES))  # no cartesian product
        self.assertNotIn("carbonate", set(summary["site"]))
        self.assertTrue((summary["feature_dft_adsorption_energy_ev"] > -1.0).all())  # failed row dropped
        top_oxygen = summary.set_index("site").loc["top_oxygen"]
        # Gamma-grid PBE gaps (1.2 - 1.0), not the mesh gaps (3.5 - 3.0).
        self.assertAlmostEqual(top_oxygen["feature_dft_pbe_gap_change_ev"], 0.2)
        self.assertAlmostEqual(top_oxygen["feature_tddft_co2_isolated_onset_ev"], 9.1)
        self.assertEqual(int(top_oxygen["md_n_trajectories"]), len(SEEDS))
        self.assertTrue(summary["group"].is_unique)
        bridge = summary.set_index("site").loc["bridge"]
        self.assertEqual(bridge["co2rr_energy_basis"], "electronic_energies_only")
        self.assertAlmostEqual(bridge["feature_co2rr_d_CO2_to_COOH_ev"], 0.3)
        self.assertAlmostEqual(top_oxygen["feature_co2rr_d_overall_ev"], 0.2)
        self.assertFalse(any("deltaG" in column for column in summary.columns))

    def test_md_windows_use_production_frames_and_carry_groups(self):
        windows = pm.build_unique_md_window_dataset(self.root, 0)
        self.assertEqual(windows["group"].nunique(), len(SITES) * len(SEEDS))
        self.assertIn("ZnO/bridge/seed_2", set(windows["group"]))
        # Equilibration frames (value 99) never enter a window.
        self.assertTrue((windows["feature_md_window_c_surface_distance_mean_a"] < 10).all())
        self.assertTrue((windows["feature_md_window_mean_oco_angle_deg"] > 100).all())
        # Each group keeps one label (its starting site).
        self.assertTrue((windows.groupby("group")["label_site_id"].nunique() == 1).all())
        for name in pm.MD_WINDOW_FEATURES.values():
            self.assertIn(name, windows.columns)

    def test_block_length_exceeds_correlation_time(self):
        windows = pm.build_unique_md_window_dataset(self.root, 0, block_tau_factor=2.0)
        frame_dt = 2.0
        self.assertTrue((windows["block_length_fs"] >= 2.0 * windows["tau_int_fs"] - frame_dt).all())
        self.assertTrue((windows["tau_int_fs"] > 5 * frame_dt).all())  # AR(1), phi = 0.9
        self.assertTrue((windows["lag1_autocorrelation"] > 0.7).all())
        # Blocks of a trajectory do not overlap.
        for _, group in windows.groupby("group"):
            starts = group["window_start_fs"].to_numpy()
            ends = group["window_end_fs"].to_numpy()
            self.assertTrue((starts[1:] > ends[:-1]).all())

    def test_window_cap_makes_blocks_longer_not_shorter(self):
        uncapped = pm.build_unique_md_window_dataset(self.root, 0)
        capped = pm.build_unique_md_window_dataset(self.root, 2)
        self.assertTrue((capped.groupby("group").size() <= 2).all())
        self.assertGreaterEqual(capped["block_length_fs"].min(), uncapped["block_length_fs"].min())

    def test_trajectory_dataset_has_one_row_per_seed(self):
        trajectories = pm.build_md_trajectory_dataset(self.root)
        self.assertEqual(len(trajectories), len(SITES) * len(SEEDS))
        self.assertTrue(trajectories["group"].is_unique)

    def test_expanded_target_is_rejected(self):
        argv = ["prog", "--results-dir", str(self.root), "--output-dir", str(self.root / "out"),
                "--targets", "expanded"]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(SystemExit, "cartesian product"):
                pm.main()

    def test_main_writes_grouped_datasets(self):
        out = self.root / "out"
        argv = ["prog", "--results-dir", str(self.root), "--output-dir", str(out)]
        with patch.object(sys, "argv", argv), patch("builtins.print"):
            pm.main()
        windows = pd.read_csv(out / "md_window_unique_dataset.csv")
        self.assertIn("group", windows.columns)
        summary = pd.read_csv(out / "multitech_selected_features_dataset.csv")
        self.assertEqual(len(summary), len(SITES))


class AutocorrelationTests(unittest.TestCase):
    def test_white_noise_is_uncorrelated(self):
        tau, lag1 = pm.integrated_autocorrelation_time(np.random.default_rng(1).normal(size=5000))
        self.assertLess(tau, 1.5)
        self.assertLess(abs(lag1), 0.1)

    def test_ar1_correlation_time(self):
        tau, lag1 = pm.integrated_autocorrelation_time(ar1(20000, PHI, np.random.default_rng(2)))
        self.assertAlmostEqual(lag1, PHI, delta=0.03)
        self.assertGreater(tau, 12)
        self.assertLess(tau, 26)


class ScreeningTests(unittest.TestCase):
    def test_failed_and_carbonate_rows_are_excluded_and_gamma_gap_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dft_summary(root)
            write_tddft_summary(root)
            dft = analyze_results.read_csv_rows(root / "dft_summary.csv")
            tddft = analyze_results.read_csv_rows(root / "tddft_summary.csv")
            ranked = analyze_results.merge_results(dft, tddft)
        self.assertEqual(sorted(row["site"] for row in ranked), sorted(SITES))
        self.assertNotIn("total_score", ranked[0])
        energies = [row["adsorption_energy_ev"] for row in ranked]
        self.assertEqual(energies, sorted(energies))
        self.assertTrue(all(row["clean_band_gap_gamma_ev"] == 1.0 for row in ranked))
        self.assertIn("co2_isolated_onset_ev", ranked[0])


class QMLExportTests(unittest.TestCase):
    def rows(self):
        return [
            {"material": "ZnO", "site": "top_metal", "adsorption_energy_ev": "-0.5",
             "adsorbed_band_gap_gamma_ev": "1.2"},
            {"material": "ZnO", "site": "top_oxygen", "adsorption_energy_ev": "",
             "adsorbed_band_gap_gamma_ev": "1.1"},
            {"material": "ZnO", "site": "bridge", "adsorption_energy_ev": "-0.1",
             "adsorbed_band_gap_gamma_ev": "nan"},
            {"material": "ZnO", "site": "carbonate", "adsorption_energy_ev": "-1.5",
             "adsorbed_band_gap_gamma_ev": "1.0"},
        ]

    def test_missing_values_are_dropped_not_imputed(self):
        with patch("builtins.print"):
            rows = export_qml_dataset.build_qml_rows(self.rows())
        self.assertEqual([row["site"] for row in rows], ["top_metal"])
        self.assertEqual(rows[0]["x_adsorption_energy_raw"], -0.5)
        self.assertFalse(any(key.startswith("target_score") for key in rows[0]))
        self.assertEqual(rows[0]["label_site_id"], 0)

    def test_parse_float_never_returns_zero_for_missing(self):
        self.assertTrue(np.isnan(export_qml_dataset.parse_float("")))
        self.assertTrue(np.isnan(export_qml_dataset.parse_float("inf")))

    def test_export_round_trip(self):
        rows = [dict(row, adsorption_energy_ev=row["adsorption_energy_ev"] or "-0.3",
                     adsorbed_band_gap_gamma_ev="1.3" if row["adsorbed_band_gap_gamma_ev"] == "nan"
                     else row["adsorbed_band_gap_gamma_ev"]) for row in self.rows()]
        with tempfile.TemporaryDirectory() as tmp, patch("builtins.print"):
            output = Path(tmp) / "qml.csv"
            export_qml_dataset.write_csv(output, export_qml_dataset.build_qml_rows(rows))
            with output.open() as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual(len(exported), 3)
        scaled = [float(row["x_adsorption_energy_scaled"]) for row in exported]
        self.assertEqual((min(scaled), max(scaled)), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
