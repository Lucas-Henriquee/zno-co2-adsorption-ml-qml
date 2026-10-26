"""TDDFT post-processing without GPAW: units, oscillator strengths, onset, TRK, exit code."""
import csv
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd
from ase.units import Hartree

from scripts.datasets import prepare_tddft_qml_dataset as dataset
from scripts.simulations.tddft import run_tddft


class FakeExcitation:
    def __init__(self, f, fxyz):
        self._osc = [f, *fxyz]

    def get_oscillator_strength(self):
        return np.asarray(self._osc)


class FakeLrTDDFT:
    """Mimics gpaw.lrtddft.LrTDDFT after diagonalize(): energies in Hartree."""

    def __init__(self, energies_hartree, strengths):
        self._energies = np.asarray(energies_hartree, dtype=float)
        self._excitations = [FakeExcitation(f, (f / 3.0, f / 3.0, f / 3.0)) for f in strengths]

    def get_energies(self):
        return self._energies

    def __iter__(self):
        return iter(self._excitations)


# Transition 1 is dark (f below threshold): first excitation != onset.
ENERGIES_HA = [0.30, 0.35, 0.40, 0.60]
STRENGTHS = [1.0e-5, 0.02, 0.50, 0.10]


def make_args(**overrides):
    with patch.object(sys, "argv", ["run_tddft", "--materials", "ZnO", "--sites", "top_metal", "bridge"]):
        args = run_tddft.parse_args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TransitionExtractionTests(unittest.TestCase):
    def test_energies_are_converted_from_hartree_to_ev(self):
        energies, f, fxyz = run_tddft.extract_transitions(FakeLrTDDFT(ENERGIES_HA, STRENGTHS))
        np.testing.assert_allclose(energies, np.asarray(ENERGIES_HA) * Hartree)
        self.assertAlmostEqual(energies[0], 8.163, places=2)

    def test_oscillator_strength_uses_total_component_zero(self):
        _, f, fxyz = run_tddft.extract_transitions(FakeLrTDDFT(ENERGIES_HA, STRENGTHS))
        np.testing.assert_allclose(f, STRENGTHS)
        np.testing.assert_allclose(fxyz[:, 2], np.asarray(STRENGTHS) / 3.0)

    def test_missing_oscillator_strengths_fail_loudly(self):
        lr = FakeLrTDDFT(ENERGIES_HA, STRENGTHS[:2])
        with self.assertRaises(RuntimeError):
            run_tddft.extract_transitions(lr)
        with self.assertRaises(RuntimeError):
            run_tddft.extract_transitions(FakeLrTDDFT([], []))


class SummaryTests(unittest.TestCase):
    def summarize(self, **kwargs):
        energies, f, _ = run_tddft.extract_transitions(FakeLrTDDFT(ENERGIES_HA, STRENGTHS))
        options = {"osc_threshold": 1.0e-3, "max_transitions": 0, "valence_electrons": 16.0}
        options.update(kwargs)
        return run_tddft.summarize_transitions(energies, f, **options)

    def test_onset_applies_threshold_to_f(self):
        summary = self.summarize()
        self.assertAlmostEqual(summary["co2_isolated_first_excitation_ev"], 0.30 * Hartree)
        self.assertAlmostEqual(summary["co2_isolated_onset_ev"], 0.35 * Hartree)
        self.assertAlmostEqual(summary["co2_isolated_peak_ev"], 0.40 * Hartree)
        self.assertAlmostEqual(summary["co2_isolated_peak_f"], 0.50)
        self.assertTrue(math.isnan(self.summarize(osc_threshold=1.0)["co2_isolated_onset_ev"]))

    def test_trk_sum_uses_all_transitions(self):
        summary = self.summarize(max_transitions=2)
        self.assertAlmostEqual(summary["trk_sum"], sum(STRENGTHS))
        self.assertAlmostEqual(summary["trk_fraction"], sum(STRENGTHS) / 16.0)
        self.assertAlmostEqual(summary["co2_isolated_total_f"], STRENGTHS[0] + STRENGTHS[1])
        self.assertEqual(summary["co2_isolated_num_transitions"], 2.0)
        self.assertEqual(summary["co2_isolated_num_transitions_computed"], 4.0)


class WorkflowTests(unittest.TestCase):
    def run_workflow(self, root: Path, dft_rows, fail_site=None):
        dft_dir = root / "dft"
        tddft_dir = root / "tddft"
        with (root / "dft_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["material", "site", "dft_status"])
            writer.writeheader()
            writer.writerows(dft_rows)
        for row in dft_rows:
            case = dft_dir / row["material"] / row["site"]
            case.mkdir(parents=True)
            (case / f"{row['material']}_{row['site']}_optimized.xyz").write_text("stub\n")
        # Stale outputs from an earlier run must not survive.
        (root / "tddft_summary.csv").write_text("material,site,tddft_status\nTiO2,bridge,ok\n")
        stale = tddft_dir / "ZnO" / "bridge"
        stale.mkdir(parents=True)
        (stale / "transitions.csv").write_text("stale\n")

        def fake_run_lrtddft(xyz_file, out_dir, params):
            if fail_site and fail_site in str(out_dir):
                raise RuntimeError("SCF did not converge")
            energies, f, fxyz = run_tddft.extract_transitions(FakeLrTDDFT(ENERGIES_HA, STRENGTHS))
            return energies, f, fxyz, 16.0

        with patch.multiple(
            run_tddft,
            RESULTS_DIR=root,
            DFT_RESULTS_DIR=dft_dir,
            TDDFT_RESULTS_DIR=tddft_dir,
            ensure_directories=lambda: None,
            run_lrtddft=fake_run_lrtddft,
        ):
            failures = run_tddft.run_tddft_workflow(make_args())
        return failures, pd.read_csv(root / "tddft_summary.csv")

    def test_summary_columns_and_transitions_in_ev(self):
        rows = [{"material": "ZnO", "site": site, "dft_status": "ok"} for site in ("top_metal", "bridge")]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            failures, summary = self.run_workflow(root, rows)
            self.assertEqual(failures, 0)
            self.assertEqual(list(summary["site"]), ["top_metal", "bridge"])
            self.assertNotIn("TiO2", set(summary["material"]))
            for column in run_tddft.RESULT_COLUMNS + ("tddft_status", "tddft_mode", "gs_nbands", "lr_energy_window_ev"):
                self.assertIn(column, summary.columns)
            self.assertFalse(any(column.startswith("tddft_onset") for column in summary.columns))
            self.assertEqual(int(summary["gs_nbands"].iloc[0]), 68)
            self.assertAlmostEqual(summary["co2_isolated_onset_ev"].iloc[0], 0.35 * Hartree)
            transitions = pd.read_csv(root / "tddft" / "ZnO" / "bridge" / "transitions.csv")
            np.testing.assert_allclose(transitions["energy_ev"], np.asarray(ENERGIES_HA) * Hartree)
            np.testing.assert_allclose(transitions["oscillator_strength"], STRENGTHS)

            # The dataset builder reads the new columns.
            frame = dataset.read_summary(root / "tddft_summary.csv")
            built = dataset.build_summary_dataset(frame)
            self.assertAlmostEqual(built["feature_2_co2_isolated_onset_ev"].iloc[0], 0.35 * Hartree)
            merged = dataset.build_transitions_dataset(frame, dataset.read_transitions(root / "tddft"))
            self.assertEqual(len(merged), 8)

    def test_failed_case_is_reported_and_counted(self):
        rows = [{"material": "ZnO", "site": site, "dft_status": "ok"} for site in ("top_metal", "bridge")]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            failures, summary = self.run_workflow(root, rows, fail_site="bridge")
            self.assertEqual(failures, 1)
            self.assertEqual(list(summary["tddft_status"]), ["ok", "failed"])
            self.assertFalse((root / "tddft" / "ZnO" / "bridge" / "transitions.csv").exists())

    def test_cases_with_dft_not_ok_are_skipped(self):
        rows = [
            {"material": "ZnO", "site": "top_metal", "dft_status": "ok"},
            {"material": "ZnO", "site": "bridge", "dft_status": "failed"},
        ]
        with TemporaryDirectory() as directory:
            failures, summary = self.run_workflow(Path(directory), rows)
            self.assertEqual(failures, 1)
            self.assertEqual(summary["tddft_status"].iloc[1], "skipped_dft_not_ok")
            self.assertIn("not 'ok'", summary["error_message"].iloc[1])

    def test_main_exits_non_zero_on_failure(self):
        with patch.object(run_tddft, "_require_gpaw"), patch.object(run_tddft, "parse_args"), patch.object(
            run_tddft, "run_tddft_workflow", return_value=2
        ):
            with self.assertRaises(SystemExit) as raised:
                run_tddft.main()
        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
