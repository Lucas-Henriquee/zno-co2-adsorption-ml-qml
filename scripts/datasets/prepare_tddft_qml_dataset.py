"""Summary and transition tables from the LrTDDFT spectrum of the isolated CO2.

WARNING: these are excitations of the CO2 molecule alone, cut out of its adsorbed
geometry and placed in a box, not of the ZnO/CO2 system. The physisorbed CO2
keeps nearly the same (linear) geometry on every site, so the TDDFT descriptors
are essentially identical across sites: they carry no information about the
adsorption site and are NOT suitable as features for site classification. The
tables are kept as spectroscopic references. Each row carries ``group`` =
material/site (one TDDFT calculation); the transitions of one calculation are not
independent samples and must never be split between train and test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.common.project_config import DATA_DIR

CLASSIFICATION_WARNING = (
    "[tddft-dataset] WARNING: TDDFT descriptors of the isolated CO2 are (almost) identical "
    "across sites and are not suitable for site classification; group = material/site "
    "marks the single calculation behind every row."
)


SITE_LABELS = {
    "top_metal": 0,
    "top_oxygen": 1,
    "bridge": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build summary and transition-level datasets from the LrTDDFT spectrum of the adsorbed CO2 "
            "molecule isolated in a box (not the optical response of the ZnO-CO2 system)."
        )
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tddft-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["summary", "transitions", "both"], required=True)
    output_dir = DATA_DIR / "qml_datasets" / "tddft"
    parser.add_argument("--summary-output", type=Path, default=output_dir / "tddft_summary_dataset.csv")
    parser.add_argument("--transitions-output", type=Path, default=output_dir / "tddft_transitions_dataset.csv")
    return parser.parse_args()


def read_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"TDDFT summary file not found: {path}")
    df = pd.read_csv(path)

    required = {
        "material",
        "site",
        "tddft_status",
        "co2_isolated_total_f",
        "co2_isolated_onset_ev",
        "co2_isolated_peak_ev",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"TDDFT summary is missing columns: {missing}")

    df = df[df["tddft_status"] == "ok"].copy()

    df["label_site"] = df["site"]
    df["label_id"] = df["site"].map(SITE_LABELS)

    if df["label_id"].isna().any():
        raise ValueError("Unknown site found in summary.")

    df["label_id"] = df["label_id"].astype(int)
    df["group"] = df["material"].astype(str) + "/" + df["site"].astype(str)

    return df


def build_summary_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()

    data["feature_1_co2_isolated_total_f"] = data["co2_isolated_total_f"]
    data["feature_2_co2_isolated_onset_ev"] = data["co2_isolated_onset_ev"]
    data["feature_3_co2_isolated_peak_ev"] = data["co2_isolated_peak_ev"]

    cols = [
        "material",
        "site",
        "group",
        "label_site",
        "label_id",
        "feature_1_co2_isolated_total_f",
        "feature_2_co2_isolated_onset_ev",
        "feature_3_co2_isolated_peak_ev",
    ]

    return data[cols].sort_values(["material", "site"])


def read_transitions(tddft_dir: Path) -> pd.DataFrame:
    if not tddft_dir.is_dir():
        raise FileNotFoundError(f"TDDFT results directory not found: {tddft_dir}")
    rows = []

    for path in sorted(tddft_dir.glob("*/*/transitions.csv")):
        material = path.parent.parent.name
        site = path.parent.name

        df = pd.read_csv(path)
        df["material"] = material
        df["site"] = site

        rows.append(df)

    if not rows:
        raise FileNotFoundError("No transitions.csv files found.")

    return pd.concat(rows, ignore_index=True)


def build_transitions_dataset(summary: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "material",
        "site",
        "co2_isolated_peak_ev",
    ]

    data = transitions.merge(
        summary[summary_cols],
        on=["material", "site"],
        how="inner",
    )

    data["label_site"] = data["site"]
    data["label_id"] = data["site"].map(SITE_LABELS)

    if data["label_id"].isna().any():
        raise ValueError("Unknown site found in transitions.")

    data["label_id"] = data["label_id"].astype(int)
    # Every transition of one calculation shares this id (grouped splits only).
    data["group"] = data["material"].astype(str) + "/" + data["site"].astype(str)

    data["feature_1_energy_ev"] = data["energy_ev"]
    data["feature_2_oscillator_strength"] = data["oscillator_strength"]
    data["feature_3_delta_from_peak_ev"] = (data["energy_ev"] - data["co2_isolated_peak_ev"])

    cols = [
        "material",
        "site",
        "group",
        "label_site",
        "label_id",
        "transition_index",
        "feature_1_energy_ev",
        "feature_2_oscillator_strength",
        "feature_3_delta_from_peak_ev",
    ]

    return data[cols].sort_values(["material", "site", "transition_index"])


def save_dataset(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\nSaved: {path}")
    print(f"Samples: {len(df)}")
    if "group" in df.columns:
        print(f"Groups (independent TDDFT calculations): {df['group'].nunique()}")
    print(CLASSIFICATION_WARNING)
    print("Classes:")
    print(df["label_site"].value_counts())


def main():
    args = parse_args()

    summary = read_summary(args.summary)

    if args.mode in ["summary", "both"]:
        summary_dataset = build_summary_dataset(summary)
        save_dataset(summary_dataset, args.summary_output)

    if args.mode in ["transitions", "both"]:
        if args.tddft_dir is None:
            raise SystemExit("--tddft-dir is required for transitions or both mode.")
        transitions = read_transitions(args.tddft_dir)
        transitions_dataset = build_transitions_dataset(summary, transitions)
        save_dataset(transitions_dataset, args.transitions_output)


if __name__ == "__main__":
    main()
