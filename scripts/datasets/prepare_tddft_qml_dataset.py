"""Summary and transition reference tables from the LrTDDFT spectrum of the isolated CO2.

WARNING: these are excitations of the CO2 molecule alone, cut out of its adsorbed
geometry and placed in a box, not of the ZnO/CO2 system. The physisorbed CO2
keeps nearly the same (linear) geometry on every site, so the TDDFT descriptors
are essentially identical across sites: they carry no information about the
adsorption site and are NOT suitable as features for site classification. The
tables are spectroscopic references only: they hold no ``feature_*`` columns and
no site label. Each row carries ``group`` = material/site (one TDDFT calculation).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.common.project_config import DATA_DIR
from scripts.simulations.tddft.run_tddft import LEGACY_COLUMN_ALIASES

CLASSIFICATION_WARNING = (
    "[tddft-dataset] WARNING: TDDFT descriptors of the isolated CO2 are (almost) identical "
    "across sites and are not suitable for site classification; these are reference tables "
    "(no feature_ columns, no label), group = material/site marks the single calculation "
    "behind every row."
)

REQUIRED_COLUMNS = (
    "co2_isolated_first_excitation_ev",
    "co2_isolated_first_bright_excitation_ev",
    "co2_isolated_peak_ev",
)
# Absent from older tddft_summary.csv files; written as NaN then.
OPTIONAL_COLUMNS = (
    "co2_isolated_first_excitation_f",
    "co2_isolated_peak_f",
    "co2_isolated_ks_ionization_threshold_ev",
    "co2_isolated_first_bright_above_ks_threshold",
    "trk_sum",
    "trk_fraction",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build summary and transition-level reference tables from the LrTDDFT spectrum of the "
            "adsorbed CO2 molecule isolated in a box (not the optical response of the ZnO-CO2 "
            "system; not ML features)."
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
    df = df.rename(columns={old: new for old, new in LEGACY_COLUMN_ALIASES.items() if new not in df.columns})

    missing = sorted({"material", "site", "tddft_status", *REQUIRED_COLUMNS}.difference(df.columns))
    if missing:
        raise ValueError(f"TDDFT summary is missing columns: {missing}")
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = float("nan")

    df = df[df["tddft_status"] == "ok"].copy()
    df["group"] = df["material"].astype(str) + "/" + df["site"].astype(str)
    return df


def build_summary_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["material", "site", "group", *REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]
    return summary[cols].sort_values(["material", "site"])


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
    data = transitions.merge(
        summary[["material", "site", "group", "co2_isolated_peak_ev"]],
        on=["material", "site"],
        how="inner",
    )
    data["delta_from_peak_ev"] = data["energy_ev"] - data["co2_isolated_peak_ev"]

    cols = [
        "material",
        "site",
        "group",
        "transition_index",
        "energy_ev",
        "oscillator_strength",
        "delta_from_peak_ev",
    ]

    return data[cols].sort_values(["material", "site", "transition_index"])


def save_dataset(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\nSaved: {path}")
    print(f"Rows: {len(df)}")
    if "group" in df.columns:
        print(f"Groups (independent TDDFT calculations): {df['group'].nunique()}")
    print(CLASSIFICATION_WARNING)


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
