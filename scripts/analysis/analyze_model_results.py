from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COMPARISON_COLUMNS = [
    "rank",
    "model",
    "requested_model",
    "feature_set",
    "n_features",
    "n_qubits",
    "samples",
    "train_samples",
    "test_samples",
    "split_strategy",
    "train_groups",
    "test_groups",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_f1_macro",
    "train_balanced_accuracy",
    "train_test_gap",
    "diagnosis",
    "scaler",
    "kernel",
    "simulator",
    "split_sha256",
    "random_state",
    "test_size",
    "features_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a concise comparison table and report for one model benchmark."
    )
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def display_model(row: pd.Series) -> str:
    requested = str(row.get("requested_model", "")).lower()
    if requested == "svm":
        return "Classical SVM"
    if requested == "qsvm":
        return "Local QSVM"
    if requested == "qsvm_ibm":
        return "IBM QSVM"
    return str(row.get("model", requested or "Unknown"))


def split_status(table: pd.DataFrame) -> str:
    """Require a nonempty fingerprint for every row before claiming a shared split."""
    if table.empty or "split_sha256" not in table:
        return "unknown"
    hashes = table["split_sha256"].fillna("").astype(str).str.strip()
    if hashes.eq("").any():
        return "unknown"
    return "yes" if hashes.nunique() == 1 else "no"


def grouped_status(table: pd.DataFrame) -> str:
    """'yes' only if every run recorded a grouped (trajectory-wise) split."""
    if table.empty or "split_strategy" not in table:
        return "unknown"
    strategies = table["split_strategy"].fillna("").astype(str).str.strip()
    if strategies.eq("").any():
        return "unknown"
    return "yes" if strategies.eq("grouped").all() else "no"


def validate_metrics(table: pd.DataFrame) -> None:
    for column in ("test_accuracy", "test_balanced_accuracy", "test_f1_macro", "train_balanced_accuracy"):
        values = pd.to_numeric(table[column], errors="coerce")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise SystemExit(f"Completed model runs require finite {column} values between 0 and 1.")
    gaps = pd.to_numeric(table["train_test_gap"], errors="coerce")
    if not np.isfinite(gaps).all() or not gaps.between(-1, 1).all():
        raise SystemExit("Completed model runs require finite train_test_gap values between -1 and 1.")
    for column in ("samples", "train_samples", "test_samples"):
        if column not in table:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if not np.isfinite(values).all() or not ((values > 0) & (values % 1 == 0)).all():
            raise SystemExit(f"Completed model runs require positive integer {column} values.")
    if {"train_samples", "test_samples"}.issubset(table.columns):
        if not (table["train_samples"] + table["test_samples"] == table["samples"]).all():
            raise SystemExit("Train and test sample counts must sum to the dataset sample count.")


def write_report(table: pd.DataFrame, output: Path, excluded_runs: int = 0) -> None:
    shared_split = split_status(table)
    sample_counts = ", ".join(str(int(value)) for value in sorted(table["samples"].unique()))
    test_groups = ""
    if "test_groups" in table:
        values = pd.to_numeric(table["test_groups"], errors="coerce").dropna().unique()
        test_groups = ", ".join(str(int(value)) for value in sorted(values))
    lines = [
        "# Model Comparison Report",
        "",
        f"- Completed model runs: {len(table)}",
        f"- Excluded unsuccessful runs: {excluded_runs}",
        f"- Shared train/test split: {shared_split}",
        f"- Grouped split (no MD trajectory in both train and test): {grouped_status(table)}",
        f"- Dataset sample counts across runs: {sample_counts}",
        f"- Independent test trajectories (groups) across runs: {test_groups or 'unavailable'}",
        "",
        "## Results",
        "",
        "| Rank | Model | Balanced accuracy | Macro F1 | Train-test gap | Diagnosis |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| {rank} | {model} | {ba:.4f} | {f1:.4f} | {gap:.4f} | {diagnosis} |".format(
                rank=int(row["rank"]),
                model=row["model"],
                ba=float(row["test_balanced_accuracy"]),
                f1=float(row["test_f1_macro"]),
                gap=float(row["train_test_gap"]),
                diagnosis=row["diagnosis"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The ordering summarizes the recorded test scores. It is not a statistical significance test or evidence of quantum advantage. Dataset provenance and independent sampling units must be checked before interpreting generalization.",
            "",
            "Use multiple independent trajectories, repeated seeds, uncertainty intervals, and grouped validation before drawing comparative scientific conclusions.",
        ]
    )
    if shared_split != "yes":
        lines.extend(["", "A shared train/test split is not verified for every run. The score ordering must not be interpreted as a controlled comparison."])
    if "random_state" in table:
        seeds = table["random_state"].dropna().unique()
        lines.extend(["", f"Recorded random seeds: {', '.join(str(int(seed)) for seed in sorted(seeds)) or 'unavailable'}."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_path = args.benchmark_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Benchmark summary not found: {summary_path}")

    data = pd.read_csv(summary_path)
    required = {
        "requested_model",
        "status",
        "samples",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1_macro",
        "train_balanced_accuracy",
        "train_test_gap",
        "diagnosis",
        "split_sha256",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise SystemExit(f"Benchmark summary is missing columns: {missing}")

    completed = data[data["status"] == "ok"].copy()
    if completed.empty:
        raise SystemExit("Benchmark summary contains no completed model runs.")
    validate_metrics(completed)
    completed["model"] = completed.apply(display_model, axis=1)
    completed = completed.sort_values(
        ["test_balanced_accuracy", "test_f1_macro"],
        ascending=False,
    ).reset_index(drop=True)
    completed.insert(0, "rank", range(1, len(completed) + 1))
    available = [column for column in COMPARISON_COLUMNS if column in completed.columns]
    comparison = completed[available]

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "model_comparison.csv"
    report_path = output_dir / "model_report.md"
    comparison.to_csv(comparison_path, index=False)
    write_report(comparison, report_path, excluded_runs=len(data) - len(completed))
    print(f"[model-analysis] wrote {comparison_path}")
    print(f"[model-analysis] wrote {report_path}")


if __name__ == "__main__":
    main()
