from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.common.project_config import RESULTS_DIR


def load_metrics(base_dir: Path, model_type: str) -> list[dict]:
    rows = []
    for metrics_path in base_dir.rglob("metrics.json"):
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WARNING: could not read {metrics_path}; skipping it")
            continue

        data["model_type"] = model_type
        # Flatten nested blocks written by the model scripts.
        for prefix, key in [("", "test_metrics"), ("train_", "train_metrics")]:
            for name, value in (data.get(key) or {}).items():
                data.setdefault(f"{prefix}{name}", value)
        diagnosis = data.get("fit_diagnosis") or {}
        if isinstance(diagnosis, dict):
            data.setdefault("train_test_gap", diagnosis.get("train_test_gap"))
            data["fit_diagnosis"] = diagnosis.get("diagnosis")
        grouping = data.pop("grouping", None) or {}
        data["train_groups"] = grouping.get("train_groups")
        data["test_groups"] = grouping.get("test_groups")
        data["run_dir"] = str(metrics_path.parent)
        rows.append(data)

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svm-dir", type=Path, default=RESULTS_DIR / "svm")
    parser.add_argument("--qsvm-dir", type=Path, default=RESULTS_DIR / "qsvm")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "comparison_table.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    if args.svm_dir.exists():
        rows.extend(load_metrics(args.svm_dir, "SVM"))
    if args.qsvm_dir.exists():
        rows.extend(load_metrics(args.qsvm_dir, "QSVM"))

    if not rows:
        print("No metrics.json files were found. Run the sweep first.")
        return

    df = pd.DataFrame(rows)

    # Columns shared by both models, in the order used for comparison.
    common_cols = [
        "model_type",
        "feature_set",
        "n_features",
        "scaler",
        "accuracy",
        "balanced_accuracy",
        "train_balanced_accuracy",
        "train_test_gap",
        "fit_diagnosis",
        "cv_balanced_accuracy_mean",
        "split_strategy",
        "train_groups",
        "test_groups",
        "fit_time_s",
        "predict_time_s",
        "total_time_s",
        "test_size",
        "samples",
        "max_samples",
        "kernel",
        "reps",
        "entanglement",
        "run_dir",
    ]
    ordered_cols = [c for c in common_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    df = df.sort_values(
        by=["model_type", "balanced_accuracy"],
        ascending=[True, False],
        na_position="last",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Comparison table saved to: {args.output}")
    print(f"Total runs: {len(df)} ({(df['model_type'] == 'SVM').sum()} SVM, {(df['model_type'] == 'QSVM').sum()} QSVM)")
    print()
    print("Top 5 by balanced_accuracy for each model:")
    for model_type in df["model_type"].unique():
        subset = df[df["model_type"] == model_type].head(5)
        print(f"\n-- {model_type} --")
        cols_to_show = [c for c in ["feature_set", "scaler", "balanced_accuracy", "train_test_gap", "fit_diagnosis"] if c in subset.columns]
        print(subset[cols_to_show].to_string(index=False))

    print("\nLargest train-test gap (possible overfitting) for each model:")
    for model_type in df["model_type"].unique():
        subset = df[df["model_type"] == model_type].nlargest(3, "train_test_gap")
        print(f"\n-- {model_type} --")
        print(subset[["feature_set", "scaler", "balanced_accuracy", "train_test_gap"]].to_string(index=False))    


if __name__ == "__main__":
    main()
