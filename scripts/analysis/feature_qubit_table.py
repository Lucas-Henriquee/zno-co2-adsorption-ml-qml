import argparse
from pathlib import Path

import pandas as pd

from scripts.common.project_config import RESULTS_DIR
from scripts.models.benchmarks.run_benchmarks import FEATURE_SETS as FEATURE_SETS_QSVM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Number of features and qubits per QSVM feature set.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "feature_qubit_table.csv")
    return parser.parse_args()


def build_table(feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, features in feature_sets.items():
        n_features = len(features)
        rows.append({
            "feature_set": name,
            "n_features": n_features,
            "n_qubits": n_features,  # ZZFeatureMap maps 1 feature to 1 qubit
        })

    return pd.DataFrame(rows).sort_values("n_features").reset_index(drop=True)

def main():
    args = parse_args()
    df = build_table(FEATURE_SETS_QSVM)

    output_csv = args.output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("Table: Number of features and qubits per feature set (QSVM)\n")
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_csv}")

if __name__ == "__main__":
    main()
