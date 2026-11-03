from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


RUNS = [
    {
        "run": "SVM all12 RBF",
        "dir": "results/svm_multitech_expanded_rbf_standard",
        "model": "SVM",
        "dataset": "Expanded",
        "feature_set": "all12",
        "method": "RBF",
        "scaler": "standard",
        "purpose": "Full feature diagnostic",
    },
    {
        "run": "SVM static6 linear",
        "dir": "results/svm_multitech_static_only_linear_standard",
        "model": "SVM",
        "dataset": "Expanded",
        "feature_set": "static6",
        "method": "linear",
        "scaler": "standard",
        "purpose": "Static fingerprint test",
    },
    {
        "run": "SVM dynamic6 linear",
        "dir": "results/svm_multitech_dynamic_only_linear_standard",
        "model": "SVM",
        "dataset": "Expanded",
        "feature_set": "dynamic6",
        "method": "linear",
        "scaler": "standard",
        "purpose": "Dynamic linear baseline",
    },
    {
        "run": "SVM dynamic6 RBF",
        "dir": "results/svm_multitech_dynamic_only_rbf_standard",
        "model": "SVM",
        "dataset": "Expanded",
        "feature_set": "dynamic6",
        "method": "RBF",
        "scaler": "standard",
        "purpose": "Main dynamic SVM baseline",
    },
    {
        "run": "SVM dynamic4 linear",
        "dir": "results/svm_multitech_dynamic_4feat_linear_standard",
        "model": "SVM",
        "dataset": "Expanded",
        "feature_set": "dynamic4",
        "method": "linear",
        "scaler": "standard",
        "purpose": "Compact linear baseline",
    },
    {
        "run": "SVM dynamic4 RBF",
        "dir": "results/svm_multitech_dynamic_4feat_rbf_standard",
        "model": "SVM",
        "dataset": "Expanded",
        "feature_set": "dynamic4",
        "method": "RBF",
        "scaler": "standard",
        "purpose": "Compact nonlinear baseline",
    },
    {
        "run": "SVM tddft3 RBF",
        "dir": "results/svm_tddft_only_rbf_standard",
        "model": "SVM",
        "dataset": "Expanded",
        "feature_set": "tddft3",
        "method": "RBF",
        "scaler": "standard",
        "purpose": "TDDFT-only test",
    },
    {
        "run": "SVM md3 RBF",
        "dir": "results/svm_md_only_rbf_standard",
        "model": "SVM",
        "dataset": "Expanded",
        "feature_set": "md3",
        "method": "RBF",
        "scaler": "standard",
        "purpose": "MD-only expanded test",
    },
    {
        "run": "SVM shuffled labels",
        "dir": "results/svm_dynamic6_shuffled_label",
        "model": "SVM",
        "dataset": "Expanded",
        "feature_set": "dynamic6",
        "method": "linear",
        "scaler": "standard",
        "purpose": "Sanity check",
    },
    {
        "run": "SVM unique MD RBF",
        "dir": "results/svm_md3_unique_windows_rbf_minmax",
        "model": "SVM",
        "dataset": "Unique MD windows",
        "feature_set": "md3",
        "method": "RBF",
        "scaler": "minmax",
        "purpose": "Unique-window RBF validation",
    },
    {
        "run": "SVM unique MD linear",
        "dir": "results/svm_md3_unique_windows_linear_minmax",
        "model": "SVM",
        "dataset": "Unique MD windows",
        "feature_set": "md3",
        "method": "linear",
        "scaler": "minmax",
        "purpose": "Unique-window linear validation",
    },
    {
        "run": "QSVM md3 reps1 90",
        "dir": "results/qsvm_md3_reps1_90samples",
        "model": "QSVM",
        "dataset": "Expanded subset",
        "feature_set": "md3",
        "method": "ZZFeatureMap",
        "scaler": "minmax [0, pi]",
        "purpose": "Small QSVM pipeline test",
    },
    {
        "run": "QSVM md3 reps1 300",
        "dir": "results/qsvm_md3_reps1_300samples",
        "model": "QSVM",
        "dataset": "Expanded subset",
        "feature_set": "md3",
        "method": "ZZFeatureMap",
        "scaler": "minmax [0, pi]",
        "purpose": "Larger expanded QSVM test",
    },
    {
        "run": "QSVM md3 reps2 150",
        "dir": "results/qsvm_md3_reps2_150samples",
        "model": "QSVM",
        "dataset": "Expanded subset",
        "feature_set": "md3",
        "method": "ZZFeatureMap",
        "scaler": "minmax [0, pi]",
        "purpose": "Deeper QSVM feature map",
    },
    {
        "run": "QSVM unique MD reps1",
        "dir": "results/qsvm_md3_unique_windows_reps1",
        "model": "QSVM",
        "dataset": "Unique MD windows",
        "feature_set": "md3",
        "method": "ZZFeatureMap",
        "scaler": "minmax [0, pi]",
        "purpose": "Unique-window QSVM reps1",
    },
    {
        "run": "QSVM unique MD reps2",
        "dir": "results/qsvm_md3_unique_windows_reps2",
        "model": "QSVM",
        "dataset": "Unique MD windows",
        "feature_set": "md3",
        "method": "ZZFeatureMap",
        "scaler": "minmax [0, pi]",
        "purpose": "Unique-window QSVM reps2",
    },
    {
        "run": "QSVM dynamic4 reps1 300",
        "dir": "results/qsvm_dynamic4_reps1_300samples",
        "model": "QSVM",
        "dataset": "Expanded subset",
        "feature_set": "dynamic4",
        "method": "ZZFeatureMap",
        "scaler": "minmax [0, pi]",
        "purpose": "Direct comparison with dynamic4 SVM",
    },
]


def load_metrics(run: dict) -> dict | None:
    path = Path(run["dir"]) / "metrics.json"

    if not path.exists():
        print(f"Missing: {path}")
        return None

    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    row = dict(run)
    row["samples"] = metrics.get("samples")
    row["train_samples"] = metrics.get("train_samples")
    row["test_samples"] = metrics.get("test_samples")
    row["n_features"] = metrics.get("n_features")
    row["accuracy"] = metrics.get("accuracy")
    row["balanced_accuracy"] = metrics.get("balanced_accuracy")
    row["cv_balanced_accuracy_mean"] = metrics.get("cv_balanced_accuracy_mean")
    row["cv_balanced_accuracy_std"] = metrics.get("cv_balanced_accuracy_std")
    row["reps"] = metrics.get("reps")
    row["entanglement"] = metrics.get("entanglement")
    row["max_samples"] = metrics.get("max_samples")
    row["metrics_path"] = str(path)

    return row


def load_results() -> pd.DataFrame:
    rows = [row for run in RUNS if (row := load_metrics(run)) is not None]

    if not rows:
        raise FileNotFoundError("No metrics.json files were found.")

    return pd.DataFrame(rows)


def format_score(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def format_cv(row: pd.Series) -> str:
    mean = row.get("cv_balanced_accuracy_mean")
    std = row.get("cv_balanced_accuracy_std")

    if pd.isna(mean):
        return ""

    if pd.isna(std):
        return f"{float(mean):.4f}"

    return f"{float(mean):.4f} +/- {float(std):.4f}"


def save_markdown_table(df: pd.DataFrame, path: Path, title: str, columns: list[str]) -> None:
    lines = [f"# {title}\n\n"]
    lines.append("| " + " | ".join(columns) + " |\n")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|\n")

    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row.get(col, "")
            if col in ["accuracy", "balanced_accuracy"]:
                value = format_score(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |\n")

    path.write_text("".join(lines), encoding="utf-8")


def save_tables(df: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    data = df.copy()
    data["cv_balanced_accuracy"] = data.apply(format_cv, axis=1)
    data["accuracy"] = data["accuracy"].apply(format_score)
    data["balanced_accuracy"] = data["balanced_accuracy"].apply(format_score)

    svm_cols = [
        "run",
        "dataset",
        "feature_set",
        "method",
        "scaler",
        "samples",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "cv_balanced_accuracy",
        "purpose",
    ]

    qsvm_cols = [
        "run",
        "dataset",
        "feature_set",
        "reps",
        "entanglement",
        "max_samples",
        "samples",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "purpose",
    ]

    all_cols = [
        "run",
        "model",
        "dataset",
        "feature_set",
        "method",
        "scaler",
        "samples",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "cv_balanced_accuracy",
        "reps",
        "entanglement",
        "max_samples",
        "purpose",
    ]

    svm = data[data["model"] == "SVM"][svm_cols].copy()
    qsvm = data[data["model"] == "QSVM"][qsvm_cols].copy()
    all_results = data[all_cols].copy()

    svm.to_csv(TABLES_DIR / "table_1_svm_results.csv", index=False)
    qsvm.to_csv(TABLES_DIR / "table_2_qsvm_results.csv", index=False)
    all_results.to_csv(TABLES_DIR / "ml_model_results_all_metrics.csv", index=False)

    save_markdown_table(svm, TABLES_DIR / "table_1_svm_results.md", "Table 1 - SVM results", svm_cols)
    save_markdown_table(qsvm, TABLES_DIR / "table_2_qsvm_results.md", "Table 2 - QSVM results", qsvm_cols)

    print(f"Saved: {TABLES_DIR / 'table_1_svm_results.csv'}")
    print(f"Saved: {TABLES_DIR / 'table_1_svm_results.md'}")
    print(f"Saved: {TABLES_DIR / 'table_2_qsvm_results.csv'}")
    print(f"Saved: {TABLES_DIR / 'table_2_qsvm_results.md'}")
    print(f"Saved: {TABLES_DIR / 'ml_model_results_all_metrics.csv'}")


def save_bar_plot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    data = df.dropna(subset=["balanced_accuracy"]).sort_values("balanced_accuracy")

    plt.figure(figsize=(10, max(5, 0.35 * len(data))))
    plt.barh(data["run"], data["balanced_accuracy"])
    plt.xlabel("Balanced accuracy")
    plt.ylabel("Run")
    plt.title(title)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_unique_md_plot(df: pd.DataFrame) -> None:
    data = df[df["dataset"] == "Unique MD windows"].dropna(subset=["balanced_accuracy"])

    if data.empty:
        return

    save_bar_plot(
        data,
        FIGURES_DIR / "unique_md_window_svm_qsvm_comparison.png",
        "Unique MD-window dataset: SVM vs QSVM",
    )


def save_confusion_matrix(run_name: str, run_dir: str) -> None:
    path = Path(run_dir) / "confusion_matrix.csv"

    if not path.exists():
        print(f"Missing: {path}")
        return

    cm = pd.read_csv(path, index_col=0)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm.values)
    plt.xticks(range(len(cm.columns)), cm.columns)
    plt.yticks(range(len(cm.index)), cm.index)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(run_name)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm.values[i, j]), ha="center", va="center")

    safe_name = run_name.lower().replace(" ", "_").replace("/", "_")
    output_path = FIGURES_DIR / f"confusion_matrix_{safe_name}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_figures(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    save_bar_plot(
        df,
        FIGURES_DIR / "model_comparison_balanced_accuracy.png",
        "SVM and QSVM comparison",
    )

    save_bar_plot(
        df[df["model"] == "SVM"],
        FIGURES_DIR / "svm_balanced_accuracy.png",
        "SVM balanced accuracy",
    )

    save_bar_plot(
        df[df["model"] == "QSVM"],
        FIGURES_DIR / "qsvm_balanced_accuracy.png",
        "QSVM balanced accuracy",
    )

    save_unique_md_plot(df)

    selected = [
        ("SVM dynamic6 RBF", "results/svm_multitech_dynamic_only_rbf_standard"),
        ("SVM dynamic4 RBF", "results/svm_multitech_dynamic_4feat_rbf_standard"),
        ("SVM shuffled labels", "results/svm_dynamic6_shuffled_label"),
        ("SVM unique MD RBF", "results/svm_md3_unique_windows_rbf_minmax"),
        ("QSVM md3 reps1 300", "results/qsvm_md3_reps1_300samples"),
        ("QSVM dynamic4 reps1 300", "results/qsvm_dynamic4_reps1_300samples"),
        ("QSVM unique MD reps1", "results/qsvm_md3_unique_windows_reps1"),
        ("QSVM unique MD reps2", "results/qsvm_md3_unique_windows_reps2"),
    ]

    for run_name, run_dir in selected:
        save_confusion_matrix(run_name, run_dir)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    save_tables(results)
    save_figures(results)

    print("\nDone.")
    print(f"Tables: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()