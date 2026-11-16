"""Summary of the IBM site-memory runs already on disk (never contacts IBM).

Reads every ``site_memory_ibm.py`` output folder given (or found under ``--root`` with
the prefix ``--prefix``) that holds hardware kernels, and writes to ``--output-dir``:

* ``ibm_summary.csv``: one row per folder and window: backend, shots, reps,
  entanglement, bandwidth (A = standard ZZ map, lambda 1; B = matched, ``auto``),
  scaling; correct/total and p of the RBF SVM, the ideal QSVM and the hardware QSVM;
  kernel deviation from the ideal (MAE, RMSE, bias, max); mean transpiled depth and
  two-qubit gate count; reported quantum seconds.
* ``ibm_repeats.csv``: configurations run more than once (same window, shots, reps,
  entanglement, bandwidth and scaling): mean absolute difference between the measured
  kernels of each pair of runs (off-diagonal) and how many hardware predictions agree.
  This is the run-to-run variation against which IBM-ideal differences are judged.
* ``ibm_summary.json``: totals (quantum seconds, circuits, shots) and the folders used.
* ``ibm_summary.png``: correct predictions of the three models per configuration, with
  the chance level.

Folders written by ``--reuse-hardware`` repeat an earlier hardware run and are skipped;
plan-only folders have no hardware kernels and are skipped. All folders must come from
the same MD data (same input hashes); otherwise the script stops.

    python -m scripts.analysis.site_memory_ibm_summary --output-dir results/analyses/site_memory/ibm_summary
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HARDWARE_DIR = "hardware_kernels"
MODELS = ("svm_rbf", "qsvm_ideal", "qsvm_ibm")
CONFIG_KEYS = ("t_start_fs", "shots", "reps", "entanglement", "bandwidth", "scaling")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summary of saved IBM site-memory runs (no IBM access).")
    parser.add_argument("--inputs", nargs="*", type=Path, default=None,
                        help="Output folders of site_memory_ibm.py (default: every <root>/<prefix>* folder).")
    parser.add_argument("--root", type=Path, default=Path("results/analyses/site_memory"))
    parser.add_argument("--prefix", default="ibm_")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def bandwidth_label(value) -> str:
    if value in (None, 1, 1.0, "1"):
        return "A"
    return "B" if value == "auto" else f"lambda={value}"


def data_id(summary: dict) -> str:
    """Hash of the MD input files the run read (same data <=> same id)."""
    hashes = sorted(item["sha256"] for item in summary.get("inputs", []))
    return hashlib.sha256("".join(hashes).encode()).hexdigest()[:12]


def load_folder(folder: Path) -> dict | None:
    """Everything of one hardware run, or None for plan-only and reuse folders."""
    summary_path = folder / "site_memory_ibm_summary.json"
    record_path = folder / HARDWARE_DIR / "hardware_record.json"
    if not summary_path.is_file() or not record_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hardware = summary.get("hardware") or {}
    if hardware.get("reused_from"):
        return None
    return {
        "folder": folder,
        "summary": summary,
        "record": json.loads(record_path.read_text(encoding="utf-8")),
        "windows": pd.read_csv(folder / "site_memory_ibm_windows.csv"),
        "predictions": pd.read_csv(folder / "site_memory_ibm_predictions.csv"),
        "deviation": pd.read_csv(folder / "site_memory_ibm_kernel_deviation.csv")
        if (folder / "site_memory_ibm_kernel_deviation.csv").is_file() else pd.DataFrame(),
    }


def window_kernels(record: dict, window: int) -> dict:
    return {name: kernel for name, kernel in record["kernels"].items() if name.startswith(f"w{window}_")}


def summary_rows(run: dict) -> list[dict]:
    plan, hardware = run["summary"]["plan"], run["summary"].get("hardware") or {}
    rows = []
    for window, part in run["windows"].groupby("window"):
        by_model = {row["model"]: row for _, row in part.iterrows()}
        kernels = window_kernels(run["record"], int(window))
        stats = [kernel.get("circuit_stats") or {} for kernel in kernels.values()]
        seconds = [kernel.get("quantum_seconds") for kernel in kernels.values()]
        deviation = run["deviation"]
        deviation = deviation[deviation["window"] == window] if not deviation.empty else deviation
        row = {
            "folder": run["folder"].name,
            "backend": hardware.get("backend"),
            "window": int(window),
            "t_start_fs": float(part["t_start_fs"].iloc[0]),
            "t_end_fs": float(part["t_end_fs"].iloc[0]),
            "shots": int(hardware.get("shots") or plan.get("shots")),
            "reps": int(plan.get("reps", 1)),
            "entanglement": plan.get("entanglement", "linear"),
            "bandwidth": bandwidth_label(plan.get("qsvm_bandwidth", 1.0)),
            "scaling": plan.get("scaling"),
            "n_trajectories": int(part["n_trajectories"].iloc[0]),
        }
        for model in MODELS:
            item = by_model.get(model)
            row[f"{model}_correct"] = int(item["n_correct"]) if item is not None else None
            row[f"{model}_p"] = float(item["p_value"]) if item is not None else None
        for key in ("kernel_mae", "kernel_rmse", "kernel_bias", "kernel_max_abs"):
            row[key] = float(deviation[key].iloc[0]) if not deviation.empty else None
        circuits = sum(int(s.get("n_circuits") or 0) for s in stats)
        row["circuits"] = circuits
        row["depth_mean"] = (sum(float(s["depth_mean"]) * int(s["n_circuits"]) for s in stats) / circuits
                             if circuits else None)
        row["two_qubit_gates_mean"] = (sum(float(s["two_qubit_gates_mean"]) * int(s["n_circuits"])
                                           for s in stats) / circuits if circuits else None)
        row["quantum_seconds"] = (float(sum(seconds)) if seconds and all(seconds) else None)
        rows.append(row)
    return rows


def repeat_rows(runs: list[dict], table: pd.DataFrame) -> list[dict]:
    """Pairs of runs with the same configuration: kernel difference and prediction agreement."""
    by_name = {run["folder"].name: run for run in runs}
    rows = []
    for config, group in table.groupby(list(CONFIG_KEYS)):
        if len(group) < 2:
            continue
        for (_, first), (_, second) in itertools.combinations(group.iterrows(), 2):
            a_run, b_run = by_name[first["folder"]], by_name[second["folder"]]
            window_a, window_b = int(first["window"]), int(second["window"])
            names = sorted(window_kernels(a_run["record"], window_a))
            differences = []
            for name in names:
                other = name.replace(f"w{window_a}_", f"w{window_b}_", 1)
                a_path = a_run["folder"] / HARDWARE_DIR / f"{name}.npy"
                b_path = b_run["folder"] / HARDWARE_DIR / f"{other}.npy"
                if not (a_path.is_file() and b_path.is_file()):
                    continue
                a, b = np.load(a_path), np.load(b_path)
                if a.shape != b.shape:
                    continue
                delta = np.abs(a - b)
                if delta.ndim == 2 and delta.shape[0] == delta.shape[1]:
                    delta = delta[~np.eye(delta.shape[0], dtype=bool)]
                differences.append(delta.ravel())
            values = np.concatenate(differences) if differences else np.array([])
            pa = a_run["predictions"].query("model == 'qsvm_ibm' and window == @window_a")
            pb = b_run["predictions"].query("model == 'qsvm_ibm' and window == @window_b")
            merged = pa.merge(pb, on=["seed", "site"], suffixes=("_a", "_b"))
            rows.append({
                **{key: first[key] for key in CONFIG_KEYS},
                "folder_a": first["folder"], "folder_b": second["folder"],
                "ibm_correct_a": first["qsvm_ibm_correct"], "ibm_correct_b": second["qsvm_ibm_correct"],
                "kernel_run_to_run_mae": float(values.mean()) if values.size else None,
                "kernel_run_to_run_max": float(values.max()) if values.size else None,
                "same_predictions": int((merged["predicted_site_a"] == merged["predicted_site_b"]).sum()),
                "n_trajectories": int(len(merged)),
            })
    return rows


def config_label(row) -> str:
    extra = []
    if row["shots"] != 512:
        extra.append(f"{row['shots']} shots")
    if row["reps"] != 1:
        extra.append(f"reps {row['reps']}")
    if row["entanglement"] != "linear":
        extra.append(row["entanglement"])
    label = f"{row['t_start_fs']:g} fs, {row['bandwidth']}"
    return label + (f", {', '.join(extra)}" if extra else "")


def plot_summary(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = table.sort_values(["t_start_fs", "bandwidth", "shots", "reps", "folder"]).reset_index(drop=True)
    base = [config_label(row) for _, row in table.iterrows()]
    counts = {label: base.count(label) for label in base}
    seen: dict[str, int] = {}
    labels = []
    for label in base:  # number repeated configurations
        seen[label] = seen.get(label, 0) + 1
        labels.append(f"{label} (run {seen[label]})" if counts[label] > 1 else label)
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(6.0, 0.38 * len(table) + 1.2), constrained_layout=True)
    styles = {"svm_rbf": ("s", "SVM (RBF)"), "qsvm_ideal": ("o", "QSVM ideal"), "qsvm_ibm": ("D", "QSVM IBM")}
    for offset, model in zip((-0.2, 0.0, 0.2), MODELS):
        marker, name = styles[model]
        fraction = table[f"{model}_correct"] / table["n_trajectories"]
        ax.plot(fraction, y + offset, linestyle="none", marker=marker, markersize=5, label=name)
    chance = 1.0 / 3.0
    ax.axvline(chance, color="#777777", linestyle="--", linewidth=0.9, label="Chance")
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Correct predictions (fraction of trajectories)")
    backend = ", ".join(sorted(set(table["backend"].dropna())))
    ax.set_title(f"IBM hardware runs ({backend})", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    folders = args.inputs if args.inputs else sorted(path for path in args.root.glob(f"{args.prefix}*")
                                                     if path.is_dir())
    runs = [run for run in (load_folder(folder.expanduser().resolve()) for folder in folders) if run]
    if not runs:
        raise SystemExit("[ibm-summary] no folder with hardware kernels found (plan-only and reuse folders "
                         "are skipped).")
    ids = {data_id(run["summary"]) for run in runs}
    if len(ids) > 1:
        raise SystemExit("[ibm-summary] the folders come from different MD data; pass --inputs with one data set.")
    table = pd.DataFrame([row for run in runs for row in summary_rows(run)])
    repeats = pd.DataFrame(repeat_rows(runs, table))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "ibm_summary.csv", index=False)
    repeats.to_csv(output / "ibm_repeats.csv", index=False)
    plot_summary(table, output / "ibm_summary.png")
    seconds = table["quantum_seconds"]
    totals = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "folders": [str(run["folder"]) for run in runs],
        "data_id": ids.pop(),
        "backends": sorted(set(table["backend"].dropna())),
        "hardware_windows": int(len(table)),
        "circuits": int(table["circuits"].sum()),
        "shots": int((table["circuits"] * table["shots"]).sum()),
        "quantum_seconds": float(seconds.sum()) if seconds.notna().all() else None,
        "quantum_seconds_missing_in": table.loc[seconds.isna(), "folder"].unique().tolist(),
    }
    (output / "ibm_summary.json").write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    shown = ["folder", "t_start_fs", "bandwidth", "shots", "reps", "svm_rbf_correct", "qsvm_ideal_correct",
             "qsvm_ibm_correct", "n_trajectories", "kernel_mae", "kernel_bias", "depth_mean",
             "two_qubit_gates_mean", "quantum_seconds"]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table[shown].round(4).to_string(index=False))
        if not repeats.empty:
            print(repeats.round(4).to_string(index=False))
    print(f"[ibm-summary] {len(runs)} hardware runs, {totals['circuits']} circuits, quantum time "
          + ("unknown for some runs" if totals["quantum_seconds"] is None else f"{totals['quantum_seconds']:g} s"))
    print(f"[ibm-summary] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
