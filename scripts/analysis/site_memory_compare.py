"""Compare saved site-memory results across temperatures and QSVM settings (no recomputation).

Reads the output folders of ``scripts.analysis.site_memory`` given with ``--inputs``.
Each folder is identified from its summary: MD temperature, number of seeds, and the
QSVM feature map (entanglement, reps). The reference setting is linear entanglement with
one repetition; other settings are sensitivity runs.

Outputs in ``--output-dir``:

* ``site_memory_compare.csv``: one row per folder and model: temperature, seeds, QSVM
  entanglement and reps, memory time (end of the significant run from t = 0, as in
  site_memory.py), balanced accuracy of the first window, and the mean balanced accuracy
  of the windows starting at or after ``--late-from-fs`` (the residual information).
* ``site_memory_compare.png``: one panel per temperature (reference setting), balanced
  accuracy against time for ``--models``; filled markers are windows with p <= alpha;
  the dashed line is chance.
* ``site_memory_compare.json``: the folders used and any warning (for example, different
  numbers of seeds between temperatures, which makes memory times not comparable: the
  smallest attainable p depends on the number of seeds).

    python -m scripts.analysis.site_memory_compare \\
      --inputs results/analyses/site_memory/100k_6seeds results/analyses/site_memory/300k_v2 \\
      --output-dir results/analyses/site_memory/compare
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.analysis.site_memory import MODEL_LABELS, memory_time

DEFAULT_MODELS = ("svm_rbf", "qsvm", "qsvm_bw")
REFERENCE = ("linear", 1)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved site-memory results (no recomputation).")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="Output folders of scripts.analysis.site_memory.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--late-from-fs", type=float, default=600.0,
                        help="Windows starting at or after this time give the late mean accuracy.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def load(folder: Path) -> dict:
    summary_path, windows_path = folder / "site_memory_summary.json", folder / "site_memory_windows.csv"
    if not summary_path.is_file() or not windows_path.is_file():
        raise SystemExit(f"[site-memory-compare] {folder} is not a site_memory output folder.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    kernel = summary.get("qsvm_kernel") or {}
    return {
        "folder": folder,
        "summary": summary,
        "windows": pd.read_csv(windows_path),
        "temperature": float(summary["md_settings"]["temperature_kelvin"]),
        "seeds": len(summary["seeds"]),
        "entanglement": kernel.get("entanglement", "linear"),
        "reps": int(kernel.get("reps", 1)),
    }


def compare_rows(run: dict, models: list[str], late_from: float) -> list[dict]:
    windows, alpha = run["windows"], float(run["summary"]["alpha"])
    rows = []
    for model in models:
        part = windows[windows["model"] == model].sort_values("t_start_fs")
        if part.empty:
            continue
        late = part[part["t_start_fs"] >= late_from - 1e-9]
        rows.append({
            "folder": run["folder"].name,
            "temperature_k": run["temperature"],
            "seeds": run["seeds"],
            "qsvm_entanglement": run["entanglement"],
            "qsvm_reps": run["reps"],
            "reference_setting": (run["entanglement"], run["reps"]) == REFERENCE,
            "model": model,
            "memory_time_fs": memory_time(windows, model, alpha),
            "first_window_accuracy": float(part["balanced_accuracy"].iloc[0]),
            "late_mean_accuracy": float(late["balanced_accuracy"].mean()) if not late.empty else None,
            "late_windows": int(len(late)),
            "chance": float(run["summary"]["chance_level"]),
        })
    return rows


def plot(references: list[dict], models: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(references), figsize=(4.2 * len(references), 3.2), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, run in zip(axes[0], references):
        windows, alpha = run["windows"], float(run["summary"]["alpha"])
        for model in models:
            part = windows[windows["model"] == model].sort_values("t_start_fs")
            if part.empty:
                continue
            centre = 0.5 * (part["t_start_fs"] + part["t_end_fs"])
            line, = ax.plot(centre, part["balanced_accuracy"], marker="o", markersize=3.2, linewidth=1.1,
                            markerfacecolor="white", label=MODEL_LABELS.get(model, model))
            significant = part["p_value"] <= alpha
            ax.plot(centre[significant], part["balanced_accuracy"][significant], linestyle="none", marker="o",
                    markersize=3.2, color=line.get_color())
        ax.axhline(run["summary"]["chance_level"], color="#777777", linestyle="--", linewidth=0.9,
                   label="Chance")
        ax.set_title(f"{run['temperature']:g} K ({run['seeds']} seeds)", fontsize=9)
        ax.set_xlabel("Time from the start of the AIMD (fs)")
        ax.set_ylim(-0.03, 1.05)
    axes[0][0].set_ylabel("Balanced accuracy")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    runs = [load(folder.expanduser().resolve()) for folder in args.inputs]
    widths = {float(run["summary"]["window_fs"]) for run in runs}
    if len(widths) > 1:
        raise SystemExit(f"[site-memory-compare] the folders use different window widths {sorted(widths)}.")
    references = sorted((run for run in runs if (run["entanglement"], run["reps"]) == REFERENCE),
                        key=lambda run: run["temperature"])
    temperatures = [run["temperature"] for run in references]
    if len(set(temperatures)) != len(temperatures):
        raise SystemExit("[site-memory-compare] more than one reference folder (linear, reps 1) per temperature.")
    warnings = []
    if len({run["seeds"] for run in references}) > 1:
        warnings.append("different numbers of seeds between temperatures: memory times are not comparable "
                        "(the smallest attainable p depends on the number of seeds)")
    table = pd.DataFrame([row for run in runs for row in compare_rows(run, args.models, args.late_from_fs)])
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table.sort_values(["temperature_k", "reference_setting", "qsvm_entanglement", "qsvm_reps", "model"],
                      ascending=[True, False, True, True, True]).to_csv(output / "site_memory_compare.csv",
                                                                         index=False)
    if references:
        plot(references, args.models, output / "site_memory_compare.png")
    else:
        warnings.append("no reference folder (linear entanglement, reps 1): no figure")
    (output / "site_memory_compare.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(run["folder"]) for run in runs],
        "reference_setting": {"entanglement": REFERENCE[0], "reps": REFERENCE[1]},
        "late_from_fs": args.late_from_fs,
        "warnings": warnings,
    }, indent=2) + "\n", encoding="utf-8")
    shown = ["temperature_k", "seeds", "qsvm_entanglement", "qsvm_reps", "model", "memory_time_fs",
             "late_mean_accuracy"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table[shown].round(3).to_string(index=False))
    for warning in warnings:
        print(f"[site-memory-compare] warning: {warning}")
    print(f"[site-memory-compare] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
