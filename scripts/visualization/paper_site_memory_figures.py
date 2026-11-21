"""Manuscript figures and tables of the site-memory study, from saved outputs only.

Nothing is recomputed: the items are drawn from the PBE and PBE-D3 simulation runs
(``dft_summary.csv`` and ``neb/<material>/*/neb_run_summary.json``), the folders of
``scripts.analysis.site_memory`` (100 K and 300 K, plus sensitivity folders) and the
folders of ``scripts.analysis.site_memory_ibm_summary``. The TDDFT spectra, the CO2RR
pathway and the structures come from the paper stage (``run_analysis paper``).

Items (each skipped with a message when its inputs are not given):

* ``adsorption_pbe_d3``  (a) adsorption energy, PBE and PBE-D3; (b) C height above the
  topmost surface atom, PBE and PBE-D3.
* ``neb_pbe_d3``  NEB path top metal -> bridge -> top oxygen, PBE and PBE-D3, each
  relative to its own top-metal image.
* ``site_memory``  one panel per temperature: balanced accuracy against time for the SVM
  (RBF), the standard QSVM and the scaled QSVM; the dashed line is a random guess
  (1 / number of sites). Significance is in the table and the data CSV, not on the plot.
* ``ibm_hardware``  one panel per temperature: correct predictions of the SVM, the ideal
  QSVM and the IBM QSVM in each hardware window (512 shots, reps 1; the first run of a
  repeated configuration), standard and scaled QSVM.
* ``table_site_memory``  memory time, first-window and late mean accuracy per model and
  temperature (CSV, Markdown, LaTeX); ``table_adsorption`` E_ads PBE and PBE-D3.
* ``supp_band_gap``  Gamma gap of the clean and adsorbed slabs (PBE).
* ``supp_height_time``  C height above the top plane against time, mean over seeds with
  the seed range, per site, 100 K and 300 K.
* ``supp_qsvm_sensitivity``  memory times of every given site_memory folder (feature-map
  entanglement and reps), as a table.
* ``supp_ibm_runs``  every IBM hardware window of the given summaries, as a table.

Every figure is written as PNG and PDF with a ``_data.csv`` of the plotted numbers;
``registry.csv`` lists every file with its inputs.

    python -m scripts.visualization.paper_site_memory_figures \\
      --pbe-run results/simulations/zno_ic_final --d3-run results/simulations/zno_d3_stage1 \\
      --memory-100k results/analyses/site_memory/100k_6seeds \\
      --memory-300k results/analyses/site_memory/300k_v2 \\
      --sensitivity results/analyses/site_memory/100k_6seeds_* results/analyses/site_memory/300k_circular \\
      --ibm-summaries results/analyses/site_memory/ibm_summary_100k results/analyses/site_memory/ibm_summary_300k \\
      --output-dir results/analyses/site_memory/paper
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.analysis.paper_tables import dft_rows_for, is_finite, ordered_sites, safe_float  # noqa: E402
from scripts.analysis.site_memory import memory_time  # noqa: E402
from scripts.visualization.paper_figures import (  # noqa: E402
    annotate_bars,
    apply_style,
    fig07_frame,
    site_color,
    site_label,
)

ITEMS = ("adsorption_pbe_d3", "neb_pbe_d3", "site_memory", "ibm_hardware", "table_site_memory",
         "table_adsorption", "supp_band_gap", "supp_height_time", "supp_qsvm_sensitivity", "supp_ibm_runs")
METHODS = {"PBE": "#9e9e9e", "PBE-D3": "#3949ab"}
# Models in black and greys (line style and marker tell them apart), so that the site
# colours (blue, orange, green) keep meaning sites in every figure.
MODEL_STYLE = {
    "svm_rbf": ("SVM (RBF)", "#000000", "s", "-"),
    "qsvm": ("QSVM (ZZ, standard)", "#616161", "o", "--"),
    "qsvm_bw": ("QSVM (ZZ, scaled)", "#a0a0a0", "^", ":"),
}
IBM_STYLE = {
    "svm_rbf": ("SVM (RBF)", "#000000", "s"),
    "qsvm_ideal": ("QSVM ideal (simulator)", "#9e9e9e", "o"),
    "qsvm_ibm": ("QSVM IBM (ibm_marrakesh)", "#c62828", "D"),
}
BANDWIDTH_NAMES = {"A": "standard", "B": "scaled"}
MEMORY_MODELS = ("svm_rbf", "qsvm", "qsvm_bw")
CHANCE_LABEL = "Random guess (1/{n})"
FEATURE_HEIGHT = "c_height_top_plane_ang"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Writer:
    """Saves figures and tables and keeps the registry."""

    def __init__(self, output: Path, dpi: int):
        self.output, self.dpi, self.rows = output, dpi, []
        output.mkdir(parents=True, exist_ok=True)

    def figure(self, item: str, fig, data: pd.DataFrame, inputs: list[Path]) -> None:
        for suffix in ("png", "pdf"):
            fig.savefig(self.output / f"{item}.{suffix}", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        data.to_csv(self.output / f"{item}_data.csv", index=False)
        self.register(item, [f"{item}.png", f"{item}.pdf", f"{item}_data.csv"], inputs)

    def table(self, item: str, table: pd.DataFrame, inputs: list[Path], caption: str) -> None:
        table.to_csv(self.output / f"{item}.csv", index=False)
        (self.output / f"{item}.md").write_text(markdown(table, caption), encoding="utf-8")
        (self.output / f"{item}.tex").write_text(latex(table, caption), encoding="utf-8")
        self.register(item, [f"{item}.csv", f"{item}.md", f"{item}.tex"], inputs)

    def register(self, item: str, files: list[str], inputs: list[Path]) -> None:
        sources = ";".join(f"{path}" for path in inputs)
        for name in files:
            self.rows.append({"item": item, "file": name, "sha256": sha256(self.output / name),
                              "inputs": sources})
            print(f"[paper-site-memory] wrote {self.output / name}")

    def close(self) -> None:
        pd.DataFrame(self.rows).to_csv(self.output / "registry.csv", index=False)


def text_cell(value) -> str:
    if isinstance(value, float):
        return "" if not np.isfinite(value) else f"{value:g}"
    return str(value)


def markdown(table: pd.DataFrame, caption: str) -> str:
    lines = [caption, "", "| " + " | ".join(table.columns) + " |", "|" + " --- |" * len(table.columns)]
    lines += ["| " + " | ".join(text_cell(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def latex(table: pd.DataFrame, caption: str) -> str:
    def esc(text: str) -> str:
        return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{esc(caption)}}}",
             r"\begin{tabular}{" + "l" * len(table.columns) + "}", r"\hline",
             " & ".join(esc(str(c)) for c in table.columns) + r" \\", r"\hline"]
    lines += [" & ".join(esc(text_cell(v)) for v in row) + r" \\" for row in table.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold")


# ---------------------------------------------------------------- DFT and NEB
def dft_frame(runs: dict[str, Path], material: str) -> pd.DataFrame:
    records = []
    for method, run in runs.items():
        rows = dft_rows_for(run / "dft_summary.csv", material)
        for site in ordered_sites(rows):
            row = rows[site]
            if row.get("dft_status") != "ok":
                continue
            records.append({
                "method": method, "site": site,
                "adsorption_energy_ev": safe_float(row.get("adsorption_energy_ev")),
                "c_surface_distance_ang": safe_float(row.get("c_surface_distance_ang")),
                "c_min_slab_distance_ang": safe_float(row.get("c_min_slab_distance_ang")),
                "oco_angle_deg": safe_float(row.get("oco_angle_deg")),
                "clean_band_gap_gamma_ev": safe_float(row.get("clean_band_gap_gamma_ev")),
                "adsorbed_band_gap_gamma_ev": safe_float(row.get("adsorbed_band_gap_gamma_ev")),
            })
    return pd.DataFrame(records)


def plot_adsorption(data: pd.DataFrame):
    sites = list(dict.fromkeys(data["site"]))
    methods = [m for m in METHODS if m in set(data["method"])]
    x = np.arange(len(sites))
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True)
    panels = (("adsorption_energy_ev", "Adsorption energy (eV)", "{:+.3f}"),
              ("c_surface_distance_ang", "C height above the\ntopmost surface atom (Å)", "{:.2f}"))
    for ax, letter, (column, ylabel, fmt) in zip(axes, "ab", panels):
        for k, method in enumerate(methods):
            part = data[data["method"] == method].set_index("site").reindex(sites)[column].to_numpy()
            bars = ax.bar(x + (k - (len(methods) - 1) / 2) * width, part, width, color=METHODS[method],
                          label=method)
            annotate_bars(ax, bars, part, fmt)
        if column == "adsorption_energy_ev":
            finite = [abs(v) for v in data[column] if is_finite(v)]
            limit = max(0.1, 1.3 * max(finite, default=0.0))
            ax.axhline(0.0, color="#444444", linewidth=0.6)
            ax.set_ylim(-limit, 0.35 * limit)
        else:
            ax.margins(y=0.18)
        ax.set_xticks(x, [site_label(s).replace(" ", "\n") for s in sites])
        ax.set_ylabel(ylabel)
        panel_letter(ax, letter)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHODS[m]) for m in methods]
    fig.legend(handles, methods, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(methods))
    return fig


def neb_table(run: Path, material: str) -> pd.DataFrame | None:
    rows = []
    for summary in sorted((run / "neb" / material).glob("*/neb_run_summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        energies = data.get("relative_energies") or []
        for index, value in enumerate(energies):
            rows.append({"pair": data.get("path", summary.parent.name), "initial_site": data["initial_site"],
                         "final_site": data["final_site"], "image_index": index,
                         "rel_energy_eV": float(value)})
    return pd.DataFrame(rows) if rows else None


def neb_frame(runs: dict[str, Path], material: str) -> tuple[pd.DataFrame, list[Path]]:
    frames, inputs = [], []
    for method, run in runs.items():
        table = neb_table(run, material)
        if table is None:
            print(f"[skip] neb_pbe_d3: no NEB summaries in {run}")
            continue
        inputs += sorted((run / "neb" / material).glob("*/neb_run_summary.json"))
        frame = fig07_frame(table)
        frame.insert(0, "method", method)
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), inputs


def plot_neb(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
    for method in [m for m in METHODS if m in set(data["method"])]:
        part = data[data["method"] == method].drop_duplicates("global_index").sort_values("global_index")
        ax.plot(part["global_index"], part["rel_energy_ev"], marker="o", color=METHODS[method], label=method)
    ax.axhline(0.0, color="#555555", linewidth=0.7, linestyle="--")
    ends = data[data["endpoint_site"].fillna("") != ""][["global_index", "endpoint_site"]].drop_duplicates()
    for _, row in ends.iterrows():
        ax.axvline(row["global_index"], color="#bdbdbd", linewidth=0.6, zorder=0)
        ax.text(row["global_index"], 1.01, site_label(str(row["endpoint_site"])),
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(sorted(set(int(x) for x in data["global_index"])))
    ax.set_xlabel("Reaction coordinate (NEB image)")
    ax.set_ylabel("Energy relative to top metal (eV)")
    ax.legend(loc="lower right")
    return fig


# ---------------------------------------------------------------- site memory
def load_memory(folder: Path) -> dict:
    summary = json.loads((folder / "site_memory_summary.json").read_text(encoding="utf-8"))
    return {"folder": folder, "summary": summary, "windows": pd.read_csv(folder / "site_memory_windows.csv"),
            "temperature": float(summary["md_settings"]["temperature_kelvin"]), "seeds": len(summary["seeds"])}


def plot_memory(runs: list[dict]):
    fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 2.9), sharey=True, constrained_layout=True,
                             squeeze=False)
    records = []
    n_sites = len(runs[0]["summary"]["sites"])
    for ax, letter, run in zip(axes[0], "abcd", runs):
        alpha = float(run["summary"]["alpha"])
        for model in MEMORY_MODELS:
            label, color, marker, style = MODEL_STYLE[model]
            part = run["windows"][run["windows"]["model"] == model].sort_values("t_start_fs")
            if part.empty:
                continue
            centre = 0.5 * (part["t_start_fs"] + part["t_end_fs"]).to_numpy()
            accuracy = part["balanced_accuracy"].to_numpy()
            significant = (part["p_value"] <= alpha).to_numpy()
            ax.plot(centre, accuracy, color=color, linestyle=style, linewidth=1.2, marker=marker, markersize=3.6,
                    zorder=3 if model == "svm_rbf" else 2)
            records += [{"temperature_k": run["temperature"], "model": model, "t_centre_fs": c,
                         "balanced_accuracy": a, "p_value": p, "significant": bool(s)}
                        for c, a, p, s in zip(centre, accuracy, part["p_value"], significant)]
        ax.axhline(1.0 / n_sites, color="#c62828", linestyle="--", linewidth=0.9)
        ax.set_title(f"{run['temperature']:g} K", fontsize=9.5)
        ax.set_xlabel("Time from the start of the AIMD (fs)")
        ax.set_ylim(0.0, 1.05)
        panel_letter(ax, letter)
    axes[0][0].set_ylabel("Balanced accuracy")
    handles = [Line2D([], [], color=MODEL_STYLE[m][1], marker=MODEL_STYLE[m][2], linestyle=MODEL_STYLE[m][3],
                      markersize=4.5, label=MODEL_STYLE[m][0]) for m in MEMORY_MODELS]
    handles.append(Line2D([], [], color="#c62828", linestyle="--", label=CHANCE_LABEL.format(n=n_sites)))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=7.2)
    return fig, pd.DataFrame(records)


def memory_rows(run: dict, models, late_from: float) -> list[dict]:
    windows, alpha = run["windows"], float(run["summary"]["alpha"])
    kernel = run["summary"].get("qsvm_kernel") or {}
    rows = []
    for model in models:
        part = windows[windows["model"] == model].sort_values("t_start_fs")
        if part.empty:
            continue
        late = part[part["t_start_fs"] >= late_from - 1e-9]
        end = float(part["t_end_fs"].max())
        memory = memory_time(windows, model, alpha)
        # Significant up to the last window: the memory outlasts the trajectory (whose last
        # frame is the start of the last window).
        shown = f">= {part['t_start_fs'].max():g} (whole run)" if memory >= end else f"{memory:g}"
        rows.append({
            "Temperature (K)": run["temperature"], "Seeds": run["seeds"],
            "Entanglement": kernel.get("entanglement", "linear"), "Reps": int(kernel.get("reps", 1)),
            "Model": MODEL_STYLE.get(model, (model,))[0],
            "Memory time (fs)": shown,
            "First window accuracy": round(float(part["balanced_accuracy"].iloc[0]), 3),
            f"Mean accuracy from {late_from:g} fs": round(float(late["balanced_accuracy"].mean()), 3)
            if not late.empty else float("nan"),
        })
    return rows


def height_frame(runs: list[dict]) -> pd.DataFrame:
    records = []
    for run in runs:
        features = pd.read_csv(run["folder"] / "site_memory_features.csv")
        if FEATURE_HEIGHT not in features.columns:
            continue
        grouped = features.groupby(["site", "t_start_fs", "t_end_fs"])[FEATURE_HEIGHT]
        for (site, start, end), values in grouped:
            records.append({"temperature_k": run["temperature"], "site": site,
                            "t_centre_fs": 0.5 * (start + end), "mean_height_ang": values.mean(),
                            "min_height_ang": values.min(), "max_height_ang": values.max(),
                            "seeds": int(values.size)})
    return pd.DataFrame(records)


def plot_height(data: pd.DataFrame):
    temperatures = sorted(set(data["temperature_k"]))
    fig, axes = plt.subplots(1, len(temperatures), figsize=(3.3 * len(temperatures), 2.8), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, letter, temperature in zip(axes[0], "abcd", temperatures):
        part = data[data["temperature_k"] == temperature]
        for index, site in enumerate(ordered_sites(part["site"])):
            rows = part[part["site"] == site].sort_values("t_centre_fs")
            color = site_color(site, index)
            ax.fill_between(rows["t_centre_fs"], rows["min_height_ang"], rows["max_height_ang"], color=color,
                            alpha=0.15, linewidth=0)
            ax.plot(rows["t_centre_fs"], rows["mean_height_ang"], color=color, label=site_label(site))
        ax.set_title(f"{temperature:g} K", fontsize=9.5)
        ax.set_xlabel("Time from the start of the AIMD (fs)")
        panel_letter(ax, letter)
    axes[0][0].set_ylabel("C height above the top plane (Å)")
    axes[0][-1].legend(loc="upper right")
    return fig


# ---------------------------------------------------------------- IBM
def ibm_frame(folders: list[Path]) -> tuple[pd.DataFrame, list[Path]]:
    frames, inputs = [], []
    for folder in folders:
        path = folder / "ibm_summary.csv"
        if not path.is_file():
            print(f"[skip] ibm: {path} not found")
            continue
        table = pd.read_csv(path)
        temperature = None
        for name in sorted(set(table["folder"])):
            match = [token for token in name.split("_") if token.endswith("k") and token[:-1].isdigit()]
            temperature = float(match[0][:-1]) if match else temperature
        table.insert(0, "temperature_k", temperature)
        frames.append(table)
        inputs.append(path)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), inputs


def ibm_main_rows(table: pd.DataFrame) -> pd.DataFrame:
    main = table[(table["shots"] == 512) & (table["reps"] == 1)].sort_values("folder")
    return main.drop_duplicates(["temperature_k", "t_start_fs", "bandwidth"], keep="first")


def plot_ibm(table: pd.DataFrame):
    temperatures = sorted(set(table["temperature_k"].dropna()))
    counts = [len(table[table["temperature_k"] == t]) for t in temperatures]
    # Panels stacked, each as tall as its number of windows, so the rows have one spacing.
    fig, axes = plt.subplots(len(temperatures), 1, figsize=(4.6, 0.36 * sum(counts) + 0.9 * len(temperatures)),
                             sharex=True, constrained_layout=True, squeeze=False,
                             gridspec_kw={"height_ratios": counts})
    records = []
    for ax, letter, temperature in zip(axes[:, 0], "abcd", temperatures):
        part = table[table["temperature_k"] == temperature].sort_values(["t_start_fs", "bandwidth"])
        labels = [f"{t:g} fs, {BANDWIDTH_NAMES.get(b, b)}" for t, b in zip(part["t_start_fs"], part["bandwidth"])]
        y = np.arange(len(part))
        for offset, model in zip((-0.22, 0.0, 0.22), IBM_STYLE):
            name, color, marker = IBM_STYLE[model]
            fraction = (part[f"{model}_correct"] / part["n_trajectories"]).to_numpy()
            ax.scatter(fraction, y + offset, marker=marker, color=color, s=22, zorder=3)
        for k, (_, row) in enumerate(part.iterrows()):
            ax.text(1.03, k + 0.22, f"{int(row['qsvm_ibm_correct'])}/{int(row['n_trajectories'])}",
                    va="center", fontsize=7, color=IBM_STYLE["qsvm_ibm"][1])
            records.append({"temperature_k": temperature, "window_start_fs": row["t_start_fs"],
                            "qsvm": BANDWIDTH_NAMES.get(row["bandwidth"], row["bandwidth"]),
                            **{f"{m}_correct": int(row[f"{m}_correct"]) for m in IBM_STYLE},
                            "n_trajectories": int(row["n_trajectories"]), "folder": row["folder"]})
        ax.axvline(1.0 / 3.0, color="#757575", linestyle="--", linewidth=0.9)
        ax.set_yticks(y, labels, fontsize=7.5)
        ax.set_ylim(len(part) - 0.5, -0.5)
        ax.set_xlim(0.0, 1.12)
        ax.text(0.01, 1.02, f"{temperature:g} K", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
        panel_letter(ax, letter)
    axes[-1, 0].set_xlabel("Correct predictions (fraction of 18 trajectories)")
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=5, label=n)
               for n, c, m in IBM_STYLE.values()]
    handles.append(Line2D([], [], color="#757575", linestyle="--", label=CHANCE_LABEL.format(n=3)))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=7.2)
    return fig, pd.DataFrame(records)


# ---------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figures and tables of the site-memory study (no recomputation).")
    parser.add_argument("--pbe-run", type=Path, default=None, help="PBE simulation run (DFT and NEB).")
    parser.add_argument("--d3-run", type=Path, default=None, help="PBE-D3 simulation run (DFT and NEB).")
    parser.add_argument("--memory-100k", type=Path, default=None, help="site_memory folder at 100 K.")
    parser.add_argument("--memory-300k", type=Path, default=None, help="site_memory folder at 300 K.")
    parser.add_argument("--sensitivity", nargs="*", type=Path, default=[],
                        help="Further site_memory folders (other entanglement or reps).")
    parser.add_argument("--ibm-summaries", nargs="*", type=Path, default=[],
                        help="Folders of site_memory_ibm_summary (one per temperature).")
    parser.add_argument("--late-from-fs", type=float, default=600.0)
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--items", nargs="+", choices=ITEMS, default=list(ITEMS))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    apply_style()
    out = Writer(args.output_dir.expanduser().resolve(), args.dpi)
    wanted = set(args.items)
    runs = {name: path.expanduser().resolve() for name, path in (("PBE", args.pbe_run), ("PBE-D3", args.d3_run))
            if path is not None}
    memories = [load_memory(path.expanduser().resolve()) for path in (args.memory_100k, args.memory_300k)
                if path is not None]
    memories.sort(key=lambda run: run["temperature"])

    if runs and wanted & {"adsorption_pbe_d3", "table_adsorption", "supp_band_gap"}:
        dft = dft_frame(runs, args.material)
        sources = [run / "dft_summary.csv" for run in runs.values()]
        if "adsorption_pbe_d3" in wanted:
            out.figure("adsorption_pbe_d3", plot_adsorption(dft), dft, sources)
        if "table_adsorption" in wanted:
            table = dft.pivot(index="site", columns="method", values="adsorption_energy_ev")
            table = table.reindex(ordered_sites(table.index)).round(3).reset_index()
            table.insert(0, "Site", [site_label(s) for s in table.pop("site")])
            table.columns = ["Site"] + [f"E_ads {m} (eV)" for m in table.columns[1:]]
            out.table("table_adsorption", table, sources, "Adsorption energies of CO2 on ZnO(10-10).")
        if "supp_band_gap" in wanted and "PBE" in runs:
            pbe = dft[dft["method"] == "PBE"]
            fig, ax = plt.subplots(figsize=(3.2, 2.5), constrained_layout=True)
            x = np.arange(len(pbe))
            ax.bar(x - 0.19, pbe["clean_band_gap_gamma_ev"], 0.38, color="#90caf9", label="Clean slab")
            ax.bar(x + 0.19, pbe["adsorbed_band_gap_gamma_ev"], 0.38, color="#ef9a9a", label="Adsorbed CO$_2$")
            ax.set_xticks(x, [site_label(s).replace(" ", "\n") for s in pbe["site"]])
            ax.set_ylabel("Band gap at Γ (eV, PBE)")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
            out.figure("supp_band_gap", fig, pbe, [runs["PBE"] / "dft_summary.csv"])
    elif wanted & {"adsorption_pbe_d3", "table_adsorption", "supp_band_gap"}:
        print("[skip] DFT items: give --pbe-run and/or --d3-run")

    if "neb_pbe_d3" in wanted:
        neb, sources = neb_frame(runs, args.material) if runs else (pd.DataFrame(), [])
        if neb.empty:
            print("[skip] neb_pbe_d3: no NEB data")
        else:
            out.figure("neb_pbe_d3", plot_neb(neb), neb, sources)

    memory_inputs = [run["folder"] / "site_memory_windows.csv" for run in memories]
    if memories and "site_memory" in wanted:
        fig, data = plot_memory(memories)
        out.figure("site_memory", fig, data, memory_inputs)
    if memories and "table_site_memory" in wanted:
        table = pd.DataFrame([row for run in memories for row in memory_rows(run, MEMORY_MODELS, args.late_from_fs)])
        out.table("table_site_memory", table.drop(columns=["Entanglement", "Reps"]), memory_inputs,
                  "Site memory: end of the significant run of windows from t = 0 (p <= 0.05) and accuracies; "
                  "random guess = 1/3.")
    if memories and "supp_height_time" in wanted:
        heights = height_frame(memories)
        if not heights.empty:
            out.figure("supp_height_time", plot_height(heights), heights,
                       [run["folder"] / "site_memory_features.csv" for run in memories])
    if not memories and wanted & {"site_memory", "table_site_memory", "supp_height_time"}:
        print("[skip] site-memory items: give --memory-100k and/or --memory-300k")

    if "supp_qsvm_sensitivity" in wanted:
        folders = [load_memory(path.expanduser().resolve()) for path in args.sensitivity]
        everything = sorted(memories + folders, key=lambda run: (run["temperature"], str(run["folder"])))
        if everything:
            table = pd.DataFrame([row for run in everything
                                  for row in memory_rows(run, MEMORY_MODELS, args.late_from_fs)])
            table = table.drop_duplicates()
            out.table("supp_qsvm_sensitivity", table,
                      [run["folder"] / "site_memory_windows.csv" for run in everything],
                      "QSVM feature-map sensitivity of the site memory (the SVM rows do not depend on it).")

    if wanted & {"ibm_hardware", "supp_ibm_runs"}:
        ibm, sources = ibm_frame([path.expanduser().resolve() for path in args.ibm_summaries])
        if ibm.empty:
            print("[skip] IBM items: give --ibm-summaries")
        else:
            if "ibm_hardware" in wanted:
                fig, data = plot_ibm(ibm_main_rows(ibm))
                out.figure("ibm_hardware", fig, data, sources)
            if "supp_ibm_runs" in wanted:
                columns = ["temperature_k", "folder", "t_start_fs", "bandwidth", "shots", "reps",
                           "svm_rbf_correct", "qsvm_ideal_correct", "qsvm_ibm_correct", "n_trajectories",
                           "kernel_mae", "kernel_bias", "depth_mean", "two_qubit_gates_mean", "quantum_seconds"]
                table = ibm[[c for c in columns if c in ibm.columns]].copy()
                table["bandwidth"] = table["bandwidth"].map(lambda b: BANDWIDTH_NAMES.get(b, b))
                out.table("supp_ibm_runs", table.round(4), sources, "Every IBM hardware window (ibm_marrakesh).")
    out.close()
    print(f"[paper-site-memory] {len({row['item'] for row in out.rows})} items written to {out.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
