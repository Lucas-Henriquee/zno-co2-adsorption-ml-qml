"""Manuscript figures fig03-fig07 in a simple style (one idea per figure).

Inputs (RUN = simulation run, A = analysis run):

* fig03: ``RUN/dft_summary.csv`` (required for fig03): E_ads, C-surface distance
  (``c_surface_distance_ang``, C height above the topmost slab atom) and the Gamma gap of
  the clean and adsorbed slab; only rows with ``dft_status`` ok.
* fig04: ``RUN/tddft_summary.csv`` and ``RUN/tddft/<MAT>/<site>/transitions.csv``.
* fig05: ``A/science/co2rr/co2rr_corrected_summary.csv`` (one variant, default c).
* fig06: ``A/science/md/md_trajectory_analysis_summary.csv`` and
  ``RUN/md/<MAT>/<site>/md_frame_analysis.csv`` (one seed per site).
* fig07: ``A/science/neb/neb_image_energies.csv``: the NEB pairs are chained into one
  migration path top_metal -> bridge -> top_oxygen (energies relative to the first image of
  the chain); pairs that cannot be chained are drawn on the same axes, each relative to its
  own initial image.

Every figure is written as ``figNN_<name>.png`` and ``.pdf`` together with
``figNN_<name>_data.csv`` holding exactly the plotted numbers. A missing optional input
skips its figure with a ``[skip]`` line; a missing ``dft_summary.csv`` is an error.
Caveats (uncertainties, unresolved barriers, desorption) belong in the captions and
Table 2, not in the figures; titles carry no ranking.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.paper_tables import (  # noqa: E402
    SITE_DISPLAY,
    co2rr_validity,
    dft_rows_for,
    is_finite,
    ordered_sites,
    read_rows,
    require_file,
    safe_float,
)

FIGURES = ("fig03", "fig04", "fig05", "fig06", "fig07")
FIGURE_NAMES = {
    "fig03": "fig03_dft_descriptors",
    "fig04": "fig04_tddft_spectra",
    "fig05": "fig05_co2rr_pathway",
    "fig06": "fig06_md_distance",
    "fig07": "fig07_neb_profiles",
}
SITE_COLORS = {"top_metal": "#1f77b4", "top_oxygen": "#ff7f0e", "bridge": "#2ca02c"}
EXTRA_COLORS = ("#9467bd", "#8c564b", "#e377c2", "#17becf")
GAP_COLORS = {"clean_band_gap_gamma_ev": "#90caf9", "adsorbed_band_gap_gamma_ev": "#ef9a9a"}
GAP_LABELS = {"clean_band_gap_gamma_ev": "Clean slab", "adsorbed_band_gap_gamma_ev": "Adsorbed CO$_2$"}
CO2RR_STATES = (
    ("CO2(g)", None),
    ("COOH*", "dG_CO2_to_COOH_ev"),
    ("CO*", "dG_COOH_to_CO_ev"),
    ("CO(g)", "dG_CO_desorption_ev"),
)
STATE_LABELS = {"CO2(g)": "CO$_2$(g)", "COOH*": "COOH*", "CO*": "CO*", "CO(g)": "CO(g)"}
DEFAULT_DESORPTION_THRESHOLD_ANG = 4.5  # md_trajectory_analysis.DEFAULT_DESORPTION_THRESHOLD_ANG
DEFAULT_MD_SEED = "1234"
TITLES = {
    "fig03": "DFT descriptors of CO$_2$ on {material}",
    "fig04": "TDDFT spectra of the isolated CO$_2$ (adsorbed geometries)",
    "fig05": "CO$_2$ → CO free-energy profile (U = 0 V vs RHE)",
    "fig06": "AIMD: CO$_2$–surface distance (seed {seed})",
    "fig07": "NEB migration path",
}
# fig04: spectral window, wavelength ticks of the top axis and shaded regions.
HC_EV_NM = 1239.841984
FIG04_E_MIN_EV = 3.0
FIG04_E_MAX_MIN_EV = 13.0
FIG04_WAVELENGTH_TICKS_NM = (400, 300, 200, 150, 120, 100)
FIG04_REGIONS = (
    # (label, lower eV, upper eV or None = window end, colour)
    ("UV (400–200 nm)", HC_EV_NM / 400.0, HC_EV_NM / 200.0, "#ede7f6"),
    ("Vacuum UV (< 200 nm)", HC_EV_NM / 200.0, None, "#eceff1"),
)
FIG04_LABEL_X_EV = 7.0
# fig05 legend text of a site that has no U_L.
CO2RR_INVALID_TEXT = {
    "unbound": "intermediates desorbed",
    "wrong_intermediate": "not COOH*/CO*",
}
# fig07: order of the sites along the migration path.
NEB_PATH_ORDER = ("top_metal", "bridge", "top_oxygen")
NEB_SEGMENT_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b")


def skip(message: str) -> None:
    print(f"[skip] {message}")


def site_label(site: str) -> str:
    return SITE_DISPLAY.get(site, site.replace("_", " "))


def site_color(site: str, index: int = 0) -> str:
    return SITE_COLORS.get(site, EXTRA_COLORS[index % len(EXTRA_COLORS)])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 5.0,
    })


def save(fig, data: pd.DataFrame, output_dir: Path, name: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{name}.png", output_dir / f"{name}.pdf", output_dir / f"{name}_data.csv"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    data.to_csv(paths[2], index=False)
    for path in paths:
        print(f"[paper-figures] wrote {path}")
    return paths


def annotate_bars(ax, bars, values, fmt: str) -> None:
    """Value above a positive bar, below a negative one."""
    for bar, value in zip(bars, values):
        if not is_finite(value):
            continue
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 2 if value >= 0 else -2), textcoords="offset points",
            ha="center", va="bottom" if value >= 0 else "top", fontsize=7.0,
        )


# ---------------------------------------------------------------- fig03
FIG03_PANELS = (
    ("adsorption_energy", ("adsorption_energy_ev",), "eV", "Adsorption energy (eV)", "(a) Adsorption energy"),
    ("c_surface_distance", ("c_surface_distance_ang",), "Angstrom", "C–surface distance (Å)",
     "(b) C–surface distance"),
    ("band_gap", ("clean_band_gap_gamma_ev", "adsorbed_band_gap_gamma_ev"), "eV", "Band gap at Γ (eV)",
     "(c) Band gap"),
)


def fig03_data(results_dir: Path, material: str) -> pd.DataFrame:
    dft = dft_rows_for(require_file(results_dir / "dft_summary.csv", "DFT summary for fig03"), material)
    sites = [site for site in ordered_sites(dft) if dft[site].get("dft_status") == "ok"]
    if not sites:
        raise SystemExit(f"[paper-figures] no {material} site with dft_status ok for fig03")
    records = []
    for panel, keys, unit, _ylabel, _title in FIG03_PANELS:
        values = [(site, key, safe_float(dft[site].get(key))) for key in keys for site in sites]
        if not any(is_finite(value) for _, _, value in values):
            skip(f"fig03 panel {panel}: no {'/'.join(keys)} values")
            continue
        for site, key, value in values:
            records.append({"panel": panel, "site": site, "quantity": key, "value": value, "unit": unit})
    return pd.DataFrame(records)


def plot_fig03(data: pd.DataFrame, material: str):
    panels = [p for p in FIG03_PANELS if p[0] in set(data["panel"])]
    sites = list(dict.fromkeys(data["site"]))
    x = np.arange(len(sites))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.35 * len(panels), 2.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (panel, keys, _unit, ylabel, title) in zip(axes, panels):
        part = data[data["panel"] == panel]
        if panel == "band_gap":
            width = 0.38
            for offset, key in zip((-width / 2, width / 2), keys):
                values = part[part["quantity"] == key].set_index("site").reindex(sites)["value"].to_numpy()
                ax.bar(x + offset, values, width, color=GAP_COLORS[key], label=GAP_LABELS[key])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7.0,
                      handlelength=1.2, columnspacing=0.8)
        else:
            values = part.set_index("site").reindex(sites)["value"].to_numpy()
            bars = ax.bar(x, values, 0.6, color=[site_color(s, i) for i, s in enumerate(sites)])
            annotate_bars(ax, bars, values, "{:+.3f}" if panel == "adsorption_energy" else "{:.2f}")
        if panel == "adsorption_energy":
            # Symmetric axis around zero: bars may point up (unfavourable) or down (favourable).
            finite = [abs(v) for v in part["value"] if is_finite(v)]
            limit = max(0.1, 1.2 * max(finite, default=0.0))
            ax.axhline(0.0, color="#444444", linewidth=0.6)
            ax.set_ylim(-limit, limit)
        else:
            ax.margins(y=0.15)
        ax.set_xticks(x)
        ax.set_xticklabels([site_label(s).replace(" ", "\n") for s in sites])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    fig.suptitle(TITLES["fig03"].format(material=material), fontsize=10)
    return fig


# ---------------------------------------------------------------- fig04
def gaussian_broaden(grid: np.ndarray, energies: np.ndarray, osc: np.ndarray, sigma: float) -> np.ndarray:
    """sum_i f_i g(E - E_i) with a unit-area Gaussian g (eV^-1)."""
    if len(energies) == 0:
        return np.zeros_like(grid)
    delta = (grid[:, None] - energies[None, :]) / sigma
    return (np.exp(-0.5 * delta**2) / (sigma * math.sqrt(2.0 * math.pi))) @ osc


def curve_label(site: str) -> str:
    """Site name of the fig04 curves and peak labels (draft style: top-metal)."""
    return site.replace("_", "-")


def fig04_window(spectra: dict[str, pd.DataFrame], sigma: float) -> tuple[float, float]:
    """3 eV to max(13 eV, last written transition + 4 sigma)."""
    last = max(float(t["energy_ev"].max()) for t in spectra.values())
    return FIG04_E_MIN_EV, max(FIG04_E_MAX_MIN_EV, last + 4.0 * sigma)


def fig04_data(results_dir: Path, material: str, sigma: float) -> pd.DataFrame | None:
    summary_path = results_dir / "tddft_summary.csv"
    if not summary_path.exists():
        skip(f"fig04: {summary_path} not found")
        return None
    rows = {row["site"]: row for row in read_rows(summary_path)
            if row.get("material") == material and row.get("tddft_status") == "ok"}
    spectra = {}
    for site in ordered_sites(rows):
        path = results_dir / "tddft" / material / site / "transitions.csv"
        if not path.exists():
            skip(f"fig04 {site}: {path} not found")
            continue
        table = pd.read_csv(path)
        if table.empty:
            skip(f"fig04 {site}: {path} has no transitions")
            continue
        spectra[site] = table.sort_values("energy_ev")
    if not spectra:
        skip(f"fig04: no {material} site with tddft_status ok and transitions.csv")
        return None

    # transitions.csv holds only the lowest transitions (run_tddft --max-transitions): a
    # curve is truncated above its last written transition (stated in the caption).
    e_min, e_max = fig04_window(spectra, sigma)
    grid = np.linspace(e_min, e_max, 1200)
    curves = {
        site: gaussian_broaden(grid, table["energy_ev"].to_numpy(dtype=float),
                               table["oscillator_strength"].to_numpy(dtype=float), sigma)
        for site, table in spectra.items()
    }
    # One common factor (global maximum): relative intensities between the sites are kept.
    scale = max(float(curve.max()) for curve in curves.values())
    if not scale > 0.0:
        skip("fig04: every oscillator strength is zero in the plotted window")
        return None
    records = []
    for site, curve in curves.items():
        peak = int(np.argmax(curve))
        label = f"{material} | {curve_label(site)}"
        for energy, value in zip(grid, curve):
            records.append({"site": site, "curve": label, "energy_ev": energy, "intensity_per_ev": value,
                            "absorption_au": value / scale, "sigma_ev": sigma,
                            "peak_energy_ev": float(grid[peak]), "peak_absorption_au": float(curve[peak] / scale)})
    return pd.DataFrame(records)


def energy_to_nm(energy):
    energy = np.asarray(energy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(energy > 0, HC_EV_NM / np.where(energy > 0, energy, 1.0), np.inf)


def plot_fig04(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    e_min, e_max = float(data["energy_ev"].min()), float(data["energy_ev"].max())
    for label, lower, upper, color in FIG04_REGIONS:
        upper = e_max if upper is None else min(upper, e_max)
        ax.axvspan(lower, upper, color=color, alpha=0.6, linewidth=0, zorder=0)
        for edge in (lower, upper):
            if e_min < edge < e_max:
                ax.axvline(edge, color="#555555", linewidth=0.7, linestyle="--", zorder=1)
        ax.text(0.5 * (lower + upper), 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5, color="#444444")

    sites = list(dict.fromkeys(data["site"]))
    label_y = np.linspace(0.80, 0.80 - 0.17 * (len(sites) - 1), len(sites))
    for i, site in enumerate(sites):
        part = data[data["site"] == site]
        color = site_color(site, i)
        ax.plot(part["energy_ev"], part["absorption_au"], color=color, label=str(part["curve"].iloc[0]), zorder=3)
        peak_e, peak_y = float(part["peak_energy_ev"].iloc[0]), float(part["peak_absorption_au"].iloc[0])
        ax.annotate(f"{curve_label(site)}\n{peak_e:.2f} eV", xy=(peak_e, peak_y),
                    xytext=(FIG04_LABEL_X_EV, label_y[i]), textcoords="data", ha="left", va="center",
                    fontsize=7.0, color=color, zorder=4,
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.6, "shrinkA": 2, "shrinkB": 1,
                                "relpos": (1.0, 0.5)})
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Normalized absorption (a.u.)")
    top = ax.secondary_xaxis("top", functions=(energy_to_nm, energy_to_nm))
    top.set_xticks([nm for nm in FIG04_WAVELENGTH_TICKS_NM if e_min <= HC_EV_NM / nm <= e_max])
    top.set_xlabel("Wavelength (nm)")
    # Legend in the UV region (left, centre) so it covers neither the curves nor the labels.
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), fontsize=7.0, handlelength=1.5)
    return fig


# ---------------------------------------------------------------- fig05
def validity_text(validity: str) -> str:
    if validity in CO2RR_INVALID_TEXT:
        return CO2RR_INVALID_TEXT[validity]
    return validity.replace("_", " ") if validity else "no data"


def fig05_label(site: str, validity: str, u_l: float) -> str:
    """Valid: 'Top metal (U_L = -1.98 V)'; else 'Top oxygen (intermediates desorbed)' etc."""
    if validity == "ok":
        return site_label(site) + (f" (U$_\\mathrm{{L}}$ = {u_l:.2f} V)" if is_finite(u_l) else "")
    return f"{site_label(site)} ({validity_text(validity)})"


def fig05_data(analysis_dir: Path, material: str, variant: str) -> pd.DataFrame | None:
    path = analysis_dir / "science" / "co2rr" / "co2rr_corrected_summary.csv"
    if not path.exists():
        skip(f"fig05: {path} not found")
        return None
    rows = {row["site"]: row for row in read_rows(path)
            if row.get("material") == material and row.get("variant") == variant}
    if not rows:
        skip(f"fig05: no {material} rows of variant {variant!r} in {path}")
        return None
    records = []
    for site in ordered_sites(rows):
        row = rows[site]
        validity = co2rr_validity(row)
        u_l = safe_float(row.get("limiting_potential_v")) if validity == "ok" else float("nan")
        total = 0.0
        for index, (state, key) in enumerate(CO2RR_STATES):
            if key is not None:
                total += safe_float(row.get(key))
            records.append({"site": site, "variant": variant, "state_index": index, "state": state,
                            "dG_ev": total, "pathway_validity": validity, "limiting_potential_v": u_l})
    return pd.DataFrame(records)


def plot_fig05(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    for i, site in enumerate(dict.fromkeys(data["site"])):
        part = data[data["site"] == site].sort_values("state_index")
        validity = str(part["pathway_validity"].iloc[0])
        u_l = float(part["limiting_potential_v"].iloc[0])
        ax.plot(part["state_index"], part["dG_ev"], marker="o", linestyle="-", color=site_color(site, i),
                label=fig05_label(site, validity, u_l))
    ax.axhline(0.0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_xticks(range(len(CO2RR_STATES)))
    ax.set_xticklabels([STATE_LABELS[state] for state, _ in CO2RR_STATES])
    ax.set_xlabel("Reaction coordinate")
    ax.set_ylabel("ΔG (eV)")
    ax.set_title(TITLES["fig05"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    return fig


# ---------------------------------------------------------------- fig06
def fig06_data(results_dir: Path, analysis_dir: Path, material: str, seed: str) -> pd.DataFrame | None:
    summary_path = analysis_dir / "science" / "md" / "md_trajectory_analysis_summary.csv"
    if not summary_path.exists():
        skip(f"fig06: {summary_path} not found")
        return None
    summary = pd.read_csv(summary_path, dtype={"seed": str})
    summary = summary[(summary["material"] == material) & (summary["n_production_frames"].fillna(0) > 0)]
    if summary.empty:
        skip(f"fig06: no {material} trajectory with production frames in {summary_path}")
        return None
    records = []
    for site in ordered_sites(summary["site"]):
        path = results_dir / "md" / material / site / "md_frame_analysis.csv"
        if not path.exists():
            skip(f"fig06 {site}: {path} not found")
            continue
        frames = pd.read_csv(path, dtype={"seed": str})
        frames = frames[frames["phase"] == "production"]
        listed = [str(s) for s in summary.loc[summary["site"] == site, "seed"]]
        available = [s for s in listed if (frames["seed"].astype(str) == s).any()]
        if not available:
            skip(f"fig06 {site}: no production frames in {path}")
            continue
        chosen = seed if seed in available else available[0]
        if chosen != seed:
            print(f"[paper-figures] fig06 {site}: seed {seed} not available, using seed {chosen}")
        traj = frames[frames["seed"].astype(str) == chosen].sort_values("time_fs")
        start = float(traj["time_fs"].iloc[0])
        for _, frame in traj.iterrows():
            records.append({"site": site, "seed": chosen, "time_fs": float(frame["time_fs"]),
                            "time_ps": (float(frame["time_fs"]) - start) / 1000.0,
                            "co2_surface_distance_ang": float(frame["c_o_min_slab_distance_ang"])})
    if not records:
        skip("fig06: no frame data for any site")
        return None
    return pd.DataFrame(records)


def plot_fig06(data: pd.DataFrame, seed: str):
    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    for i, site in enumerate(dict.fromkeys(data["site"])):
        part = data[data["site"] == site]
        used = str(part["seed"].iloc[0])
        label = site_label(site) + ("" if used == seed else f" (seed {used})")
        ax.plot(part["time_ps"], part["co2_surface_distance_ang"], color=site_color(site, i),
                linewidth=1.1, label=label)
    ax.set_xlabel("Production time (ps)")
    ax.set_ylabel("CO$_2$–surface distance (Å)")
    ax.set_title(TITLES["fig06"].format(seed=seed))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.5, columnspacing=1.0)
    return fig


# ---------------------------------------------------------------- fig07
def neb_pair_images(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Images of each NEB pair, sorted by image index (pairs in file order)."""
    return {pair: table[table["pair"] == pair].sort_values("image_index")
            for pair in dict.fromkeys(table["pair"])}


def chain_neb_pairs(pairs: dict[str, pd.DataFrame], order=NEB_PATH_ORDER):
    """Links (pair, reversed) along consecutive sites of `order`, or (None, missing links).

    A pair stored in the other direction is used with its image order reversed; a pair
    stored in the path direction is preferred.
    """
    links, missing = [], []
    for start, stop in zip(order[:-1], order[1:]):
        found = None
        for pair, part in pairs.items():
            ends = (str(part["initial_site"].iloc[0]), str(part["final_site"].iloc[0]))
            if ends == (start, stop):
                found = (pair, False)
                break
            if ends == (stop, start) and found is None:
                found = (pair, True)
        if found is None:
            missing.append(f"{start} -> {stop}")
        else:
            links.append(found)
    return (links, []) if not missing else (None, missing)


def segment_label(initial: str, final: str) -> str:
    return f"{str(initial).replace('_', ' ')} → {str(final).replace('_', ' ')}"


def neb_chain_records(pairs: dict[str, pd.DataFrame], links) -> list[dict]:
    """One path: x = cumulative image index, energies relative to the first image of the chain."""
    records, offset_index, offset_energy = [], 0, 0.0
    for pair, reverse in links:
        part = pairs[pair]
        rel = part["rel_energy_eV"].to_numpy(dtype=float)
        initial, final = str(part["initial_site"].iloc[0]), str(part["final_site"].iloc[0])
        if reverse:
            rel = rel[::-1] - rel[-1]
            initial, final = final, initial
        n = len(rel)
        for k, value in enumerate(rel):
            records.append({
                "segment": segment_label(initial, final), "pair": pair, "reversed": bool(reverse),
                "image_index": k, "global_index": offset_index + k,
                "endpoint_site": initial if k == 0 else final if k == n - 1 else "",
                "rel_energy_ev": offset_energy + float(value), "mode": "chain",
            })
        offset_index += n - 1
        offset_energy += float(rel[-1])
    return records


def neb_separate_records(pairs: dict[str, pd.DataFrame]) -> list[dict]:
    """Fallback: every pair on the same axes, relative to its own initial image."""
    records = []
    for pair, part in pairs.items():
        initial, final = str(part["initial_site"].iloc[0]), str(part["final_site"].iloc[0])
        n = len(part)
        for k, value in enumerate(part["rel_energy_eV"].to_numpy(dtype=float)):
            records.append({
                "segment": segment_label(initial, final), "pair": pair, "reversed": False,
                "image_index": k, "global_index": k,
                "endpoint_site": initial if k == 0 else final if k == n - 1 else "",
                "rel_energy_ev": float(value), "mode": "separate",
            })
    return records


def fig07_data(analysis_dir: Path) -> pd.DataFrame | None:
    path = analysis_dir / "science" / "neb" / "neb_image_energies.csv"
    if not path.exists():
        skip(f"fig07: {path} not found")
        return None
    table = pd.read_csv(path)
    if table.empty:
        skip(f"fig07: {path} has no images")
        return None
    return fig07_frame(table)


def fig07_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Plotted data of fig07 from a neb_image_energies table (pair, initial_site,
    final_site, image_index, rel_energy_eV)."""
    pairs = neb_pair_images(table)
    links, missing = chain_neb_pairs(pairs)
    if links is None:
        skip(f"fig07 chain: no NEB pair for {', '.join(missing)}; each pair is drawn relative to its "
             "own initial image")
        return pd.DataFrame(neb_separate_records(pairs))
    unused = [pair for pair in pairs if pair not in {pair for pair, _ in links}]
    if unused:
        skip(f"fig07: pairs not on the path {' -> '.join(NEB_PATH_ORDER)}: {', '.join(unused)}")
    return pd.DataFrame(neb_chain_records(pairs, links))


def plot_fig07(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
    chained = (data["mode"] == "chain").all()
    for i, segment in enumerate(dict.fromkeys(data["segment"])):
        part = data[data["segment"] == segment].sort_values("global_index")
        ax.plot(part["global_index"], part["rel_energy_ev"], marker="o",
                color=NEB_SEGMENT_COLORS[i % len(NEB_SEGMENT_COLORS)], label=segment)
    ax.axhline(0.0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_xticks(sorted(set(int(x) for x in data["global_index"])))
    ends = data[data["endpoint_site"].fillna("") != ""][["global_index", "endpoint_site"]].drop_duplicates()
    if chained:
        for _, row in ends.iterrows():
            ax.axvline(row["global_index"], color="#9e9e9e", linewidth=0.6, zorder=0)
            ax.text(row["global_index"], 1.01, str(row["endpoint_site"]).replace("_", " "),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5)
        ax.set_ylabel("Energy relative to top metal (eV)" if data["endpoint_site"].iloc[0] == "top_metal"
                      else f"Energy relative to {str(data['endpoint_site'].iloc[0]).replace('_', ' ')} (eV)")
    else:
        ax.set_ylabel("Energy relative to the initial image (eV)")
    ax.set_xlabel("Migration path along the Zn–O dimer (NEB images)")
    ax.set_title(TITLES["fig07"], pad=14 if chained else 6)
    ax.legend(loc="best")
    return fig


# ---------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manuscript figures fig03-fig07 of one run.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Simulation run directory (RUN).")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Analysis run directory (A).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <analysis-dir>/paper.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--tddft-sigma-ev", type=float, default=0.10,
                        help="Gaussian broadening sigma (eV) of fig04.")
    parser.add_argument("--desorption-threshold", type=float, default=DEFAULT_DESORPTION_THRESHOLD_ANG,
                        help="Desorption threshold of the MD analysis (A); not drawn, kept for the command line "
                             "shared with md_trajectory_analysis.")
    parser.add_argument("--md-seed", default=DEFAULT_MD_SEED,
                        help="MD seed drawn in fig06 (a site without it uses its first available seed).")
    parser.add_argument("--co2rr-variant", default="c", choices=("a", "b", "c"),
                        help="Variant of co2rr_corrected_summary.csv drawn in fig05.")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("[paper-figures] --dpi must be positive")
    if args.tddft_sigma_ev <= 0:
        raise SystemExit("[paper-figures] --tddft-sigma-ev must be positive")
    results_dir = args.results_dir.expanduser().resolve()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"[paper-figures] results directory not found: {results_dir}")
    output_dir = (args.output_dir or analysis_dir / "paper").expanduser().resolve()
    seed = str(args.md_seed)
    apply_style()

    producers = {
        "fig03": (lambda: fig03_data(results_dir, args.material), lambda d: plot_fig03(d, args.material)),
        "fig04": (lambda: fig04_data(results_dir, args.material, args.tddft_sigma_ev), plot_fig04),
        "fig05": (lambda: fig05_data(analysis_dir, args.material, args.co2rr_variant), plot_fig05),
        "fig06": (lambda: fig06_data(results_dir, analysis_dir, args.material, seed),
                  lambda d: plot_fig06(d, seed)),
        "fig07": (lambda: fig07_data(analysis_dir), plot_fig07),
    }
    written = 0
    for figure in FIGURES:
        if figure not in args.figures:
            continue
        build, plot = producers[figure]
        data = build()
        if data is None:
            continue
        save(plot(data), data, output_dir, FIGURE_NAMES[figure], args.dpi)
        written += 1
    print(f"[paper-figures] {written} of {len(args.figures)} requested figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
