from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from scripts.common.project_config import RESULTS_DIR, TDDFT_RESULTS_DIR


ANALYSIS_DIR = RESULTS_DIR / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"
FIGURE_DPI = 600
SITE_ORDER = ("top_metal", "top_oxygen", "bridge")
MATERIAL_ORDER = ("ZnO", "TiO2", "CeO2")
SITE_DISPLAY = {
    "top_metal": "Top metal",
    "top_oxygen": "Top oxygen",
    "bridge": "Bridge",
}
SITE_SHORT = {
    "top_metal": "TM",
    "top_oxygen": "TO",
    "bridge": "BR",
}
MATERIAL_COLORS = {
    "ZnO": "#2E6FBB",
    "TiO2": "#E07A2D",
    "CeO2": "#2A9D6F",
}
MATERIAL_MARKERS = {
    "ZnO": "o",
    "TiO2": "s",
    "CeO2": "^",
}
SITE_MARKERS = {
    "top_metal": "o",
    "top_oxygen": "s",
    "bridge": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render publication figures from CO2-reduction analysis tables."
    )
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--tddft-dir", type=Path, default=TDDFT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI)
    return parser.parse_args()


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib import ticker
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required for figure rendering. "
            "Install with: pip install matplotlib"
        ) from exc

    return plt, patches, ticker, Normalize, Line2D


def safe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ordered_unique(values: list[str], preferred_order: tuple[str, ...]) -> list[str]:
    present = set(values)
    ordered = [value for value in preferred_order if value in present]
    extras = sorted(v for v in present if v not in preferred_order)
    return ordered + extras


def save_figure(fig, base_name: str, plt) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{base_name}.png"
    pdf_path = FIGURES_DIR / f"{base_name}.pdf"
    svg_path = FIGURES_DIR / f"{base_name}.svg"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {png_path}")
    print(f"[figures] wrote {pdf_path}")
    print(f"[figures] wrote {svg_path}")


def apply_publication_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "savefig.transparent": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _site_label(site: str) -> str:
    return SITE_DISPLAY.get(site, site.replace("_", " ").title())


def _within_axis(target: float, lo: float, hi: float, margin: float = 0.10) -> bool:
    span = hi - lo
    if span <= 0:
        return False
    return (lo - margin * span) <= target <= (hi + margin * span)


def figure_adsorption_heatmap(rows: list[dict[str, str]], plt, Normalize, ticker) -> None:
    materials = ordered_unique([row["material"] for row in rows], MATERIAL_ORDER)
    sites = ordered_unique([row["site"] for row in rows], SITE_ORDER)

    matrix = np.full((len(materials), len(sites)), np.nan, dtype=float)
    for row in rows:
        m_idx = materials.index(row["material"])
        s_idx = sites.index(row["site"])
        matrix[m_idx, s_idx] = safe_float(row.get("adsorption_energy_ev"))

    finite = matrix[np.isfinite(matrix)]
    if len(finite) == 0:
        raise SystemExit("No finite adsorption energies for heatmap.")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGnBu", norm=norm, aspect="auto")

    ax.set_xticks(range(len(sites)))
    ax.set_yticks(range(len(materials)))
    ax.set_xticklabels([_site_label(site) for site in sites])
    ax.set_yticklabels(materials)
    ax.set_xlabel("Adsorption site")
    ax.set_ylabel("Metal oxide")
    ax.set_title("CO2 Adsorption Energy Landscape")

    for i in range(len(materials)):
        for j in range(len(sites)):
            value = matrix[i, j]
            label = "--" if not math.isfinite(value) else f"{value:.2f}"
            color = "white" if math.isfinite(value) and norm(value) > 0.55 else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.set_label("Adsorption energy, E_ads (eV)")
    _clean_axes(ax)

    save_figure(fig, "fig01_adsorption_heatmap", plt)


def figure_activation_scatter(rows: list[dict[str, str]], plt, Normalize, ticker) -> None:
    filtered = [row for row in rows if row.get("quality_flag", "") == "ok"]
    if not filtered:
        raise SystemExit("No valid rows for activation figure.")

    materials = ordered_unique([row["material"] for row in filtered], MATERIAL_ORDER)
    sites = ordered_unique([row["site"] for row in filtered], SITE_ORDER)

    stretch = np.full((len(sites), len(materials)), np.nan, dtype=float)
    bend = np.full((len(sites), len(materials)), np.nan, dtype=float)

    for row in filtered:
        s_idx = sites.index(row["site"])
        m_idx = materials.index(row["material"])
        co_avg = safe_float(row.get("co_bond_avg_ang"))
        angle = safe_float(row.get("oco_angle_deg"))
        if math.isfinite(co_avg):
            stretch[s_idx, m_idx] = co_avg - 1.16
        if math.isfinite(angle):
            bend[s_idx, m_idx] = 180.0 - angle

    s_finite = stretch[np.isfinite(stretch)]
    b_finite = bend[np.isfinite(bend)]
    if len(s_finite) == 0 or len(b_finite) == 0:
        raise SystemExit("No finite activation values available.")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8), constrained_layout=True, sharey=True)

    stretch_norm = Normalize(vmin=float(np.min(s_finite)), vmax=float(np.max(s_finite)))
    bend_norm = Normalize(vmin=float(np.min(b_finite)), vmax=float(np.max(b_finite)))

    im0 = axes[0].imshow(stretch, cmap="YlOrBr", norm=stretch_norm, aspect="auto")
    im1 = axes[1].imshow(bend, cmap="PuBuGn", norm=bend_norm, aspect="auto")

    for ax in axes:
        ax.set_xticks(range(len(materials)))
        ax.set_xticklabels(materials)
        ax.set_yticks(range(len(sites)))
        ax.set_yticklabels([_site_label(site) for site in sites])
        ax.set_xlabel("Metal oxide")
        _clean_axes(ax)

    axes[0].set_ylabel("Adsorption site")
    axes[0].set_title("CO bond stretch, Δd (Ang)")
    axes[1].set_title("O-C-O bend, Δθ (deg)")

    for i in range(len(sites)):
        for j in range(len(materials)):
            s_val = stretch[i, j]
            b_val = bend[i, j]
            axes[0].text(j, i, "--" if not math.isfinite(s_val) else f"{s_val:.4f}", ha="center", va="center", fontsize=8.5)
            axes[1].text(j, i, "--" if not math.isfinite(b_val) else f"{b_val:.2f}", ha="center", va="center", fontsize=8.5)

    cbar0 = fig.colorbar(im0, ax=axes[0], pad=0.02)
    cbar0.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    cbar0.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    cbar0.set_label("Δd (Ang)")

    cbar1 = fig.colorbar(im1, ax=axes[1], pad=0.02)
    cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar1.set_label("Δθ (deg)")

    fig.suptitle("CO2 Activation Components", y=1.02, fontsize=13)
    save_figure(fig, "fig02_activation_scatter", plt)


def co2_onset(row: dict[str, str]) -> float:
    """TDDFT onset of the isolated CO2 molecule (co2_isolated_onset_ev).

    The legacy name tddft_onset_ev is read only as a fallback. This is not an
    absorption onset of the ZnO/CO2 system.
    """
    value = row.get("co2_isolated_onset_ev")
    if value in (None, ""):
        value = row.get("tddft_onset_ev")
    return safe_float(value)


def figure_photo_thermo_map(rows: list[dict[str, str]], plt, Line2D) -> None:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("quality_flag", "") != "ok":
            continue
        ads = safe_float(row.get("adsorption_energy_ev"))
        onset = co2_onset(row)
        if math.isfinite(ads) and math.isfinite(onset):
            filtered.append(row)

    if not filtered:
        raise SystemExit("No valid rows for photo-thermo map.")

    materials = ordered_unique([row["material"] for row in filtered], MATERIAL_ORDER)
    sites = ordered_unique([row["site"] for row in filtered], SITE_ORDER)

    ads_values = [safe_float(row["adsorption_energy_ev"]) for row in filtered]
    onset_values = [co2_onset(row) for row in filtered]
    x_min, x_max = min(ads_values), max(ads_values)
    y_min, y_max = min(onset_values), max(onset_values)
    x_pad = max(0.04, 0.10 * (x_max - x_min if x_max > x_min else 1.0))
    y_pad = max(0.004, 0.14 * (y_max - y_min if y_max > y_min else 1.0))

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)

    for material in materials:
        material_rows = [
            row for row in filtered if row["material"] == material and row["site"] in SITE_ORDER
        ]
        material_rows = sorted(material_rows, key=lambda row: SITE_ORDER.index(row["site"]))
        if len(material_rows) < 2:
            continue
        ax.plot(
            [safe_float(row["adsorption_energy_ev"]) for row in material_rows],
            [co2_onset(row) for row in material_rows],
            color=MATERIAL_COLORS.get(material, "#666666"),
            linewidth=1.0,
            linestyle=":",
            alpha=0.65,
            zorder=1,
        )

    for row in filtered:
        material = row["material"]
        site = row["site"]
        x_val = safe_float(row["adsorption_energy_ev"])
        y_val = co2_onset(row)
        color = MATERIAL_COLORS.get(material, "#666666")
        marker = SITE_MARKERS.get(site, "o")

        ax.scatter(
            x_val,
            y_val,
            s=95,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.75,
            alpha=0.96,
            zorder=3,
        )

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("Adsorption energy, E_ads (eV)")
    ax.set_ylabel("Isolated-CO2 TDDFT onset (eV)")
    ax.set_title("Adsorption Energy vs Isolated-CO2 TDDFT Onset")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.32)
    ax.text(
        0.014,
        0.97,
        "Onset of the isolated CO2 molecule, not of the ZnO/CO2 system",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#3D556E",
    )
    _clean_axes(ax)

    material_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=MATERIAL_COLORS.get(material, "#666666"),
            markeredgecolor="black",
            markersize=7.8,
            label=material,
        )
        for material in materials
    ]
    site_handles = [
        Line2D(
            [0],
            [0],
            marker=SITE_MARKERS.get(site, "o"),
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#2F2F2F",
            markersize=7.8,
            label=_site_label(site),
        )
        for site in sites
    ]

    material_legend = ax.legend(
        handles=material_handles,
        title="Oxide",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    ax.add_artist(material_legend)
    ax.legend(
        handles=site_handles,
        title="Site marker",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.28),
        frameon=False,
    )

    save_figure(fig, "fig03_photo_thermo_map", plt)


def figure_site_ranking(rows: list[dict[str, str]], plt, patches) -> None:
    ranked = sorted(
        [
            row
            for row in rows
            if row.get("quality_flag", "") == "ok"
            and math.isfinite(safe_float(row.get("reduction_proxy_score")))
        ],
        key=lambda row: safe_float(row.get("reduction_proxy_score")),
        reverse=True,
    )
    if not ranked:
        raise SystemExit("No valid rows for ranking figure.")

    labels = [f"{row['material']} | {_site_label(row['site'])}" for row in ranked]
    scores = [safe_float(row.get("reduction_proxy_score")) for row in ranked]
    colors = [MATERIAL_COLORS.get(row["material"], "#333333") for row in ranked]

    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    y = np.arange(len(labels))
    bars = ax.barh(y, scores, color=colors, edgecolor="black", linewidth=0.6)

    for idx, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.0025,
            bar.get_y() + bar.get_height() / 2,
            f"{scores[idx]:.3f}",
            ha="left",
            va="center",
            fontsize=8.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_ylabel("Oxide-site candidate")
    ax.set_xlabel("Reduction proxy score (a.u.)")
    ax.set_title("Ranked Oxide-Site Candidates")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    _clean_axes(ax)

    legend_handles = [
        patches.Patch(facecolor=MATERIAL_COLORS[mat], edgecolor="black", label=mat)
        for mat in MATERIAL_ORDER
        if any(row["material"] == mat for row in ranked)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=max(1, len(legend_handles)),
        frameon=False,
        title="Oxide",
    )

    save_figure(fig, "fig04_site_ranking", plt)


def _load_transitions(material: str, site: str) -> tuple[np.ndarray, np.ndarray]:
    path = TDDFT_RESULTS_DIR / material / site / "transitions.csv"
    if not path.exists():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    energies: list[float] = []
    oscillator_strengths: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            energy = safe_float(row.get("energy_ev"))
            osc = safe_float(row.get("oscillator_strength"))
            if math.isfinite(energy) and math.isfinite(osc) and osc >= 0.0:
                energies.append(energy)
                oscillator_strengths.append(osc)

    if not energies:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    energies_arr = np.asarray(energies, dtype=float)
    osc_arr = np.asarray(oscillator_strengths, dtype=float)
    order = np.argsort(energies_arr)
    return energies_arr[order], osc_arr[order]


def _gaussian_broaden(
    energy_grid: np.ndarray,
    transition_energies: np.ndarray,
    oscillator_strengths: np.ndarray,
    sigma_ev: float,
) -> np.ndarray:
    if len(transition_energies) == 0 or len(oscillator_strengths) == 0:
        return np.zeros_like(energy_grid)
    if sigma_ev <= 0.0:
        raise ValueError("sigma_ev must be positive.")

    delta = (energy_grid[:, None] - transition_energies[None, :]) / sigma_ev
    kernel = np.exp(-0.5 * delta**2) / (sigma_ev * math.sqrt(2.0 * math.pi))
    return kernel @ oscillator_strengths


def figure_tddft_gaussian_spectra(rows: list[dict[str, str]], plt) -> None:
    valid_rows = [
        row
        for row in rows
        if row.get("quality_flag", "") == "ok"
        and math.isfinite(safe_float(row.get("reduction_proxy_score")))
    ]
    if not valid_rows:
        print("[figures] skipped fig05_tddft_gaussian_spectra (no valid quality rows)")
        return

    best_by_material: dict[str, dict[str, str]] = {}
    for row in valid_rows:
        material = row["material"]
        score = safe_float(row.get("reduction_proxy_score"))
        current = best_by_material.get(material)
        if current is None or score > safe_float(current.get("reduction_proxy_score")):
            best_by_material[material] = row

    selected_materials = ordered_unique(list(best_by_material.keys()), MATERIAL_ORDER)
    if not selected_materials:
        print("[figures] skipped fig05_tddft_gaussian_spectra (no selected materials)")
        return

    sigma_ev = 0.10
    spectra_payload: list[dict[str, object]] = []
    all_min: list[float] = []
    all_max: list[float] = []

    for material in selected_materials:
        row = best_by_material[material]
        site = row["site"]
        energies, osc = _load_transitions(material=material, site=site)
        if len(energies) == 0:
            continue
        all_min.append(float(np.min(energies)))
        all_max.append(float(np.max(energies)))
        spectra_payload.append(
            {
                "material": material,
                "site": site,
                "energies": energies,
                "osc": osc,
            }
        )

    if not spectra_payload:
        print("[figures] skipped fig05_tddft_gaussian_spectra (missing transitions.csv files)")
        return

    e_min = max(0.0, min(all_min) - 4.0 * sigma_ev)
    e_max = max(all_max) + 4.0 * sigma_ev
    if e_max <= e_min:
        print("[figures] skipped fig05_tddft_gaussian_spectra (invalid energy window)")
        return

    grid = np.linspace(e_min, e_max, 1400)
    global_max = 0.0
    for payload in spectra_payload:
        intensity = _gaussian_broaden(
            energy_grid=grid,
            transition_energies=payload["energies"],
            oscillator_strengths=payload["osc"],
            sigma_ev=sigma_ev,
        )
        payload["intensity"] = intensity
        global_max = max(global_max, float(np.max(intensity)))

    norm_scale = global_max if global_max > 0.0 else 1.0

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for payload in spectra_payload:
        material = str(payload["material"])
        site = str(payload["site"])
        color = MATERIAL_COLORS.get(material, "#555555")
        y = payload["intensity"] / norm_scale

        ax.plot(
            grid,
            y,
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=f"{material} ({SITE_SHORT.get(site, site)})",
        )
        ax.fill_between(grid, 0.0, y, color=color, alpha=0.10)

    ax.set_xlim(e_min, e_max)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("Excitation energy (eV)")
    ax.set_ylabel("Normalized intensity (a.u.)")
    ax.set_title("Gaussian-Broadened TDDFT Spectra (Best Site per Oxide)")
    ax.grid(axis="both", linestyle="--", linewidth=0.45, alpha=0.30)
    ax.text(
        0.015,
        0.965,
        f"Gaussian broadening σ = {sigma_ev:.2f} eV",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color="#3E4B5A",
    )
    _clean_axes(ax)

    ax.legend(loc="upper right", frameon=False, title="Material (site)")
    save_figure(fig, "fig05_tddft_gaussian_spectra", plt)


def main() -> None:
    global ANALYSIS_DIR, FIGURES_DIR, TDDFT_RESULTS_DIR, FIGURE_DPI
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive.")
    ANALYSIS_DIR = args.analysis_dir.expanduser().resolve()
    FIGURES_DIR = args.output_dir.expanduser().resolve()
    TDDFT_RESULTS_DIR = args.tddft_dir.expanduser().resolve()
    FIGURE_DPI = args.dpi
    plt, patches, ticker, Normalize, Line2D = _require_matplotlib()
    apply_publication_style(plt)

    co2_table = ANALYSIS_DIR / "co2_reduction_table.csv"
    rows = read_csv_rows(co2_table)

    figure_adsorption_heatmap(rows, plt=plt, Normalize=Normalize, ticker=ticker)
    figure_activation_scatter(rows, plt=plt, Normalize=Normalize, ticker=ticker)
    figure_photo_thermo_map(rows, plt=plt, Line2D=Line2D)
    figure_site_ranking(rows, plt=plt, patches=patches)
    figure_tddft_gaussian_spectra(rows, plt=plt)


if __name__ == "__main__":
    main()
