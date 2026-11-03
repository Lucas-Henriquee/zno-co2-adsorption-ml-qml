"""Publication figures from the tables of analysis_co2_reduction.py.

fig01 adsorption-energy heatmap (rows with quality_flag/dft_status ok only), fig02 CO2
activation (C-O stretch against the run's gas-phase CO2 baseline, O-C-O bend), fig03
E_ads vs first bright TDDFT excitation of the isolated CO2, fig04 sites ranked by the
CHE limiting potential, fig05 broadened TDDFT spectra of the isolated CO2 (adsorbed
geometry) for every ok site. No composite score is used.
"""
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


def row_is_ok(row: dict[str, str]) -> bool:
    """quality_flag and dft_status are ok (a column that is absent does not count against the row)."""
    return all(str(row.get(column, "ok") or "").strip() == "ok" for column in ("quality_flag", "dft_status"))


def figure_adsorption_heatmap(rows: list[dict[str, str]], plt, Normalize, ticker) -> None:
    rows = [row for row in rows if row_is_ok(row)]
    if not rows:
        raise SystemExit("No rows with quality_flag/dft_status ok for the heatmap.")
    materials = ordered_unique([row["material"] for row in rows], MATERIAL_ORDER)
    sites = ordered_unique([row["site"] for row in rows], SITE_ORDER)

    matrix = np.full((len(materials), len(sites)), np.nan, dtype=float)
    for row in rows:
        m_idx = materials.index(row["material"])
        s_idx = sites.index(row["site"])
        matrix[m_idx, s_idx] = safe_float(row.get("adsorption_energy_ev"))
    any_positive = bool(np.any(matrix[np.isfinite(matrix)] > 0.0))

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
            label = "--" if not math.isfinite(value) else f"{value:.2f}" + ("*" if value > 0.0 else "")
            color = "white" if math.isfinite(value) and norm(value) > 0.55 else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.set_label("Adsorption energy, E_ads (eV)")
    if any_positive:
        ax.text(
            0.0, -0.18,
            "* E_ads > 0: not bound at PBE level (|E_ads| < 0.05 eV is within the numerical uncertainty)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
        )
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

    # The stretch is taken from the table: it is relative to the relaxed gas-phase CO2 of
    # the same run (co_bond_baseline_ang), not to an experimental bond length.
    baselines: set[float] = set()
    for row in filtered:
        s_idx = sites.index(row["site"])
        m_idx = materials.index(row["material"])
        co_stretch = safe_float(row.get("co_bond_stretch_ang"))
        baseline = safe_float(row.get("co_bond_baseline_ang"))
        angle = safe_float(row.get("oco_angle_deg"))
        if math.isfinite(co_stretch):
            stretch[s_idx, m_idx] = co_stretch
        if math.isfinite(baseline):
            baselines.add(round(baseline, 4))
        if math.isfinite(angle):
            bend[s_idx, m_idx] = 180.0 - angle

    s_finite = stretch[np.isfinite(stretch)]
    b_finite = bend[np.isfinite(bend)]
    if len(b_finite) == 0:
        raise SystemExit("No finite activation values available.")
    if len(s_finite) == 0:
        print("[figures] fig02: no C-O stretch (no gas-phase CO2 baseline in the table); stretch panel left empty")
        s_finite = np.asarray([0.0])

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
    if len(baselines) == 1:
        axes[0].set_title(f"C-O stretch vs gas CO2 ({next(iter(baselines)):.3f} Ang), Δd (Ang)", fontsize=10)
    else:
        axes[0].set_title("C-O stretch vs gas CO2 of the same profile, Δd (Ang)", fontsize=10)
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
    """First bright TDDFT excitation of the isolated CO2 (co2_isolated_first_bright_excitation_ev).

    The older name co2_isolated_onset_ev is read as a fallback (the onset column of the
    first campaign is not: it held Hartree values). This is not an absorption onset of the ZnO/CO2
    system.
    """
    for column in ("co2_isolated_first_bright_excitation_ev", "co2_isolated_onset_ev"):
        value = row.get(column)
        if value not in (None, ""):
            return safe_float(value)
    return float("nan")


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
    ax.set_ylabel("First bright excitation, isolated CO2 (eV)")
    ax.set_title("Adsorption Energy vs First Bright TDDFT Excitation of Isolated CO2")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.32)
    ax.text(
        0.014,
        0.97,
        "Isolated CO2 in its adsorbed geometry, not the ZnO/CO2 system",
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
    """Sites ranked by the CHE limiting potential U_L (rank_by_limiting_potential).

    Sites without a valid pathway (pathway_validity unbound/wrong_intermediate) or
    without U_L have no rank and are not drawn.
    """
    ranked = sorted(
        [
            row
            for row in rows
            if row_is_ok(row)
            and str(row.get("rank_by_limiting_potential", "")).strip()
            and math.isfinite(safe_float(row.get("limiting_potential_v")))
        ],
        key=lambda row: safe_float(row.get("rank_by_limiting_potential")),
    )
    if not ranked:
        print("[figures] skipped fig04_site_ranking (no site with a limiting potential; run the CO2RR pathway step)")
        return

    labels = [f"{row['material']} | {_site_label(row['site'])}" for row in ranked]
    values = [safe_float(row.get("limiting_potential_v")) for row in ranked]
    colors = [MATERIAL_COLORS.get(row["material"], "#333333") for row in ranked]

    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.6)

    for idx, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + (0.01 if width >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{values[idx]:+.2f}",
            ha="left" if width >= 0 else "right",
            va="center",
            fontsize=8.5,
        )

    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_ylabel("Oxide-site candidate")
    ax.set_xlabel("Limiting potential, $U_L$ (V vs RHE)")
    ax.set_title("Sites Ranked by CO2 -> CO Limiting Potential (CHE)")
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


SITE_LINESTYLES = {"top_metal": "-", "top_oxygen": "--", "bridge": ":"}


def figure_tddft_gaussian_spectra(rows: list[dict[str, str]], plt) -> None:
    """Broadened LrTDDFT spectra of the isolated CO2 in its adsorbed geometry, every ok site.

    Intensity = sum_i f_i g(E - E_i) with a normalised Gaussian g of width sigma, in eV^-1
    (f is dimensionless); not normalised between sites.
    """
    valid_rows = [row for row in rows if row_is_ok(row)]
    if not valid_rows:
        print("[figures] skipped fig05_tddft_gaussian_spectra (no valid quality rows)")
        return

    material_order = {name: idx for idx, name in enumerate(MATERIAL_ORDER)}
    site_order = {name: idx for idx, name in enumerate(SITE_ORDER)}
    valid_rows.sort(key=lambda row: (material_order.get(row["material"], 99), site_order.get(row["site"], 99)))

    sigma_ev = 0.10
    spectra_payload: list[dict[str, object]] = []
    all_min: list[float] = []
    all_max: list[float] = []

    for row in valid_rows:
        material, site = row["material"], row["site"]
        energies, osc = _load_transitions(material=material, site=site)
        if len(energies) == 0:
            continue
        all_min.append(float(np.min(energies)))
        all_max.append(float(np.max(energies)))
        spectra_payload.append({"material": material, "site": site, "energies": energies, "osc": osc})

    if not spectra_payload:
        print("[figures] skipped fig05_tddft_gaussian_spectra (missing transitions.csv files)")
        return

    e_min = max(0.0, min(all_min) - 4.0 * sigma_ev)
    e_max = max(all_max) + 4.0 * sigma_ev
    if e_max <= e_min:
        print("[figures] skipped fig05_tddft_gaussian_spectra (invalid energy window)")
        return

    grid = np.linspace(e_min, e_max, 1400)
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    y_max = 0.0
    for payload in spectra_payload:
        material = str(payload["material"])
        site = str(payload["site"])
        color = MATERIAL_COLORS.get(material, "#555555")
        intensity = _gaussian_broaden(
            energy_grid=grid,
            transition_energies=payload["energies"],
            oscillator_strengths=payload["osc"],
            sigma_ev=sigma_ev,
        )
        y_max = max(y_max, float(np.max(intensity)))
        ax.plot(
            grid,
            intensity,
            color=color,
            linestyle=SITE_LINESTYLES.get(site, "-"),
            linewidth=1.8,
            alpha=0.95,
            label=f"{material} ({_site_label(site)})",
        )

    ax.set_xlim(e_min, e_max)
    ax.set_ylim(0.0, 1.06 * y_max if y_max > 0.0 else 1.0)
    ax.set_xlabel("Excitation energy (eV)")
    ax.set_ylabel("Broadened oscillator strength (eV$^{-1}$)")
    ax.set_title("LrTDDFT Spectra of Isolated CO2 (Adsorbed Geometry)")
    ax.grid(axis="both", linestyle="--", linewidth=0.45, alpha=0.30)
    ax.text(
        0.015,
        0.965,
        f"Gaussian broadening σ = {sigma_ev:.2f} eV; oscillator strength f dimensionless",
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
