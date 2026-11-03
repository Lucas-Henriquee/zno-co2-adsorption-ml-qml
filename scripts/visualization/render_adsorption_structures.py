"""Render the three relaxed ZnO–CO2 structures with their starting sites marked.

Only the exposed slab layer and CO2 are drawn. Saved DFT coordinates and
reported descriptors are checked but never modified.

Inputs of one simulation run (RUN = results/simulations/<id>):
  --input-dir RUN/dft/ZnO      (<site>/ZnO_<site>_optimized.xyz)
  --summary   RUN/dft_summary.csv
and the shared starting geometry (not copied into a run):
  --site-metadata data/geometries/ZnO/metadata.json, --initial-slab data/geometries/ZnO/clean_slab.traj
Every input must exist; a missing one is an error (no fallback).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.geometry import find_mic
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from scripts.common.project_config import DATA_DIR, DFT_RESULTS_DIR, RESULTS_DIR


SITES = ("top_metal", "top_oxygen", "bridge")
SITE_NAMES = {"top_metal": "Top metal", "top_oxygen": "Top oxygen", "bridge": "Bridge"}
COLORS = {"Zn": "#637C99", "O": "#C34F52", "C": "#333B45"}
RADII = {"Zn": 0.25, "O": 0.22, "C": 0.20}


@dataclass
class SiteData:
    atoms: object
    metrics: dict[str, float]
    site_atoms: tuple[int, ...]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, default=DFT_RESULTS_DIR / "ZnO")
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "dft_summary.csv")
    parser.add_argument(
        "--site-metadata", type=Path, default=DATA_DIR / "geometries/ZnO/metadata.json"
    )
    parser.add_argument(
        "--initial-slab", type=Path, default=DATA_DIR / "geometries/ZnO/clean_slab.traj"
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "analysis/figures")
    parser.add_argument(
        "--view", choices=("side", "top", "both", "combined"), default="combined",
        help="Render one view, two separate PNGs, or one combined PNG.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--allow-geometry-warnings", action="store_true",
        help="Create a diagnostic draft despite warnings; never labels it publication-ready.",
    )
    return parser.parse_args(argv)


def check_inputs(args) -> None:
    """Fail with one message listing every missing input file."""
    required = [
        ("--summary", args.summary),
        ("--site-metadata", args.site_metadata),
        ("--initial-slab", args.initial_slab),
    ]
    required += [
        ("--input-dir", args.input_dir / site / f"ZnO_{site}_optimized.xyz") for site in SITES
    ]
    missing = [f"{option}: {path}" for option, path in required if not Path(path).is_file()]
    if missing:
        raise SystemExit("Missing input file(s):\n  " + "\n  ".join(missing))


def geometry_metrics(atoms) -> dict[str, float]:
    slab_count = len(atoms) - 3
    oxygen_a, carbon, oxygen_b = slab_count, slab_count + 1, slab_count + 2
    return {
        "co_bond_1_ang": float(atoms.get_distance(carbon, oxygen_a)),
        "co_bond_2_ang": float(atoms.get_distance(carbon, oxygen_b)),
        "oco_angle_deg": float(atoms.get_angle(oxygen_a, carbon, oxygen_b)),
        "c_surface_distance_ang": float(
            atoms.positions[carbon, 2] - atoms.positions[:slab_count, 2].max()
        ),
    }


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["material"] == "ZnO"]
    return {row["site"]: row for row in rows}


def carbon_offset_from_site(atoms, site_atoms: tuple[int, ...]) -> float:
    """Measure lateral C offset from the same surface atom(s) after relaxation."""
    displacements = atoms.positions[list(site_atoms)] - atoms.positions[-2]
    minimum_image, _ = find_mic(displacements, atoms.cell, pbc=[True, True, False])
    return float(np.linalg.norm(minimum_image[:, :2].mean(axis=0)))


def validate(site: str, data: SiteData, summary_row: dict[str, str]) -> list[str]:
    atoms = data.atoms
    if len(atoms) < 7 or atoms[-3:].get_chemical_symbols() != ["O", "C", "O"]:
        raise ValueError(f"{site}: expected a ZnO slab followed by O, C, O")
    for key, calculated in data.metrics.items():
        reported = float(summary_row[key])
        if not np.isclose(calculated, reported, atol=0.002, rtol=0):
            raise ValueError(
                f"{site}: {key} differs from the saved DFT summary "
                f"({calculated:.4f} versus {reported:.4f})"
            )

    warnings = []
    slab_count = len(atoms) - 3
    for i in range(slab_count):
        for j in range(i + 1, slab_count):
            distance = float(atoms.get_distance(i, j))
            if distance < 1.0:
                warnings.append(
                    f"{site}: slab atoms {i} ({atoms[i].symbol}) and {j} "
                    f"({atoms[j].symbol}) are only {distance:.3f} Å apart"
                )
    offset = carbon_offset_from_site(atoms, data.site_atoms)
    if offset > 1.5:
        warnings.append(
            f"{site}: carbon is {offset:.3f} Å laterally from the relaxed "
            "surface atom(s) defining the starting site"
        )
    return warnings


def top_layer_indices(atoms) -> list[int]:
    slab_count = len(atoms) - 3
    lowest_z = float(atoms.positions[:slab_count, 2].min())
    return [i for i in range(slab_count) if atoms.positions[i, 2] > lowest_z + 2.0]


def starting_site_atoms(initial_slab, initial_xy: np.ndarray, site: str) -> tuple[int, ...]:
    """Return the surface atom(s) that defined the nominal starting site."""
    top_z = float(initial_slab.positions[:, 2].max())
    top_indices = [
        i for i, atom in enumerate(initial_slab)
        if atom.position[2] >= top_z - 0.1
    ]

    if site == "top_metal":
        symbol = "Zn"
    elif site == "top_oxygen":
        symbol = "O"
    else:
        symbol = None
    if symbol is not None:
        candidates = [i for i in top_indices if initial_slab[i].symbol == symbol]
        if not candidates:
            raise ValueError(f"{site}: no top-layer {symbol} atom")
        return (min(
            candidates,
            key=lambda i: np.linalg.norm(initial_slab.positions[i, :2] - initial_xy),
        ),)

    if site == "bridge":
        pairs = [
            (i, j) for i in top_indices for j in top_indices
            if initial_slab[i].symbol == "Zn" and initial_slab[j].symbol == "O"
        ]
        if not pairs:
            raise ValueError("bridge: no top-layer Zn–O pair")
        return min(
            pairs,
            key=lambda pair: np.linalg.norm(
                0.5 * (initial_slab.positions[pair[0], :2]
                       + initial_slab.positions[pair[1], :2]) - initial_xy
            ),
        )
    raise ValueError(f"Unsupported site: {site}")


def draw_side_view(
    ax, data: SiteData, site: str, letter: str, site_atoms: tuple[int, ...],
    show_title: bool = True,
) -> None:
    """Show the optimized outer layer and CO2 in a consistent y–z projection."""
    atoms = data.atoms
    top_indices = top_layer_indices(atoms)
    carbon_index = len(atoms) - 2
    carbon_y = float(atoms.positions[carbon_index, 1])
    positions = atoms.positions.copy()
    positions[:, 1] -= carbon_y
    top_indices = [
        i for i in top_indices
        if abs(positions[i, 1]) <= 3.15 or i in site_atoms
    ]

    for i in top_indices:
        for j in top_indices:
            if j <= i or {atoms[i].symbol, atoms[j].symbol} != {"Zn", "O"}:
                continue
            distance = float(atoms.get_distance(i, j))
            if 1.4 <= distance <= 2.5:
                ax.plot(
                    positions[[i, j], 1], positions[[i, j], 2],
                    color="#B4BDC7", linewidth=1.2, zorder=1,
                )
    for oxygen_index in (len(atoms) - 3, len(atoms) - 1):
        ax.plot(
            positions[[oxygen_index, carbon_index], 1],
            positions[[oxygen_index, carbon_index], 2],
            color="#6D7580", linewidth=2.0, zorder=2,
        )

    for index in top_indices + list(range(len(atoms) - 3, len(atoms))):
        atom = atoms[index]
        position = positions[index]
        ax.add_patch(
            Circle(
                (position[1], position[2]), RADII[atom.symbol],
                facecolor=COLORS[atom.symbol], edgecolor="white",
                linewidth=0.7, zorder=3,
            )
        )
        if index >= len(atoms) - 3:
            ax.text(
                position[1], position[2], atom.symbol, ha="center", va="center",
                color="white", fontsize=7, fontweight="bold", zorder=4,
            )

    # The outline follows the original surface atom identities through DFT
    # relaxation; it does not assert that CO2 stayed above those atoms.
    for index in site_atoms:
        atom = atoms[index]
        position = positions[index]
        ax.add_patch(
            Circle(
                (position[1], position[2]), RADII[atom.symbol] + 0.085,
                facecolor="none", edgecolor="#1D252D", linewidth=1.15, zorder=5,
            )
        )

    # Keep numerical annotations limited to the two actual C–O bonds.
    carbon_point = positions[carbon_index, [1, 2]]
    for oxygen_index in (len(atoms) - 3, len(atoms) - 1):
        oxygen_point = positions[oxygen_index, [1, 2]]
        direction = oxygen_point - carbon_point
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        label_point = 0.5 * (carbon_point + oxygen_point) + 0.95 * normal
        bond_length = float(atoms.get_distance(carbon_index, oxygen_index))
        ax.text(
            label_point[0], label_point[1], f"{bond_length:.3f} Å",
            ha="center", va="center", fontsize=6.6, color="#252B32", zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.5},
        )

    if show_title:
        ax.set_title(f"({letter}) {SITE_NAMES[site]}", fontsize=9.0, fontweight="semibold", pad=4)
    ax.set_xlim(-3.35, 3.35)
    ax.set_ylim(13.0, 19.35)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_top_view(
    ax, data: SiteData, site: str, letter: str, site_atoms: tuple[int, ...],
    show_title: bool = True,
) -> None:
    """Show the same optimized geometry from above, without extra numbers."""
    atoms = data.atoms
    top_indices = top_layer_indices(atoms)
    carbon_index = len(atoms) - 2
    positions = atoms.positions[:, :2].copy()
    positions -= atoms.positions[carbon_index, :2]
    top_indices = [
        i for i in top_indices
        if np.linalg.norm(positions[i]) <= 2.9 or i in site_atoms
    ]

    for i in top_indices:
        for j in top_indices:
            if j <= i or {atoms[i].symbol, atoms[j].symbol} != {"Zn", "O"}:
                continue
            if 1.4 <= float(atoms.get_distance(i, j)) <= 2.5:
                ax.plot(
                    positions[[i, j], 0], positions[[i, j], 1],
                    color="#CBD1D8", linewidth=1.2, zorder=1,
                )
    for oxygen_index in (len(atoms) - 3, len(atoms) - 1):
        ax.plot(
            positions[[oxygen_index, carbon_index], 0],
            positions[[oxygen_index, carbon_index], 1],
            color="#6D7580", linewidth=2.0, zorder=2,
        )

    for index in top_indices + list(range(len(atoms) - 3, len(atoms))):
        atom = atoms[index]
        point = positions[index]
        ax.add_patch(
            Circle(
                point, RADII[atom.symbol], facecolor=COLORS[atom.symbol],
                edgecolor="white", linewidth=0.7,
                alpha=0.55 if index in top_indices else 1.0,
                zorder=3 if index in top_indices else 4,
            )
        )
        if index >= len(atoms) - 3:
            ax.text(
                point[0], point[1], atom.symbol, ha="center", va="center",
                color="white", fontsize=7, fontweight="bold", zorder=5,
            )

    for index in site_atoms:
        atom = atoms[index]
        ax.add_patch(
            Circle(
                positions[index], RADII[atom.symbol] + 0.085,
                facecolor="none", edgecolor="#1D252D", linewidth=1.15, zorder=6,
            )
        )

    if show_title:
        ax.set_title(f"({letter}) {SITE_NAMES[site]}", fontsize=9.0, fontweight="semibold", pad=4)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")


def add_legend(fig) -> None:
    legend = [
        Line2D([], [], linestyle="none", marker="o", markersize=5.5,
               markerfacecolor=COLORS[symbol], markeredgecolor="white", label=symbol)
        for symbol in ("Zn", "O", "C")
    ]
    legend.append(
        Line2D([], [], linestyle="none", marker="o", markersize=7,
               markerfacecolor="none", markeredgecolor="#1D252D", label="Starting site")
    )
    fig.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=4, frameon=False, fontsize=7.2, handletextpad=0.25, columnspacing=0.9,
    )


def save_view(data_by_site: dict[str, SiteData], view: str, args, suffix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    draw = draw_side_view if view == "side" else draw_top_view
    for column, (site, letter) in enumerate(zip(SITES, "abc")):
        data = data_by_site[site]
        draw(axes[column], data, site, letter, data.site_atoms)
        axes[column].set_position([0.035 + column * 0.32, 0.16, 0.29, 0.76])
    add_legend(fig)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    view_suffix = "" if view == "side" else "_top"
    target = args.output_dir / f"zno_adsorption_sites{view_suffix}{suffix}.png"
    fig.savefig(target, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    print(f"[saved] {target}")
    plt.close(fig)


def save_combined(data_by_site: dict[str, SiteData], args, suffix: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for column, (site, letter) in enumerate(zip(SITES, "abc")):
        data = data_by_site[site]
        draw_top_view(axes[0, column], data, site, letter, data.site_atoms)
        draw_side_view(
            axes[1, column], data, site, letter, data.site_atoms, show_title=False
        )
        left = 0.075 + column * 0.31
        axes[0, column].set_position([left, 0.55, 0.27, 0.35])
        axes[1, column].set_position([left, 0.16, 0.27, 0.35])

    fig.text(0.025, 0.725, "Top view", rotation=90, ha="center", va="center", fontsize=7.3)
    fig.text(0.025, 0.335, "Side view", rotation=90, ha="center", va="center", fontsize=7.3)
    fig.add_artist(
        Line2D(
            [0.06, 0.96], [0.53, 0.53], transform=fig.transFigure,
            color="#D3D8DE", linewidth=0.55,
        )
    )
    add_legend(fig)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"zno_adsorption_sites_combined{suffix}.png"
    fig.savefig(target, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    print(f"[saved] {target}")
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    check_inputs(args)
    summary = load_summary(args.summary)
    with args.site_metadata.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    initial_slab = read(args.initial_slab)
    data_by_site = {}
    warnings = []
    for site in SITES:
        source = args.input_dir / site / f"ZnO_{site}_optimized.xyz"
        atoms = read(source)
        initial_xy = np.asarray(metadata["site_positions_angstrom"][site][:2], dtype=float)
        anchors = starting_site_atoms(initial_slab, initial_xy, site)
        data = SiteData(atoms, geometry_metrics(atoms), anchors)
        if site not in summary:
            raise ValueError(f"{site}: missing row in the DFT summary")
        warnings.extend(validate(site, data, summary[site]))
        data_by_site[site] = data

    for warning in warnings:
        print(f"[geometry warning] {warning}", flush=True)
    if warnings and not args.allow_geometry_warnings:
        raise SystemExit(
            "Figure not rendered: review the saved geometries first. "
            "Use --allow-geometry-warnings only for a diagnostic draft."
        )

    suffix = "_diagnostic" if warnings else ""
    if args.view == "combined":
        save_combined(data_by_site, args, suffix)
    else:
        views = ("side", "top") if args.view == "both" else (args.view,)
        for view in views:
            save_view(data_by_site, view, args, suffix)


if __name__ == "__main__":
    main()
