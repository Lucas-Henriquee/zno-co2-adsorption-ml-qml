from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.project_config import RESULTS_DIR
from scripts.simulations.neb.run_neb import (
    DEFAULT_BARRIER_RESOLUTION_EV,
    STATUS_CONVERGED,
    band_fit,
    barrier_metrics,
    discover_neb_runs,
    read_final_images,
    resolve_barrier_resolution,
)


# ARGUMENTS
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate NEB energy profiles and migration barrier comparison plots."
    )

    parser.add_argument(
        "--material",
        default="ZnO",
        help="Material folder below <results>/neb.",
    )

    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Directory with the NEB pair folders (default: <QML_ZNO_RESULTS_DIR>/neb/<material>).",
    )

    parser.add_argument(
        "--include-unconverged",
        action="store_true",
        help="Also plot pairs whose status is not 'converged' (flagged in the plots).",
    )

    parser.add_argument(
        "--barrier-resolution",
        type=float,
        default=None,
        help="Smallest forward barrier (eV) reported as resolved "
             "(default: methods.neb.barrier_resolution_ev of <results>/config.yaml, else 0.03).",
    )

    parser.add_argument(
        "--smooth-band",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlay the force-based spline of ase.utils.forcecurve.fit_images "
             "(NEBTools.plot_band) on each profile, using the stored image forces.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution of saved figures.",
    )

    return parser.parse_args(argv)


# BASIC HELPERS
def pathway_label(initial_site: str, final_site: str) -> str:
    return f"{initial_site} → {final_site}"


# LOAD ALL RUNS
def analyze_runs(
    runs: list[dict],
    include_unconverged: bool = False,
    resolution_ev: float = DEFAULT_BARRIER_RESOLUTION_EV,
    smooth_band: bool = False,
) -> list[dict]:
    """Barrier metrics per pair, recomputed from the stored band energies.

    The metrics stored in neb_run_summary.json are not used, so runs written with an
    older barrier definition are re-analysed consistently. Non-converged pairs are
    reported and skipped unless requested.
    """
    results = []

    for run in runs:

        folder = run["folder"]
        status = run["status"]
        converged = status == STATUS_CONVERGED

        if not converged and not include_unconverged:
            print(f"[SKIP] {folder.name}: status={status or 'unknown'}")
            continue

        images = read_final_images(folder)
        energies = []
        for atoms in images:
            try:
                energies.append(float(atoms.get_potential_energy()))
            except Exception:
                energies.append(float("nan"))
        metrics = barrier_metrics(energies, resolution_ev=resolution_ev)

        if metrics is None:
            print(f"[WARNING] {folder.name}: no valid image energies (status={status})")
            continue

        result = {
            **run,
            **metrics,
            "pathway": pathway_label(run["initial_site"], run["final_site"]),
            "converged": converged,
            "climb_activated": bool(run["summary"].get("climb_activated", False)),
            "band_fit": band_fit(images) if smooth_band else None,
        }

        results.append(result)

        print(
            f"{folder.name:30s} "
            f"Ea_fwd={metrics['forward_barrier_eV']:8.4f} eV   "
            f"Ea_rev={metrics['reverse_barrier_eV']:8.4f} eV   "
            f"DeltaE={metrics['reaction_energy_eV']:8.4f} eV   "
            f"E_max image={metrics['highest_image_index']}"
            f"{'   [no internal saddle]' if metrics['no_internal_saddle'] else ''}"
            f"{'' if metrics['barrier_resolved'] else '   [no resolved barrier]'}"
            f"{'' if converged else f'   [{status}]'}"
        )

    return results


# SAVE CSV TABLES
def save_energy_table(results: list[dict], out: Path):
    rows = []

    for result in results:

        for image_index, (energy, relative) in enumerate(
            zip(
                result["energies"],
                result["relative_energies"],
            )
        ):

            rows.append(
                {
                    "pathway": result["pathway"],
                    "initial_site": result["initial_site"],
                    "final_site": result["final_site"],
                    "status": result["status"],
                    "folder": result["folder_name"],
                    "image_index": image_index,
                    "energy_eV": energy,
                    "relative_energy_eV": relative,
                }
            )

    df = pd.DataFrame(rows)

    path = out / "neb_energy_table.csv"

    df.to_csv(path, index=False)

    print(f"\nEnergy table saved -> {path}")


def save_summary_table(results: list[dict], out: Path):
    rows = []

    for result in results:

        rows.append(
            {
                "pathway": result["pathway"],
                "initial_site": result["initial_site"],
                "final_site": result["final_site"],
                "n_internal_images": result["n_internal_images"],
                "n_total_images": result["n_total_images"],
                "forward_barrier_eV": result["forward_barrier_eV"],
                "reverse_barrier_eV": result["reverse_barrier_eV"],
                "reaction_energy_eV": result["reaction_energy_eV"],
                "highest_image_index": result["highest_image_index"],
                "highest_internal_image_index": result["highest_internal_image_index"],
                "no_internal_saddle": result["no_internal_saddle"],
                "barrier_resolution_eV": result["barrier_resolution_eV"],
                "barrier_resolved": result["barrier_resolved"],
                "climb_activated": result["climb_activated"],
                "status": result["status"],
                "folder": result["folder_name"],
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["initial_site", "final_site"]
    )

    path = out / "neb_summary_table.csv"

    df.to_csv(path, index=False)

    print(f"Summary table saved -> {path}")


# INDIVIDUAL ENERGY PROFILE
def plot_individual_profile(result: dict, output_dir: Path, dpi: int):
    relative = result["relative_energies"]

    x = np.arange(len(relative))

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.plot(
        x,
        relative,
        marker="o",
        markersize=7,
        linewidth=2,
    )

    fit = result.get("band_fit")
    if fit is not None:
        # Smooth band against the image index: the spline is drawn on the path length,
        # mapped piecewise-linearly onto the image positions of the plot.
        fit_x = np.interp(fit["fit_path_ang"], fit["path_ang"], x)
        ax.plot(fit_x, fit["fit_relative_energies"], linewidth=1.2, linestyle="-", alpha=0.6,
                label="force-based spline")

    highest = result["highest_image_index"]

    ax.plot(
        [highest],
        [relative[highest]],
        marker="*",
        markersize=14,
        linestyle="none",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xticks(x)

    ax.set_xlabel(
        "NEB image index",
        fontsize=12,
    )

    ax.set_ylabel(
        "Relative energy (eV)",
        fontsize=12,
    )

    flags = []
    if result["no_internal_saddle"]:
        flags.append("no internal saddle")
    elif not result["barrier_resolved"]:
        flags.append(f"barrier below {result['barrier_resolution_eV']:.2f} eV resolution")
    if not result["converged"]:
        flags.append(f"status: {result['status']}")

    ax.set_title(
        f"NEB energy profile: "
        f"{result['initial_site']} → {result['final_site']}"
        + (f"\n[{'; '.join(flags)}]" if flags else ""),
        fontsize=14,
    )

    text = (
        f"Forward barrier = {result['forward_barrier_eV']:.3f} eV\n"
        f"Reverse barrier = {result['reverse_barrier_eV']:.3f} eV\n"
        f"ΔE = {result['reaction_energy_eV']:.3f} eV\n"
        f"highest image (E_max) = {highest}"
    )

    ax.text(
        0.98,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "alpha": 0.1,
        },
    )

    ax.grid(
        alpha=0.25,
    )

    if fit is not None:
        ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()

    filename = (
        f"neb_profile_"
        f"{result['initial_site']}_to_{result['final_site']}.png"
    )

    output_path = output_dir / filename

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


# MIGRATION BARRIERS
def plot_barriers(
    results: list[dict],
    output_dir: Path,
    dpi: int,
):
    selected = sorted(
        results,
        key=lambda item: (
            item["initial_site"],
            item["final_site"],
        ),
    )

    labels = []
    for result in selected:
        label = result["pathway"]
        if result["no_internal_saddle"]:
            label += "\n(no internal saddle)"
        elif not result["barrier_resolved"]:
            label += "\n(below resolution)"
        if not result["converged"]:
            label += f"\n[{result['status']}]"
        labels.append(label)

    forward = [result["forward_barrier_eV"] for result in selected]
    reverse = [result["reverse_barrier_eV"] for result in selected]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for offset, values, name in (
        (-width / 2, forward, "forward (E_max − E_IS)"),
        (width / 2, reverse, "reverse (E_max − E_FS)"),
    ):

        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.95,
            label=name,
        )

        for bar, value in zip(bars, values):

            if math.isfinite(value):

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        fontsize=10,
    )

    ax.set_ylabel(
        "Migration barrier (eV)",
        fontsize=12,
    )

    ax.set_title(
        "NEB Analysis — Migration Barrier Comparison",
        fontsize=15,
        fontweight="bold",
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = output_dir / "migration_barrier_comparison.png"

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Barrier comparison saved -> {output_path}"
    )


# MAIN
def main(argv=None):
    args = parse_args(argv)

    base = args.base if args.base is not None else RESULTS_DIR / "neb" / args.material

    output_dir = base / "comparison_plots"

    profiles_dir = output_dir / "individual_profiles"

    print("=" * 70)
    print("NEB ANALYSIS")
    print("=" * 70)

    print(f"Base directory: {base}")

    print()

    runs = discover_neb_runs(base)

    if not runs:

        raise RuntimeError(
            f"No NEB run folders (with neb_run_summary.json) found inside {base}"
        )

    print(
        f"Found {len(runs)} NEB runs:"
    )

    for run in runs:

        print(
            f"  {run['folder_name']}  status={run['status']}"
        )

    print()

    resolution = resolve_barrier_resolution(args.barrier_resolution, base)
    print(f"Barrier resolution: {resolution:.3f} eV (E_max over all images, endpoints included)")

    results = analyze_runs(
        runs,
        include_unconverged=args.include_unconverged,
        resolution_ev=resolution,
        smooth_band=args.smooth_band,
    )

    if not results:

        raise RuntimeError(
            "No converged NEB pair with valid image energies was found "
            "(use --include-unconverged to inspect the others)."
        )

    print()
    print(
        f"Successfully analyzed "
        f"{len(results)} runs."
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Tables
    save_energy_table(
        results,
        output_dir,
    )

    save_summary_table(
        results,
        output_dir,
    )

    # Individual plots
    print("\nGenerating individual profiles...")

    for result in results:

        plot_individual_profile(
            result,
            profiles_dir,
            args.dpi,
        )

    print("\nGenerating migration barrier comparison...")

    plot_barriers(
        results,
        output_dir,
        args.dpi,
    )

    print()
    print("=" * 70)
    print("NEB ANALYSIS COMPLETED")
    print("=" * 70)

    print("Results saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()
