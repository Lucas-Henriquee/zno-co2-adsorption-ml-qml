from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.common.project_config import RESULTS_DIR
from scripts.simulations.neb.run_neb import (
    DEFAULT_BARRIER_RESOLUTION_EV,
    STATUS_CONVERGED,
    barrier_metrics,
    discover_neb_runs,
    read_image_energies,
    resolve_barrier_resolution,
)

BARRIER_FIELDS = (
    "forward_barrier_eV",
    "reverse_barrier_eV",
    "reaction_energy_eV",
    "highest_image_index",
    "highest_internal_image_index",
    "no_internal_saddle",
    "barrier_resolution_eV",
    "barrier_resolved",
    "n_total_images",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Tabulate NEB image energies and barriers per pair. Barriers are recomputed from "
            "the stored band energies (neb_final_images.traj), not read from neb_run_summary.json."
        )
    )
    parser.add_argument("--material", default="ZnO")
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Directory with the NEB pair folders (default: <QML_ZNO_RESULTS_DIR>/neb/<material>).",
    )
    parser.add_argument(
        "--barrier-resolution",
        type=float,
        default=None,
        help="Smallest forward barrier (eV) reported as resolved "
             "(default: methods.neb.barrier_resolution_ev of <results>/config.yaml, else 0.03).",
    )
    return parser.parse_args(argv)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_rows(base: Path, resolution_ev: float = DEFAULT_BARRIER_RESOLUTION_EV) -> tuple[list[dict], list[dict]]:
    """(per-image rows, per-pair barrier rows) recomputed from the band energies."""
    rows = []
    barrier_rows = []

    print(f"{'pair':<26}{'image':>7}{'abs_energy_eV':>16}{'rel_energy_eV':>16}")
    print("-" * 65)

    for run in discover_neb_runs(base):
        label = run["folder_name"]
        status = run["status"]
        meta = run["summary"]
        metrics = barrier_metrics(read_image_energies(run["folder"]), resolution_ev=resolution_ev)

        if metrics is None:
            print(f"{label:<26}  NO VALID IMAGE ENERGIES (status={status})")
            continue

        for i, (e_abs, e_rel) in enumerate(zip(metrics["energies"], metrics["relative_energies"])):
            print(f"{label:<26}{i:>7}{e_abs:>16.6f}{e_rel:>16.6f}")
            rows.append({
                "pair": label,
                "initial_site": run["initial_site"],
                "final_site": run["final_site"],
                "status": status,
                "image_index": i,
                "abs_energy_eV": e_abs,
                "rel_energy_eV": e_rel,
                "is_highest_image": i == metrics["highest_image_index"],
                "is_highest_internal_image": i == metrics["highest_internal_image_index"],
            })

        barrier_rows.append({
            "pair": label,
            "initial_site": run["initial_site"],
            "final_site": run["final_site"],
            "status": status,
            **{key: metrics[key] for key in BARRIER_FIELDS},
            "climb_activated": meta.get("climb_activated", ""),
        })

        flags = []
        if metrics["no_internal_saddle"]:
            flags.append("no internal saddle")
        elif not metrics["barrier_resolved"]:
            flags.append(f"below {resolution_ev:.3f} eV resolution")
        flag = f"  [{'; '.join(flags)}]" if flags else ""
        print(
            f"  -> forward barrier = {metrics['forward_barrier_eV']:.4f} eV  |  "
            f"reverse barrier = {metrics['reverse_barrier_eV']:.4f} eV  |  "
            f"reaction dE = {metrics['reaction_energy_eV']:.4f} eV  |  "
            f"E_max image {metrics['highest_image_index']}{flag}"
        )
        print(
            f"  -> status = {status}{'' if status == STATUS_CONVERGED else '  (NOT CONVERGED)'}  |  "
            f"climb_activated = {meta.get('climb_activated', '?')}  |  "
            f"fmax CI = {meta.get('fmax_ci', '?')}"
        )
        print()

    return rows, barrier_rows


def main(argv=None):
    args = parse_args(argv)
    base = args.base if args.base is not None else RESULTS_DIR / "neb" / args.material
    resolution = resolve_barrier_resolution(args.barrier_resolution, base)
    print(f"Barrier resolution: {resolution:.3f} eV (E_max over all images, endpoints included)\n")

    rows, barrier_rows = build_rows(base, resolution)

    if not rows:
        print(f"No NEB pair with valid image energies in {base}")
        return

    out_csv = base / "comparison_plots" / "neb_image_energies.csv"
    write_rows(out_csv, rows)
    print(f"Complete table saved to: {out_csv}")
    barrier_csv = base / "comparison_plots" / "neb_barriers.csv"
    write_rows(barrier_csv, barrier_rows)
    print(f"Recomputed barriers saved to: {barrier_csv}")


if __name__ == "__main__":
    main()
