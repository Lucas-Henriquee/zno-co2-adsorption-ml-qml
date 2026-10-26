from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.common.project_config import RESULTS_DIR
from scripts.simulations.neb.run_neb import (
    STATUS_CONVERGED,
    barrier_metrics,
    discover_neb_runs,
    read_image_energies,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Tabulate NEB image energies and barriers per pair.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Directory with the NEB pair folders (default: <QML_ZNO_RESULTS_DIR>/neb/<material>).",
    )
    return parser.parse_args(argv)


def build_rows(base: Path) -> list[dict]:
    rows = []

    print(f"{'pair':<26}{'image':>7}{'abs_energy_eV':>16}{'rel_energy_eV':>16}")
    print("-" * 65)

    for run in discover_neb_runs(base):
        label = run["folder_name"]
        status = run["status"]
        meta = run["summary"]
        metrics = barrier_metrics(read_image_energies(run["folder"]))

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
                "is_highest_internal_image": i == metrics["highest_image_index"],
            })

        flag = "  [no internal saddle]" if metrics["no_internal_saddle"] else ""
        print(
            f"  -> forward barrier = {metrics['forward_barrier_eV']:.4f} eV  |  "
            f"reverse barrier = {metrics['reverse_barrier_eV']:.4f} eV  |  "
            f"reaction dE = {metrics['reaction_energy_eV']:.4f} eV  |  "
            f"highest internal image {metrics['highest_image_index']}{flag}"
        )
        print(
            f"  -> status = {status}{'' if status == STATUS_CONVERGED else '  (NOT CONVERGED)'}  |  "
            f"climb_activated = {meta.get('climb_activated', '?')}  |  "
            f"fmax CI = {meta.get('fmax_ci', '?')}"
        )
        print()

    return rows


def main(argv=None):
    args = parse_args(argv)
    base = args.base if args.base is not None else RESULTS_DIR / "neb" / args.material

    rows = build_rows(base)

    if not rows:
        print(f"No NEB pair with valid image energies in {base}")
        return

    out_csv = base / "comparison_plots" / "neb_image_energies.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Complete table saved to: {out_csv}")


if __name__ == "__main__":
    main()
