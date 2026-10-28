"""CO2 -> CO pathway diagram from co2rr_corrected_summary.csv or co2rr_pathway_summary.csv.

Only the consistent sequence * + CO2(g) -> COOH* -> CO* + H2O(g) -> * + CO(g) is drawn.
The y axis says dG (CHE free energy) or dE (electronic energy only) from the
summary's energy_basis column; O* is never part of this profile. When the corrected file
of co2rr_corrections.py exists its variant (default c) is drawn at U = 0 V vs RHE.
Sites whose COOH* or CO* desorbed are drawn in grey, labelled "unbound", without U_L.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scripts.analysis.co2rr_corrections import (
    CORRECTED_CSV,
    DEFAULT_VARIANT,
    UNBOUND_STATUS,
    VARIANTS,
    is_unbound,
    load_corrected_pathway_rows,
)
from scripts.common.project_config import RESULTS_DIR
from scripts.simulations.co2rr.run_co2rr_pathways import (
    PATHWAY_STATES,
    PATHWAY_STEPS,
    energy_prefix,
    pathway_profile,
    safe_float,
)

SITE_ORDER = ("top_metal", "top_oxygen", "bridge")
SITE_LABELS = {"top_metal": "Top metal", "top_oxygen": "Top oxygen", "bridge": "Bridge"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the CO2 -> CO pathway from co2rr_pathway_summary.csv.")
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "co2rr_pathway_summary.csv")
    parser.add_argument(
        "--corrected",
        type=Path,
        default=None,
        help=f"Corrected file of co2rr_corrections.py (default: {CORRECTED_CSV} next to --summary); "
             "used when it exists unless --uncorrected.",
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=DEFAULT_VARIANT,
                        help="Variant of the corrected file (a, b or c; default c).")
    parser.add_argument("--uncorrected", action="store_true", help="Plot co2rr_pathway_summary.csv as written by the run.")
    parser.add_argument(
        "--unbound",
        choices=("grey", "hide"),
        default="grey",
        help="Unbound sites (COOH*/CO* desorbed): draw greyed and labelled, or leave out.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "analysis" / "co2rr")
    parser.add_argument("--material", default="ZnO")
    return parser.parse_args(argv)


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    """Rows to plot and a label of their source."""
    if not args.uncorrected:
        corrected_path = args.corrected if args.corrected is not None else args.summary.parent / CORRECTED_CSV
        rows = load_corrected_pathway_rows(corrected_path, args.variant)
        if rows is not None:
            return rows, f"corrected variant {args.variant}"
    return read_rows(args.summary), "as computed"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def site_profiles(
    rows: list[dict[str, str]], material: str, include_unbound: bool = True
) -> tuple[list[dict[str, object]], str]:
    """Cumulative profile per ok (and, if requested, unbound) site; one energy basis only.

    A site is unbound when its status says so or its geometry_flags contain "_desorbed";
    its profile is returned with ``unbound`` True and no limiting potential.
    """
    usable = [
        row for row in rows
        if row.get("material") == material
        and row.get("pathway_status") in ("ok", UNBOUND_STATUS)
        and row.get("energy_basis")
    ]
    bases = {row["energy_basis"] for row in usable}
    if len(bases) > 1:
        raise SystemExit(f"Rows mix energy bases {sorted(bases)}; rerun the pathway step.")
    if not usable:
        raise SystemExit(f"No ok pathway rows with an energy_basis column for {material}.")
    basis = bases.pop()
    prefix = energy_prefix(basis == "free_energy_CHE")

    order = {site: idx for idx, site in enumerate(SITE_ORDER)}
    profiles: list[dict[str, object]] = []
    for row in sorted(usable, key=lambda item: order.get(item["site"], 99)):
        steps = {step: safe_float(row.get(f"{prefix}_{step}_ev")) for step in PATHWAY_STEPS}
        if not all(math.isfinite(value) for value in steps.values()):
            continue
        unbound = row.get("pathway_status") == UNBOUND_STATUS or is_unbound(row.get("geometry_flags", ""))
        if unbound and not include_unbound:
            continue
        profiles.append(
            {
                "site": row["site"],
                "profile": pathway_profile(steps),
                "unbound": unbound,
                "limiting_step": "" if unbound else row.get("limiting_step", ""),
                "limiting_potential_v": float("nan") if unbound else safe_float(row.get("limiting_potential_v")),
                "potential_v_rhe": safe_float(row.get("potential_v_rhe")),
                "geometry_flags": row.get("geometry_flags", ""),
            }
        )
    return profiles, basis


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    rows, source = load_rows(args)
    profiles, basis = site_profiles(rows, args.material, include_unbound=args.unbound == "grey")
    free_energy = basis == "free_energy_CHE"
    symbol = "ΔG" if free_energy else "ΔE"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    value_column = "relative_free_energy_ev" if free_energy else "relative_electronic_energy_ev"
    table_path = out_dir / "co2rr_pathway_profile_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["site", "state", value_column, "energy_basis", "source", "unbound", "potential_v_rhe"]
        )
        writer.writeheader()
        for item in profiles:
            for state, value in zip(PATHWAY_STATES, item["profile"]):
                writer.writerow(
                    {"site": item["site"], "state": state, value_column: value, "energy_basis": basis,
                     "source": source, "unbound": item["unbound"], "potential_v_rhe": item["potential_v_rhe"]}
                )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(PATHWAY_STATES)))
    bound_index = 0
    for item in profiles:
        label = SITE_LABELS.get(str(item["site"]), str(item["site"]))
        if item["unbound"]:
            # Not a bound pathway: grey, no U_L, behind the bound sites.
            label += " (unbound, no $U_L$)"
            color, alpha, zorder = "0.6", 0.7, 1
        else:
            label += f" ($U_L$ = {item['limiting_potential_v']:.2f} V vs RHE)"
            color, alpha, zorder = f"C{bound_index}", 1.0, 2
            bound_index += 1
        if item["geometry_flags"]:
            label += f" [{item['geometry_flags']}]"
        # Horizontal level per state joined by thin dashed connectors.
        for idx, value in enumerate(item["profile"]):
            ax.hlines(value, idx - 0.3, idx + 0.3, linewidth=2.5, color=color, alpha=alpha, zorder=zorder,
                      label=label if idx == 0 else None)
            if idx:
                ax.plot([idx - 0.7, idx - 0.3], [item["profile"][idx - 1], value], linestyle="--",
                        linewidth=0.8, color=color, alpha=alpha, zorder=zorder)

    potential = profiles[0]["potential_v_rhe"] if profiles else 0.0
    ax.axhline(0.0, linestyle=":", linewidth=0.8, color="0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(PATHWAY_STATES, rotation=15)
    ax.set_xlabel("Reaction coordinate")
    if free_energy:
        ax.set_ylabel(f"{symbol} (eV), U = {potential:.2f} V vs RHE")
        ax.set_title(f"{args.material}: CHE free-energy profile ({source})", fontsize=10)
    else:
        ax.set_ylabel(f"{symbol} (eV), electronic energies only")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / "co2rr_pathway.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    for item in profiles:
        print(f"Site: {item['site']} ({basis}, {source})")
        if item["unbound"]:
            print("  Unbound intermediate: no limiting potential")
        else:
            print(f"  Limiting step: {item['limiting_step']}, U_L = {item['limiting_potential_v']:.3f} V vs RHE")
        print(f"  Profile ({symbol}, eV): {[round(v, 4) for v in item['profile']]}")
    print(f"wrote {table_path}\nwrote {fig_path}")


if __name__ == "__main__":
    main()
