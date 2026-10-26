"""CO2 -> CO pathway diagram from co2rr_pathway_summary.csv.

Only the consistent sequence * + CO2(g) -> COOH* -> CO* + H2O(g) -> * + CO(g) is drawn.
The y axis says dG (CHE free energy) or dE (electronic energy only) from the
summary's energy_basis column; O* is never part of this profile.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

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
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "analysis" / "co2rr")
    parser.add_argument("--material", default="ZnO")
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def site_profiles(rows: list[dict[str, str]], material: str) -> tuple[list[dict[str, object]], str]:
    """Cumulative profile per ok site; all rows must share one energy basis."""
    usable = [
        row for row in rows
        if row.get("material") == material and row.get("pathway_status") == "ok" and row.get("energy_basis")
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
        profiles.append(
            {
                "site": row["site"],
                "profile": pathway_profile(steps),
                "limiting_step": row.get("limiting_step", ""),
                "limiting_potential_v": safe_float(row.get("limiting_potential_v")),
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
    profiles, basis = site_profiles(read_rows(args.summary), args.material)
    free_energy = basis == "free_energy_CHE"
    symbol = "ΔG" if free_energy else "ΔE"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    value_column = "relative_free_energy_ev" if free_energy else "relative_electronic_energy_ev"
    table_path = out_dir / "co2rr_pathway_profile_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["site", "state", value_column, "energy_basis", "potential_v_rhe"])
        writer.writeheader()
        for item in profiles:
            for state, value in zip(PATHWAY_STATES, item["profile"]):
                writer.writerow(
                    {"site": item["site"], "state": state, value_column: value,
                     "energy_basis": basis, "potential_v_rhe": item["potential_v_rhe"]}
                )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(PATHWAY_STATES)))
    for item in profiles:
        label = SITE_LABELS.get(str(item["site"]), str(item["site"]))
        label += f" ($U_L$ = {item['limiting_potential_v']:.2f} V vs RHE)"
        if item["geometry_flags"]:
            label += f" [{item['geometry_flags']}]"
        # Horizontal level per state joined by thin dashed connectors.
        for idx, value in enumerate(item["profile"]):
            ax.hlines(value, idx - 0.3, idx + 0.3, linewidth=2.5, color=f"C{profiles.index(item)}",
                      label=label if idx == 0 else None)
            if idx:
                ax.plot([idx - 0.7, idx - 0.3], [item["profile"][idx - 1], value], linestyle="--",
                        linewidth=0.8, color=f"C{profiles.index(item)}")

    potential = profiles[0]["potential_v_rhe"] if profiles else 0.0
    ax.axhline(0.0, linestyle=":", linewidth=0.8, color="0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(PATHWAY_STATES, rotation=15)
    ax.set_xlabel("Reaction coordinate")
    if free_energy:
        ax.set_ylabel(f"{symbol} (eV), U = {potential:.2f} V vs RHE")
    else:
        ax.set_ylabel(f"{symbol} (eV), electronic energies only")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / "co2rr_pathway.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    for item in profiles:
        print(f"Site: {item['site']} ({basis})")
        print(f"  Limiting step: {item['limiting_step']}, U_L = {item['limiting_potential_v']:.3f} V vs RHE")
        print(f"  Profile ({symbol}, eV): {[round(v, 4) for v in item['profile']]}")
    print(f"wrote {table_path}\nwrote {fig_path}")


if __name__ == "__main__":
    main()
