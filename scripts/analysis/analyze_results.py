"""Tabulate successful DFT adsorption results of one simulation run.

Only rows with dft_status == ok are used (audit A1), and the exploratory DFT site
"carbonate" is excluded. Candidates are ranked by the DFT adsorption energy alone
(most negative first); no composite score is computed.

The former photocatalysis score compared the PBE band gap and the TDDFT "onset"
with experimental targets (2.40 and 2.20 eV). Both comparisons were invalid:
PBE Kohn-Sham gaps underestimate the experimental ZnO gap (3.37 eV), and the
TDDFT descriptors are excitations of the isolated CO2 molecule, not of the
ZnO/CO2 system. Both are reported for information only.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scripts.common.project_config import EXPLORATORY_SITES, RESULTS_DIR

TDDFT_COLUMNS = (
    "co2_isolated_first_excitation_ev",
    "co2_isolated_onset_ev",
    "co2_isolated_peak_ev",
    "co2_isolated_total_f",
    "trk_sum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank successful material/site DFT results by adsorption energy."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Simulation run directory, e.g. results/simulations/<run-id> (default: QML_ZNO_RESULTS_DIR).",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "analysis")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def safe_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def is_ok(row: dict[str, str], status_column: str) -> bool:
    return str(row.get(status_column, "")).strip().lower() == "ok"


def filter_ok_rows(rows: list[dict[str, str]], status_column: str) -> list[dict[str, str]]:
    """Successful rows of the classification sites (failed and carbonate rows dropped)."""
    return [
        row
        for row in rows
        if is_ok(row, status_column) and row.get("site") not in EXPLORATORY_SITES
    ]


def merge_results(
    dft_rows: list[dict[str, str]],
    tddft_rows: list[dict[str, str]],
) -> list[dict[str, float | str]]:
    dft_ok = filter_ok_rows(dft_rows, "dft_status")
    tddft_map: dict[tuple[str, str], dict[str, str]] = {
        (row["material"], row["site"]): row for row in filter_ok_rows(tddft_rows, "tddft_status")
    }

    merged: list[dict[str, float | str]] = []
    for dft_row in dft_ok:
        key = (dft_row["material"], dft_row["site"])
        tddft_row = tddft_map.get(key, {})

        adsorption_energy = safe_float(dft_row.get("adsorption_energy_ev"))
        if not math.isfinite(adsorption_energy):
            continue
        clean_gap = safe_float(dft_row.get("clean_band_gap_gamma_ev"))
        adsorbed_gap = safe_float(dft_row.get("adsorbed_band_gap_gamma_ev"))
        co_bond_1 = safe_float(dft_row.get("co_bond_1_ang"))
        co_bond_2 = safe_float(dft_row.get("co_bond_2_ang"))

        row: dict[str, float | str] = {
            "material": dft_row["material"],
            "site": dft_row["site"],
            "final_state": dft_row.get("final_state", ""),
            "adsorption_energy_ev": adsorption_energy,
            # PBE Kohn-Sham gaps on a Gamma-centred grid; not comparable to 3.37 eV.
            "clean_band_gap_gamma_ev": clean_gap,
            "adsorbed_band_gap_gamma_ev": adsorbed_gap,
            "pbe_gap_change_ev": adsorbed_gap - clean_gap,
            "co_bond_avg_ang": (co_bond_1 + co_bond_2) / 2.0,
            "oco_angle_deg": safe_float(dft_row.get("oco_angle_deg")),
            "c_surface_distance_ang": safe_float(dft_row.get("c_surface_distance_ang")),
            "c_min_slab_distance_ang": safe_float(dft_row.get("c_min_slab_distance_ang")),
            "o_min_metal_distance_ang": safe_float(dft_row.get("o_min_metal_distance_ang")),
        }
        for column in TDDFT_COLUMNS:
            row[column] = safe_float(tddft_row.get(column))
        merged.append(row)

    merged.sort(key=lambda item: float(item["adsorption_energy_ev"]))
    for rank, row in enumerate(merged, start=1):
        row["rank_by_adsorption_energy"] = rank
    return merged


def write_ranked_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        raise SystemExit("No successful DFT rows (dft_status == ok) to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | str) -> str:
    number = safe_float(value) if not isinstance(value, float) else value
    return f"{number:.4f}" if math.isfinite(number) else "n/a"


def write_markdown_report(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        raise SystemExit("No rows to report.")

    lines: list[str] = []
    lines.append("# DFT Adsorption Summary")
    lines.append("")
    lines.append(
        "Only successful DFT rows (dft_status == ok) of the classification sites are "
        "listed; the exploratory carbonate start is excluded. Ranking is by adsorption "
        "energy only (most negative first)."
    )
    lines.append("")
    lines.append(
        "Band gaps are PBE Kohn-Sham gaps on a Gamma-centred grid. PBE underestimates "
        "the experimental ZnO gap (3.37 eV); use them only to compare sites. TDDFT "
        "values describe the isolated CO2 molecule and are not photocatalyst absorption "
        "energies."
    )
    lines.append("")
    lines.append(
        "| Rank | Material | Site | E_ads (eV) | PBE gap clean (eV) | PBE gap adsorbed (eV) "
        "| PBE gap change (eV) | O-C-O (deg) |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['rank_by_adsorption_energy']} | {row['material']} | {row['site']} "
            f"| {_fmt(row['adsorption_energy_ev'])} | {_fmt(row['clean_band_gap_gamma_ev'])} "
            f"| {_fmt(row['adsorbed_band_gap_gamma_ev'])} | {_fmt(row['pbe_gap_change_ev'])} "
            f"| {_fmt(row['oco_angle_deg'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dft_csv = results_dir / "dft_summary.csv"
    tddft_csv = results_dir / "tddft_summary.csv"
    ranked_csv = output_dir / "ranked_candidates.csv"
    report_md = output_dir / "ranking_report.md"

    dft_rows = read_csv_rows(dft_csv)
    if tddft_csv.is_file():
        tddft_rows = read_csv_rows(tddft_csv)
    else:
        print(f"[analysis] {tddft_csv} not found; TDDFT columns left empty.")
        tddft_rows = []
    merged_rows = merge_results(dft_rows, tddft_rows)

    write_ranked_csv(merged_rows, ranked_csv)
    write_markdown_report(merged_rows, report_md)

    print(f"[analysis] wrote {ranked_csv}")
    print(f"[analysis] wrote {report_md}")


if __name__ == "__main__":
    main()
