from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scripts.common.project_config import RESULTS_DIR


ANALYSIS_DIR = RESULTS_DIR / "analysis"
BASELINE_CO_BOND_ANG = 1.16
TARGET_ADSORPTION_EV = -0.70
TARGET_ONSET_EV = 2.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CO2-reduction analysis and figure-ready tables."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    return parser.parse_args()


def read_csv_rows(path: Path, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise SystemExit(f"Missing file: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def gaussian_score(value: float, target: float, sigma: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return math.exp(-((value - target) ** 2) / (2.0 * sigma**2))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        raise SystemExit("No rows available to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge_dft_tddft(
    dft_rows: list[dict[str, str]],
    tddft_rows: list[dict[str, str]],
) -> list[dict[str, str | float]]:
    tddft_map = {(row["material"], row["site"]): row for row in tddft_rows}
    merged: list[dict[str, str | float]] = []

    for dft_row in dft_rows:
        material = dft_row["material"]
        site = dft_row["site"]
        tddft_row = tddft_map.get((material, site), {})

        dft_status = dft_row.get("dft_status", "")
        tddft_status = tddft_row.get("tddft_status", "")

        adsorption_energy = safe_float(dft_row.get("adsorption_energy_ev"))
        co_bond_1 = safe_float(dft_row.get("co_bond_1_ang"))
        co_bond_2 = safe_float(dft_row.get("co_bond_2_ang"))
        oco_angle = safe_float(dft_row.get("oco_angle_deg"))
        c_surface_distance = safe_float(dft_row.get("c_surface_distance_ang"))

        onset = safe_float(tddft_row.get("tddft_onset_ev"))
        peak_energy = safe_float(tddft_row.get("tddft_peak_energy_ev"))
        peak_osc = safe_float(tddft_row.get("tddft_peak_oscillator_strength"))
        total_osc = safe_float(tddft_row.get("tddft_total_oscillator_strength"))
        num_transitions = safe_float(tddft_row.get("tddft_num_transitions"))

        co_bond_avg = (co_bond_1 + co_bond_2) / 2.0 if math.isfinite(co_bond_1) and math.isfinite(co_bond_2) else float("nan")
        co_bond_stretch = co_bond_avg - BASELINE_CO_BOND_ANG if math.isfinite(co_bond_avg) else float("nan")
        oco_bend = 180.0 - oco_angle if math.isfinite(oco_angle) else float("nan")

        adsorption_score = gaussian_score(adsorption_energy, TARGET_ADSORPTION_EV, sigma=0.50)
        optical_score = gaussian_score(onset, TARGET_ONSET_EV, sigma=0.80)
        activation_score = clamp01(
            0.5 * max(0.0, (co_bond_stretch if math.isfinite(co_bond_stretch) else 0.0) / 0.20)
            + 0.5 * max(0.0, (oco_bend if math.isfinite(oco_bend) else 0.0) / 40.0)
        )

        if dft_status == "ok" and tddft_status == "ok":
            reduction_proxy_score = (
                0.45 * adsorption_score
                + 0.35 * activation_score
                + 0.20 * optical_score
            )
            quality_flag = "ok"
        else:
            reduction_proxy_score = float("nan")
            quality_flag = "incomplete"

        merged.append(
            {
                "material": material,
                "site": site,
                "dft_status": dft_status,
                "tddft_status": tddft_status,
                "dft_profile": dft_row.get("dft_profile", ""),
                "tddft_mode": tddft_row.get("tddft_mode", ""),
                "adsorption_energy_ev": adsorption_energy,
                "co_bond_avg_ang": co_bond_avg,
                "co_bond_stretch_ang": co_bond_stretch,
                "oco_angle_deg": oco_angle,
                "oco_bend_deg": oco_bend,
                "c_surface_distance_ang": c_surface_distance,
                "tddft_onset_ev": onset,
                "tddft_peak_energy_ev": peak_energy,
                "tddft_peak_oscillator_strength": peak_osc,
                "tddft_total_oscillator_strength": total_osc,
                "tddft_num_transitions": num_transitions,
                "adsorption_score": adsorption_score,
                "activation_score": activation_score,
                "optical_score": optical_score,
                "reduction_proxy_score": reduction_proxy_score,
                "quality_flag": quality_flag,
            }
        )

    merged.sort(
        key=lambda row: float(row["reduction_proxy_score"])
        if math.isfinite(float(row["reduction_proxy_score"]))
        else -1.0,
        reverse=True,
    )
    return merged


def build_figure_tables(rows: list[dict[str, str | float]]) -> dict[str, list[dict[str, str | float]]]:
    heatmap_rows = [
        {
            "material": row["material"],
            "site": row["site"],
            "adsorption_energy_ev": row["adsorption_energy_ev"],
            "reduction_proxy_score": row["reduction_proxy_score"],
        }
        for row in rows
    ]

    activation_rows = [
        {
            "material": row["material"],
            "site": row["site"],
            "co_bond_avg_ang": row["co_bond_avg_ang"],
            "oco_angle_deg": row["oco_angle_deg"],
            "co_bond_stretch_ang": row["co_bond_stretch_ang"],
            "oco_bend_deg": row["oco_bend_deg"],
            "reduction_proxy_score": row["reduction_proxy_score"],
        }
        for row in rows
    ]

    photo_thermo_rows = [
        {
            "material": row["material"],
            "site": row["site"],
            "adsorption_energy_ev": row["adsorption_energy_ev"],
            "tddft_onset_ev": row["tddft_onset_ev"],
            "tddft_peak_energy_ev": row["tddft_peak_energy_ev"],
            "tddft_peak_oscillator_strength": row["tddft_peak_oscillator_strength"],
            "reduction_proxy_score": row["reduction_proxy_score"],
        }
        for row in rows
    ]

    rank_rows = [
        {
            "rank": idx + 1,
            "material": row["material"],
            "site": row["site"],
            "reduction_proxy_score": row["reduction_proxy_score"],
            "quality_flag": row["quality_flag"],
        }
        for idx, row in enumerate(rows)
    ]

    return {
        "figure_adsorption_heatmap.csv": heatmap_rows,
        "figure_activation_scatter.csv": activation_rows,
        "figure_photo_thermo_map.csv": photo_thermo_rows,
        "figure_site_ranking.csv": rank_rows,
    }


PATHWAY_BASE_COLUMNS = ("pathway_status", "energy_basis", "profile", "limiting_step", "limiting_potential_v")


def pathway_columns(pathway_rows: list[dict[str, str]]) -> list[str]:
    """Columns of co2rr_pathway_summary.csv in file order (dG_* or dE_* as written)."""
    columns: list[str] = list(PATHWAY_BASE_COLUMNS)
    for row in pathway_rows:
        for key in row:
            if key not in ("material", "site", "notes") and key not in columns:
                columns.append(key)
    return columns


def numeric_or_text(value: str | None) -> str | float:
    number = safe_float(value)
    if math.isfinite(number):
        return number
    text = "" if value is None else str(value).strip()
    return text if text and text.lower() != "nan" else float("nan")


def build_pathway_template(
    rows: list[dict[str, str | float]],
    columns: list[str],
) -> list[dict[str, str | float]]:
    template_rows: list[dict[str, str | float]] = []
    for row in rows:
        template_row: dict[str, str | float] = {
            "material": row["material"],
            "site": row["site"],
            "quality_flag": row["quality_flag"],
        }
        for column in columns:
            template_row[column] = "" if column in ("pathway_status", "energy_basis", "profile", "limiting_step") else float("nan")
        template_row["notes"] = ""
        template_rows.append(template_row)
    return template_rows


def build_pathway_table(
    rows: list[dict[str, str | float]],
    pathway_rows: list[dict[str, str]],
) -> list[dict[str, str | float]]:
    # Summaries without energy_basis predate the CHE fix (O* summed into the profile,
    # dE labelled dG, wrong limiting potential) and are not used.
    current = [row for row in pathway_rows if "energy_basis" in row]
    legacy = len(pathway_rows) - len(current)
    if legacy:
        print(f"[co2-analysis] ignored {legacy} pathway row(s) with the old schema (no energy_basis column)")
    pathway_map = {(row["material"], row["site"]): row for row in current}
    columns = pathway_columns(current)
    table_rows = build_pathway_template(rows, columns)

    for table_row in table_rows:
        key = (str(table_row["material"]), str(table_row["site"]))
        pathway = pathway_map.get(key)
        if not pathway:
            continue

        table_row["pathway_status"] = str(pathway.get("pathway_status", "")).strip()
        table_row["energy_basis"] = str(pathway.get("energy_basis", "")).strip()
        if table_row["pathway_status"] != "ok":
            message = str(pathway.get("error_message", "")).strip()
            if message:
                table_row["notes"] = f"pathway_{table_row['pathway_status']}: {message}"
            continue

        for column in columns:
            if column in ("pathway_status", "energy_basis"):
                continue
            table_row[column] = numeric_or_text(pathway.get(column))
        note_chunks = [
            str(pathway.get("notes", "")).strip(),
            str(pathway.get("geometry_flags", "")).strip(),
        ]
        table_row["notes"] = "; ".join(chunk for chunk in note_chunks if chunk)

    return table_rows


def write_manifest(path: Path) -> None:
    lines = [
        "# CO2 Reduction Figure Manifest",
        "",
        "## Figure 1: Adsorption-Energy Heatmap",
        "- Input: `figure_adsorption_heatmap.csv`",
        "- Axes: x=`site`, y=`material`, color=`adsorption_energy_ev`",
        "- Goal: Compare thermodynamic favorability across oxide/site combinations.",
        "",
        "## Figure 2: CO2 Activation Scatter",
        "- Input: `figure_activation_scatter.csv`",
        "- Axes: x=`co_bond_avg_ang`, y=`oco_angle_deg`, color=`reduction_proxy_score`",
        "- Goal: Visualize bond stretching and bending that indicate molecular activation.",
        "",
        "## Figure 3: Photo-Thermo Map",
        "- Input: `figure_photo_thermo_map.csv`",
        "- Axes: x=`adsorption_energy_ev`, y=`tddft_onset_ev`, color=`material`, marker=`site`",
        "- Goal: Joint view of adsorption and optical response suitability with site-level markers.",
        "",
        "## Figure 4: Site Ranking Bar Chart",
        "- Input: `figure_site_ranking.csv`",
        "- Axes: x=`material_site`, y=`reduction_proxy_score`",
        "- Goal: Show the ordered shortlist for follow-up mechanistic studies.",
        "",
        "## Figure 5: Gaussian-Broadened TDDFT Spectra",
        "- Input: `results/tddft/<material>/<site>/transitions.csv` for the best-ranked site in each oxide.",
        "- Axes: x=`energy_ev`, y=`gaussian_broadened_intensity` (normalized)",
        "- Goal: Compare oxide-resolved optical fingerprints using consistent broadening (σ=0.10 eV).",
        "",
        "## Primary Table Schema",
        "- File: `co2_reduction_table.csv`",
        "- Required columns: material, site, adsorption_energy_ev, co_bond_avg_ang, oco_angle_deg, tddft_onset_ev, tddft_peak_energy_ev, reduction_proxy_score, quality_flag",
        "- Include units in paper table caption: eV and Angstrom.",
        "",
        "## Mechanistic Table Template",
        "- File: `co2rr_pathway_template.csv`",
        "- Auto-filled from `co2rr_pathway_summary.csv` when pathway simulations exist.",
        "- Pathway: * + CO2(g) -> COOH* -> CO* + H2O(g) -> * + CO(g), computational hydrogen electrode.",
        "- `energy_basis` = `free_energy_CHE`: step columns are `dG_*` (ZPE, heat capacity, -TS; slab frozen).",
        "- `energy_basis` = `electronic_energies_only`: step columns are `dE_*`; these are not free energies.",
        "- `limiting_potential_v` = -max(PCET steps at U = 0)/e in V vs RHE; CO desorption is a separate chemical step.",
        "- `*_CO2_adsorption_ev` (* + CO2(g) -> CO2*) is a reference step, not part of the profile.",
        "- `*_O_formation_from_H2O_ev`, if present, is a separate quantity and not part of the CO2 -> CO profile.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    analysis_dir = args.output_dir.expanduser().resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    dft_path = results_dir / "dft_summary.csv"
    tddft_path = results_dir / "tddft_summary.csv"
    pathway_path = results_dir / "co2rr_pathway_summary.csv"

    dft_rows = read_csv_rows(dft_path)
    tddft_rows = read_csv_rows(tddft_path)
    pathway_rows = read_csv_rows(pathway_path, required=False)
    merged_rows = merge_dft_tddft(dft_rows=dft_rows, tddft_rows=tddft_rows)

    co2_table = analysis_dir / "co2_reduction_table.csv"
    write_csv(co2_table, merged_rows)

    figure_tables = build_figure_tables(merged_rows)
    for filename, rows in figure_tables.items():
        write_csv(analysis_dir / filename, rows)

    pathway_template = analysis_dir / "co2rr_pathway_template.csv"
    write_csv(pathway_template, build_pathway_table(merged_rows, pathway_rows=pathway_rows))

    manifest = analysis_dir / "figure_manifest.md"
    write_manifest(manifest)

    print(f"[co2-analysis] wrote {co2_table}")
    for filename in figure_tables:
        print(f"[co2-analysis] wrote {analysis_dir / filename}")
    print(f"[co2-analysis] wrote {pathway_template}")
    print(f"[co2-analysis] wrote {manifest}")


if __name__ == "__main__":
    main()
