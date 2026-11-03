"""CO2-reduction analysis tables (dft_summary.csv, tddft_summary.csv, CO2RR pathway).

Writes co2_reduction_table.csv, the figure tables and co2rr_pathway_template.csv.
Sites are ranked only by the CHE limiting potential U_L of the CO2 -> CO pathway
(co2rr_corrected_summary.csv variant c by default, read from --corrected-summary or
<results-dir>/co2rr_corrected_summary.csv, else co2rr_pathway_summary.csv);
sites without a valid pathway (pathway_validity unbound or wrong_intermediate) are
listed but not ranked. No composite score is computed.

The C-O stretch of adsorbed CO2 is measured from the relaxed gas-phase CO2 of the same
run (dft/_reference/<profile>/co2_ref.<profile>_optimized.cif, else the pathway
reference pathways/_references/<profile>/co2/co2_optimized.xyz), or from the explicit
--co2-gas-bond-ang; without either it is left empty (never an experimental default).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

from scripts.analysis.co2rr_corrections import (
    CORRECTED_CSV,
    DEFAULT_VARIANT,
    INVALID_STATUSES,
    UNBOUND_STATUS,
    VARIANTS,
    WRONG_INTERMEDIATE_STATUS,
    load_corrected_pathway_rows,
    row_validity,
)
from scripts.common.project_config import EXPLORATORY_SITES, RESULTS_DIR


ANALYSIS_DIR = RESULTS_DIR / "analysis"
# Numerical uncertainty of E_ads (eV) quoted with positive adsorption energies.
EADS_UNCERTAINTY_EV = 0.05
EADS_POSITIVE_WARNING = "not bound at PBE level (E_ads > 0)"
EADS_UNCERTAINTY_WARNING = f"|E_ads| below the ~{EADS_UNCERTAINTY_EV:.2f} eV numerical uncertainty"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CO2-reduction analysis and figure-ready tables."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument(
        "--co2rr-variant",
        choices=sorted(VARIANTS),
        default=DEFAULT_VARIANT,
        help=f"Variant of {CORRECTED_CSV} used when that file exists (a: as computed, "
             "b: quasi-harmonic floor, c: floor + PBE gas corrections; default c).",
    )
    parser.add_argument(
        "--uncorrected",
        action="store_true",
        help=f"Use co2rr_pathway_summary.csv even if {CORRECTED_CSV} exists.",
    )
    parser.add_argument(
        "--corrected-summary",
        type=Path,
        default=None,
        help=f"Explicit {CORRECTED_CSV} (e.g. A/science/co2rr/{CORRECTED_CSV} of an analysis run). "
             f"Default: <results-dir>/{CORRECTED_CSV} when it exists, else co2rr_pathway_summary.csv. "
             "When given, the file must exist (no fallback to the uncorrected summary).",
    )
    parser.add_argument(
        "--co2-gas-bond-ang",
        type=float,
        default=None,
        help="Gas-phase CO2 C-O bond length (Ang) used as the baseline of the C-O stretch. "
             "Default: the relaxed CO2 reference of the same run (PBE), per DFT profile.",
    )
    return parser.parse_args(argv)


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
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        raise SystemExit("No rows available to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def eads_warning(adsorption_energy: float) -> str:
    """Warning text for E_ads > 0 (the geometric final_state is reported unchanged)."""
    if not math.isfinite(adsorption_energy) or adsorption_energy <= 0.0:
        return ""
    if adsorption_energy < EADS_UNCERTAINTY_EV:
        return f"{EADS_POSITIVE_WARNING}; {EADS_UNCERTAINTY_WARNING}"
    return EADS_POSITIVE_WARNING


def tddft_first_bright(row: dict[str, str]) -> float:
    """First bright excitation of the isolated CO2; older summaries call it co2_isolated_onset_ev."""
    value = row.get("co2_isolated_first_bright_excitation_ev")
    if value in (None, ""):
        value = row.get("co2_isolated_onset_ev")
    return safe_float(value)


def co2_bond_length_from_file(path: Path) -> float:
    """Mean C-O distance (Ang) of a CO2 geometry file (one C, two O)."""
    from ase.io import read

    atoms = read(path)
    carbons = [atom.index for atom in atoms if atom.symbol == "C"]
    oxygens = [atom.index for atom in atoms if atom.symbol == "O"]
    if len(carbons) != 1 or len(oxygens) != 2:
        raise ValueError(f"{path} is not a CO2 molecule ({atoms.get_chemical_formula()})")
    return float(sum(atoms.get_distance(carbons[0], idx, mic=True) for idx in oxygens) / 2.0)


def co2_reference_files(results_dir: Path, profile: str) -> list[Path]:
    """Relaxed gas-phase CO2 of run_dft.py, then of run_co2rr_pathways.py, for one profile."""
    return [
        results_dir / "dft" / "_reference" / profile / f"co2_ref.{profile}_optimized.cif",
        results_dir / "pathways" / "_references" / profile / "co2" / "co2_optimized.xyz",
    ]


def co2_baseline_lookup(results_dir: Path, explicit_ang: float | None = None) -> Callable[[str], tuple[float, str]]:
    """profile -> (gas-phase CO2 C-O length in Ang, its source); NaN when unavailable."""
    cache: dict[str, tuple[float, str]] = {}

    def lookup(profile: str) -> tuple[float, str]:
        if explicit_ang is not None:
            return float(explicit_ang), "--co2-gas-bond-ang"
        profile = str(profile or "")
        if profile not in cache:
            cache[profile] = (float("nan"), "unavailable")
            for path in co2_reference_files(Path(results_dir), profile) if profile else []:
                if not path.exists():
                    continue
                try:
                    cache[profile] = (co2_bond_length_from_file(path), str(path))
                    break
                except Exception as exc:  # unreadable file: try the next one
                    print(f"[co2-analysis] cannot read CO2 reference {path}: {exc}")
            if profile and not math.isfinite(cache[profile][0]):
                print(
                    f"[co2-analysis] WARNING: no relaxed CO2 reference for profile {profile!r} in {results_dir}; "
                    "co_bond_stretch_ang left empty (pass --co2-gas-bond-ang to set the baseline)"
                )
        return cache[profile]

    return lookup


def _no_baseline(profile: str) -> tuple[float, str]:
    return float("nan"), "unavailable"


def merge_dft_tddft(
    dft_rows: list[dict[str, str]],
    tddft_rows: list[dict[str, str]],
    co2_baseline: Callable[[str], tuple[float, str]] | None = None,
) -> list[dict[str, str | float]]:
    """One row per DFT case, in dft_summary.csv order (U_L and rank: attach_limiting_potentials)."""
    baseline_for = co2_baseline or _no_baseline
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

        # TDDFT is the spectrum of the isolated CO2 in its adsorbed geometry, not the
        # optical response of ZnO-CO2, so it is reported but not used for ranking.
        first_bright = tddft_first_bright(tddft_row)
        peak_energy = safe_float(tddft_row.get("co2_isolated_peak_ev"))
        peak_osc = safe_float(tddft_row.get("co2_isolated_peak_f"))
        num_transitions = safe_float(tddft_row.get("co2_isolated_num_transitions"))

        baseline, baseline_source = baseline_for(str(dft_row.get("dft_profile", "")))
        co_bond_avg = (co_bond_1 + co_bond_2) / 2.0 if math.isfinite(co_bond_1) and math.isfinite(co_bond_2) else float("nan")
        co_bond_stretch = (
            co_bond_avg - baseline if math.isfinite(co_bond_avg) and math.isfinite(baseline) else float("nan")
        )
        oco_bend = 180.0 - oco_angle if math.isfinite(oco_angle) else float("nan")

        quality_flag = "ok" if dft_status == "ok" and site not in EXPLORATORY_SITES else "incomplete"

        merged.append(
            {
                "material": material,
                "site": site,
                "dft_status": dft_status,
                "tddft_status": tddft_status,
                "dft_profile": dft_row.get("dft_profile", ""),
                "tddft_mode": tddft_row.get("tddft_mode", ""),
                "adsorption_energy_ev": adsorption_energy,
                "eads_warning": eads_warning(adsorption_energy),
                "final_state": dft_row.get("final_state", ""),
                "co_bond_avg_ang": co_bond_avg,
                "co_bond_baseline_ang": baseline,
                "co_bond_baseline_source": baseline_source,
                "co_bond_stretch_ang": co_bond_stretch,
                "oco_angle_deg": oco_angle,
                "oco_bend_deg": oco_bend,
                "c_surface_distance_ang": c_surface_distance,
                "co2_isolated_first_bright_excitation_ev": first_bright,
                "co2_isolated_peak_ev": peak_energy,
                "co2_isolated_peak_f": peak_osc,
                "co2_isolated_num_transitions": num_transitions,
                "pathway_validity": "",
                "limiting_step": "",
                "limiting_potential_v": float("nan"),
                "rank_by_limiting_potential": "",
                "quality_flag": quality_flag,
            }
        )
    return merged


def attach_limiting_potentials(
    rows: list[dict[str, str | float]],
    pathway_table: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    """Copy U_L and validity from the pathway table and rank by U_L (highest first).

    Only rows with quality_flag ok, pathway_validity ok and a finite U_L are ranked
    (one ranking over all oxides and sites); the others keep an empty rank. Returns
    the ranked rows first, then the rest in their original order.
    """
    by_key = {(str(row["material"]), str(row["site"])): row for row in pathway_table}
    for row in rows:
        pathway = by_key.get((str(row["material"]), str(row["site"])), {})
        row["pathway_validity"] = pathway.get("pathway_validity", "")
        row["limiting_step"] = pathway.get("limiting_step", "")
        value = pathway.get("limiting_potential_v", float("nan"))
        row["limiting_potential_v"] = float(value) if isinstance(value, (int, float)) else safe_float(str(value))
        row["rank_by_limiting_potential"] = ""
    ranked = [
        row for row in rows
        if row.get("quality_flag") == "ok"
        and row.get("pathway_validity") == "ok"
        and math.isfinite(float(row["limiting_potential_v"]))
    ]
    ranked.sort(key=lambda row: float(row["limiting_potential_v"]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank_by_limiting_potential"] = rank
    ranked_ids = {id(row) for row in ranked}
    return ranked + [row for row in rows if id(row) not in ranked_ids]


def build_figure_tables(rows: list[dict[str, str | float]]) -> dict[str, list[dict[str, str | float]]]:
    heatmap_rows = [
        {
            "material": row["material"],
            "site": row["site"],
            "adsorption_energy_ev": row["adsorption_energy_ev"],
            "eads_warning": row["eads_warning"],
            "quality_flag": row["quality_flag"],
        }
        for row in rows
    ]

    activation_rows = [
        {
            "material": row["material"],
            "site": row["site"],
            "co_bond_avg_ang": row["co_bond_avg_ang"],
            "oco_angle_deg": row["oco_angle_deg"],
            "co_bond_baseline_ang": row["co_bond_baseline_ang"],
            "co_bond_stretch_ang": row["co_bond_stretch_ang"],
            "oco_bend_deg": row["oco_bend_deg"],
            "quality_flag": row["quality_flag"],
        }
        for row in rows
    ]

    photo_thermo_rows = [
        {
            "material": row["material"],
            "site": row["site"],
            "adsorption_energy_ev": row["adsorption_energy_ev"],
            "eads_warning": row["eads_warning"],
            "co2_isolated_first_bright_excitation_ev": row["co2_isolated_first_bright_excitation_ev"],
            "co2_isolated_peak_ev": row["co2_isolated_peak_ev"],
            "co2_isolated_peak_f": row["co2_isolated_peak_f"],
            "quality_flag": row["quality_flag"],
        }
        for row in rows
    ]

    rank_rows = [
        {
            "rank": row.get("rank_by_limiting_potential", ""),
            "material": row["material"],
            "site": row["site"],
            "limiting_potential_v": row.get("limiting_potential_v", float("nan")),
            "limiting_step": row.get("limiting_step", ""),
            "pathway_validity": row.get("pathway_validity", ""),
            "quality_flag": row["quality_flag"],
        }
        for row in rows
    ]

    return {
        "figure_adsorption_heatmap.csv": heatmap_rows,
        "figure_activation_scatter.csv": activation_rows,
        "figure_photo_thermo_map.csv": photo_thermo_rows,
        "figure_site_ranking.csv": rank_rows,
    }


PATHWAY_BASE_COLUMNS = (
    "pathway_status", "pathway_validity", "energy_basis", "profile", "limiting_step",
    "limiting_potential_v", "limiting_potential_if_bound_v",
)
PATHWAY_TEXT_COLUMNS = ("pathway_status", "pathway_validity", "energy_basis", "profile", "limiting_step")


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
            template_row[column] = "" if column in PATHWAY_TEXT_COLUMNS else float("nan")
        template_row["notes"] = ""
        template_rows.append(template_row)
    return template_rows


def build_pathway_table(
    rows: list[dict[str, str | float]],
    pathway_rows: list[dict[str, str]],
) -> list[dict[str, str | float]]:
    """Pathway columns per site; sites without a valid pathway get no U_L.

    The validity (co2rr_corrections.row_validity) comes from the pathway_validity column,
    the corrected status or, for summaries without that column (also with --uncorrected),
    the geometry flags.
    """
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
        # Unbound sites and wrong intermediates keep their dG_* values; their U_L is empty.
        if table_row["pathway_status"] not in ("ok", *INVALID_STATUSES):
            message = str(pathway.get("error_message", "")).strip()
            if message:
                table_row["notes"] = f"pathway_{table_row['pathway_status']}: {message}"
            continue

        for column in columns:
            if column in ("pathway_status", "energy_basis"):
                continue
            table_row[column] = numeric_or_text(pathway.get(column))
        validity = row_validity(pathway)
        table_row["pathway_validity"] = validity
        if validity != "ok":
            if not math.isfinite(safe_float(str(table_row.get("limiting_potential_if_bound_v", "")))):
                table_row["limiting_potential_if_bound_v"] = numeric_or_text(pathway.get("limiting_potential_v"))
            table_row["limiting_potential_v"] = float("nan")
            table_row["limiting_step"] = ""
        note_chunks = [
            str(pathway.get("notes", "")).strip(),
            str(pathway.get("geometry_flags", "")).strip(),
        ]
        if validity == UNBOUND_STATUS:
            note_chunks.append("unbound intermediate: excluded from U_L and ranking")
        elif validity == WRONG_INTERMEDIATE_STATUS:
            note_chunks.append("relaxed structure is not COOH*/CO* (wrong intermediate): excluded from U_L and ranking")
        table_row["notes"] = "; ".join(chunk for chunk in note_chunks if chunk)

    return table_rows


def write_manifest(path: Path) -> None:
    lines = [
        "# CO2 Reduction Figure Manifest",
        "",
        "## Figure 1: Adsorption-Energy Heatmap",
        "- Input: `figure_adsorption_heatmap.csv`",
        "- Axes: x=`site`, y=`material`, color=`adsorption_energy_ev`; only rows with `quality_flag` = ok.",
        "- Goal: Compare thermodynamic favorability across oxide/site combinations.",
        f"- `eads_warning`: E_ads > 0 means not bound at PBE level; below {EADS_UNCERTAINTY_EV:.2f} eV it is within",
        "  the numerical uncertainty. The geometric `final_state` of run_dft.py is reported unchanged.",
        "",
        "## Figure 2: CO2 Activation Scatter",
        "- Input: `figure_activation_scatter.csv`",
        "- Panels: C-O stretch `co_bond_stretch_ang` = `co_bond_avg_ang` - `co_bond_baseline_ang`, and O-C-O bend.",
        "- Baseline: relaxed gas-phase CO2 of the same run (`co_bond_baseline_source`), or --co2-gas-bond-ang.",
        "- Goal: Visualize bond stretching and bending that indicate molecular activation.",
        "",
        "## Figure 3: Photo-Thermo Map",
        "- Input: `figure_photo_thermo_map.csv`",
        "- Axes: x=`adsorption_energy_ev`, y=`co2_isolated_first_bright_excitation_ev`, color=`material`, "
        "marker=`site`",
        "- Goal: Adsorption energy next to the first bright LrTDDFT excitation of the isolated CO2 in its",
        "  adsorbed geometry (not the optical response of the ZnO-CO2 system; not compared with any photocatalyst",
        "  target).",
        "",
        "## Figure 4: Site Ranking by Limiting Potential",
        "- Input: `figure_site_ranking.csv`",
        "- Axes: x=`limiting_potential_v` (V vs RHE, CHE; variant in `co2rr_pathway_template.csv`), "
        "y=`material_site`",
        "- Ranked by U_L only (highest first); sites with `pathway_validity` unbound or wrong_intermediate are",
        "  listed without rank.",
        "",
        "## Figure 5: Gaussian-Broadened TDDFT Spectra",
        "- Input: `results/tddft/<material>/<site>/transitions.csv` for every site with `quality_flag` = ok.",
        "- Spectra of the isolated CO2 in its adsorbed geometry (not of the oxide/CO2 system).",
        "- Axes: x = excitation energy (eV), y = Gaussian-broadened oscillator strength (eV^-1; f dimensionless),",
        "  sigma = 0.10 eV.",
        "",
        "## Primary Table Schema",
        "- File: `co2_reduction_table.csv`",
        "- Required columns: material, site, adsorption_energy_ev, eads_warning, co_bond_avg_ang, "
        "co_bond_baseline_ang, co_bond_stretch_ang, oco_angle_deg, co2_isolated_first_bright_excitation_ev, "
        "co2_isolated_peak_ev, pathway_validity, limiting_potential_v, rank_by_limiting_potential, quality_flag",
        "- Include units in paper table caption: eV and Angstrom.",
        "",
        "## Mechanistic Table Template",
        "- File: `co2rr_pathway_template.csv`",
        "- Auto-filled from `co2rr_corrected_summary.csv` (variant in `co2rr_variant`, default c) when it exists,",
        "  else from `co2rr_pathway_summary.csv`; see `co2rr_corrections_note.md` for the corrections.",
        "- `pathway_validity` = `unbound` (COOH* or CO* desorbed) or `wrong_intermediate` (relaxed structure is",
        "  not COOH*/CO*): dG_* listed, `limiting_potential_v` empty (formal value in",
        "  `limiting_potential_if_bound_v`), not ranked.",
        "- Pathway: * + CO2(g) -> COOH* -> CO* + H2O(g) -> * + CO(g), computational hydrogen electrode.",
        "- `energy_basis` = `free_energy_CHE`: step columns are `dG_*` (ZPE, heat capacity, -TS; slab frozen).",
        "- `energy_basis` = `electronic_energies_only`: step columns are `dE_*`; these are not free energies.",
        "- `limiting_potential_v` = -max(PCET steps at U = 0)/e in V vs RHE; CO desorption is a separate chemical step.",
        "- `*_CO2_adsorption_ev` (* + CO2(g) -> CO2*) is a reference step, not part of the profile.",
        "- `*_O_formation_from_H2O_ev`, if present, is a separate quantity and not part of the CO2 -> CO profile.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    analysis_dir = args.output_dir.expanduser().resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    dft_path = results_dir / "dft_summary.csv"
    tddft_path = results_dir / "tddft_summary.csv"
    pathway_path = results_dir / "co2rr_pathway_summary.csv"

    dft_rows = read_csv_rows(dft_path)
    tddft_rows = read_csv_rows(tddft_path)
    if args.corrected_summary is not None and args.uncorrected:
        raise SystemExit("--corrected-summary and --uncorrected are mutually exclusive.")
    if args.corrected_summary is not None:
        corrected_path = args.corrected_summary.expanduser().resolve()
        if not corrected_path.is_file():
            raise SystemExit(f"Missing corrected CO2RR summary (--corrected-summary): {corrected_path}")
    else:
        corrected_path = results_dir / CORRECTED_CSV
    corrected = None if args.uncorrected else load_corrected_pathway_rows(
        corrected_path, args.co2rr_variant
    )
    if corrected is not None:
        print(f"[co2-analysis] pathway values from {corrected_path} (variant {args.co2rr_variant})")
        pathway_rows = corrected
    else:
        pathway_rows = read_csv_rows(pathway_path, required=False)
    merged_rows = merge_dft_tddft(
        dft_rows=dft_rows,
        tddft_rows=tddft_rows,
        co2_baseline=co2_baseline_lookup(results_dir, args.co2_gas_bond_ang),
    )
    pathway_table = build_pathway_table(merged_rows, pathway_rows=pathway_rows)
    merged_rows = attach_limiting_potentials(merged_rows, pathway_table)
    for row in merged_rows:
        if row["eads_warning"]:
            print(
                f"[co2-analysis] {row['material']} {row['site']}: E_ads = "
                f"{float(row['adsorption_energy_ev']):+.3f} eV, {row['eads_warning']}"
            )

    co2_table = analysis_dir / "co2_reduction_table.csv"
    write_csv(co2_table, merged_rows)

    figure_tables = build_figure_tables(merged_rows)
    for filename, rows in figure_tables.items():
        write_csv(analysis_dir / filename, rows)

    pathway_template = analysis_dir / "co2rr_pathway_template.csv"
    write_csv(pathway_template, pathway_table)

    manifest = analysis_dir / "figure_manifest.md"
    write_manifest(manifest)

    print(f"[co2-analysis] wrote {co2_table}")
    for filename in figure_tables:
        print(f"[co2-analysis] wrote {analysis_dir / filename}")
    print(f"[co2-analysis] wrote {pathway_template}")
    print(f"[co2-analysis] wrote {manifest}")


if __name__ == "__main__":
    main()
