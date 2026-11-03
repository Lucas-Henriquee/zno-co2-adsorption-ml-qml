"""Table 2 of the manuscript: one row per adsorption site of one material.

Inputs (paths of one simulation run RUN and one analysis run A):

* ``RUN/dft_summary.csv`` (required): E_ads, final state, O-C-O angle, C-slab distance.
* ``A/science/co2rr/co2rr_corrected_summary.csv``: U_L and limiting step of one variant
  (default c); only rows with ``pathway_validity`` ok get a U_L, the others show their
  validity ("unbound", "wrong intermediate").
* ``A/science/neb/neb_barriers.csv``: every NEB path that starts or ends at the site.
  A barrier is given only when ``barrier_resolved`` / ``reverse_barrier_resolved`` is
  true; otherwise "endpoint not a minimum", "no saddle" or "< resolution".
* ``A/science/md/md_trajectory_analysis_summary.csv``: fraction of production frames at
  the start site and persistent desorbed fraction, mean over the seeds with data (n).
* ``RUN/tddft_summary.csv``: first excitation of the isolated CO2 in the adsorbed geometry.

Missing optional inputs leave their columns empty ("n/a") and print ``[skip]``.
Outputs: ``table02_site_summary.csv`` (unrounded numbers), ``.md`` and ``.tex``
(booktabs, rounded). There is no rank column: differences of ~0.05 eV in E_ads and
~0.1 V in U_L are not resolved.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

SITE_ORDER = ("top_metal", "top_oxygen", "bridge")
SITE_DISPLAY = {"top_metal": "Top metal", "top_oxygen": "Top oxygen", "bridge": "Bridge"}
EADS_UNCERTAINTY_EV = 0.05
UL_UNCERTAINTY_V = 0.1
MD_RUN_STATUSES = ("complete", "partial")
TABLE_NAME = "table02_site_summary"

CSV_COLUMNS = (
    "material",
    "site",
    "dft_status",
    "adsorption_energy_ev",
    "final_state",
    "oco_angle_deg",
    "c_min_slab_distance_ang",
    "co2rr_variant",
    "co2rr_pathway_validity",
    "limiting_potential_v",
    "limiting_step",
    "neb_paths",
    "md_n_trajectories",
    "md_fraction_at_start_site_mean",
    "md_fraction_desorbed_persistent_mean",
    "tddft_first_excitation_ev",
    "tddft_first_bright_excitation_ev",
)


# ---------------------------------------------------------------- shared helpers
def safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        text = str(value).strip()
        return float(text) if text else float("nan")
    except ValueError:
        return float("nan")


def as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "1.0", "yes")


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_optional(path: Path, label: str) -> list[dict[str, str]] | None:
    if not Path(path).exists():
        print(f"[skip] {label}: {path} not found")
        return None
    return read_rows(path)


def require_file(path: Path, label: str) -> Path:
    if not Path(path).exists():
        raise SystemExit(f"[paper] required input missing ({label}): {path}")
    return Path(path)


def ordered_sites(sites) -> list[str]:
    present = list(dict.fromkeys(sites))
    return [s for s in SITE_ORDER if s in present] + sorted(s for s in present if s not in SITE_ORDER)


def dft_rows_for(path: Path, material: str) -> dict[str, dict[str, str]]:
    rows = [row for row in read_rows(path) if row.get("material") == material]
    if not rows:
        raise SystemExit(f"[paper] no {material} rows in {path}")
    return {row["site"]: row for row in rows}


def is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


# ---------------------------------------------------------------- per-source extraction
def co2rr_for(rows: list[dict[str, str]] | None, material: str, variant: str) -> dict[str, dict[str, str]]:
    if rows is None:
        return {}
    selected = [row for row in rows if row.get("material") == material and row.get("variant") == variant]
    if not selected:
        print(f"[skip] CO2RR: no {material} rows of variant {variant!r}")
    return {row["site"]: row for row in selected}


def co2rr_validity(row: dict[str, str]) -> str:
    return str(row.get("pathway_validity") or row.get("status") or "").strip()


def neb_entry(row: dict[str, str], digits: int) -> str:
    """Text of one NEB path: fwd = barrier from its initial site, rev = from its final site."""
    initial, final = row.get("initial_site", ""), row.get("final_site", "")
    resolution = safe_float(row.get("barrier_resolution_eV"))
    res_text = f"< {resolution:.2f} eV" if is_finite(resolution) else "< resolution"
    label = f"{initial} -> {final}"
    if as_bool(row.get("endpoint_not_minimum")):
        return f"{label}: endpoint not a minimum"
    if as_bool(row.get("no_internal_saddle")):
        return f"{label}: no saddle"

    def barrier(key: str, flag: str) -> str:
        value = safe_float(row.get(key))
        return f"{value:.{digits}f} eV" if as_bool(row.get(flag)) and is_finite(value) else res_text

    forward = barrier("forward_barrier_eV", "barrier_resolved")
    reverse = barrier("reverse_barrier_eV", "reverse_barrier_resolved")
    return f"{label}: fwd {forward}, rev {reverse}"


def neb_for(rows: list[dict[str, str]] | None, site: str, digits: int) -> str:
    if rows is None:
        return ""
    entries = [
        neb_entry(row, digits)
        for row in rows
        if site in (row.get("initial_site"), row.get("final_site"))
    ]
    return "; ".join(entries)


OCCUPANCY_SITE_TYPES = ("top_metal", "top_oxygen", "bridge")


def occupancy_fractions(site_assignment, desorbed_persistent, start_site: str) -> dict[str, float]:
    """Production-frame fractions: persistently desorbed, else by lateral site assignment.

    site_assignment is lateral only, so a desorbed CO2 above its start site would count as
    "start site"; desorbed frames therefore form their own category and the other three
    categories are adsorbed frames. The four fractions add up to 1.
    """
    import numpy as np

    assignment = np.asarray([str(value) for value in site_assignment])
    desorbed = np.asarray(desorbed_persistent, dtype=float) == 1
    if assignment.size == 0:
        return {name: float("nan") for name in ("start site", "other sites", "other", "desorbed")}
    adsorbed = ~desorbed
    known = np.isin(assignment, OCCUPANCY_SITE_TYPES)
    return {
        "start site": float((adsorbed & (assignment == start_site)).mean()),
        "other sites": float((adsorbed & known & (assignment != start_site)).mean()),
        "other": float((adsorbed & ~known).mean()),
        "desorbed": float(desorbed.mean()),
    }


def frame_fraction_at_start(results_dir: Path | None, material: str, site: str, seed: str) -> float:
    """Adsorbed-at-start-site fraction of one trajectory from md_frame_analysis.csv (NaN if absent)."""
    if results_dir is None:
        return float("nan")
    path = results_dir / "md" / material / site / "md_frame_analysis.csv"
    if not path.exists():
        return float("nan")
    frames = [
        row for row in read_rows(path)
        if row.get("phase") == "production" and str(row.get("seed")) == str(seed)
    ]
    if not frames:
        return float("nan")
    fractions = occupancy_fractions(
        [row.get("site_assignment", "") for row in frames],
        [safe_float(row.get("desorbed_persistent")) for row in frames],
        site,
    )
    return fractions["start site"]


def md_for(
    rows: list[dict[str, str]] | None, material: str, site: str, results_dir: Path | None = None
) -> tuple[int, float, float]:
    """(n trajectories, mean adsorbed-at-start-site fraction, mean persistent desorbed fraction).

    The start-site fraction excludes persistently desorbed frames (from md_frame_analysis.csv);
    without the frame file it falls back to the summary's lateral fraction.
    """
    if rows is None:
        return 0, float("nan"), float("nan")
    used = [
        row for row in rows
        if row.get("material") == material and row.get("site") == site
        and row.get("status") in MD_RUN_STATUSES
        and is_finite(safe_float(row.get("fraction_production_at_start_site")))
    ]
    if not used:
        return 0, float("nan"), float("nan")
    start = []
    for row in used:
        value = frame_fraction_at_start(results_dir, material, site, row.get("seed", ""))
        start.append(value if is_finite(value) else safe_float(row["fraction_production_at_start_site"]))
    desorbed = [safe_float(row.get("fraction_production_desorbed_persistent")) for row in used]
    desorbed = [value for value in desorbed if is_finite(value)]
    return (
        len(used),
        sum(start) / len(start),
        sum(desorbed) / len(desorbed) if desorbed else float("nan"),
    )


def tddft_for(rows: list[dict[str, str]] | None, material: str) -> dict[str, dict[str, str]]:
    if rows is None:
        return {}
    return {
        row["site"]: row for row in rows
        if row.get("material") == material and row.get("tddft_status") == "ok"
    }


def build_table(
    results_dir: Path, analysis_dir: Path, material: str, variant: str
) -> list[dict[str, object]]:
    dft = dft_rows_for(require_file(results_dir / "dft_summary.csv", "DFT summary"), material)
    co2rr = co2rr_for(
        read_optional(analysis_dir / "science" / "co2rr" / "co2rr_corrected_summary.csv", "CO2RR columns"),
        material, variant,
    )
    neb_rows = read_optional(analysis_dir / "science" / "neb" / "neb_barriers.csv", "NEB column")
    md_rows = read_optional(
        analysis_dir / "science" / "md" / "md_trajectory_analysis_summary.csv", "MD columns"
    )
    tddft = tddft_for(read_optional(results_dir / "tddft_summary.csv", "TDDFT column"), material)

    table = []
    for site in ordered_sites(dft):
        row = dft[site]
        ok = row.get("dft_status") == "ok"
        c_row = co2rr.get(site, {})
        validity = co2rr_validity(c_row) if c_row else ""
        n_md, at_start, desorbed = md_for(md_rows, material, site, results_dir)
        t_row = tddft.get(site, {})
        table.append({
            "material": material,
            "site": site,
            "dft_status": row.get("dft_status", ""),
            "adsorption_energy_ev": safe_float(row.get("adsorption_energy_ev")) if ok else float("nan"),
            "final_state": row.get("final_state", "") if ok else "",
            "oco_angle_deg": safe_float(row.get("oco_angle_deg")) if ok else float("nan"),
            "c_min_slab_distance_ang": safe_float(row.get("c_min_slab_distance_ang")) if ok else float("nan"),
            "co2rr_variant": variant if c_row else "",
            "co2rr_pathway_validity": validity,
            "limiting_potential_v": (
                safe_float(c_row.get("limiting_potential_v")) if validity == "ok" else float("nan")
            ),
            "limiting_step": c_row.get("limiting_step", "") if validity == "ok" else "",
            "neb_paths": neb_for(neb_rows, site, digits=4),
            "_neb_paths_rounded": neb_for(neb_rows, site, digits=3),
            "md_n_trajectories": n_md,
            "md_fraction_at_start_site_mean": at_start,
            "md_fraction_desorbed_persistent_mean": desorbed,
            "tddft_first_excitation_ev": safe_float(t_row.get("co2_isolated_first_excitation_ev")),
            "tddft_first_bright_excitation_ev": safe_float(t_row.get("co2_isolated_first_bright_excitation_ev")),
        })
    return table


# ---------------------------------------------------------------- formatting
def fmt(value: float, digits: int, signed: bool = False) -> str:
    if not is_finite(value):
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def co2rr_text(row: dict[str, object]) -> str:
    validity = str(row["co2rr_pathway_validity"])
    if not validity:
        return "n/a"
    if validity == "ok":
        return fmt(float(row["limiting_potential_v"]), 2, signed=True)
    return validity.replace("_", " ")


def md_text(row: dict[str, object], key: str) -> str:
    n = int(row["md_n_trajectories"])
    if n == 0 or not is_finite(float(row[key])):
        return "n/a"
    return f"{float(row[key]):.2f} (n={n})"


def display_rows(table: list[dict[str, object]], variant: str) -> tuple[list[str], list[list[str]]]:
    header = [
        "Site",
        "E_ads (eV)",
        "Final state",
        "O-C-O (deg)",
        "C-slab (A)",
        f"U_L, variant {variant} (V)",
        "Limiting step",
        "NEB paths",
        "MD adsorbed at start site",
        "MD desorbed (persistent)",
        "TDDFT first exc. (eV)",
    ]
    body = []
    for row in table:
        body.append([
            SITE_DISPLAY.get(str(row["site"]), str(row["site"])),
            fmt(float(row["adsorption_energy_ev"]), 3, signed=True),
            str(row["final_state"]) or "n/a",
            fmt(float(row["oco_angle_deg"]), 1),
            fmt(float(row["c_min_slab_distance_ang"]), 2),
            co2rr_text(row),
            str(row["limiting_step"]) or ("n/a" if row["co2rr_pathway_validity"] == "ok" else "-"),
            str(row["_neb_paths_rounded"]) or "n/a",
            md_text(row, "md_fraction_at_start_site_mean"),
            md_text(row, "md_fraction_desorbed_persistent_mean"),
            fmt(float(row["tddft_first_excitation_ev"]), 3),
        ])
    return header, body


def footnotes(material: str, variant: str) -> list[str]:
    return [
        f"Differences within ~{EADS_UNCERTAINTY_EV:.2f} eV in E_ads and ~{UL_UNCERTAINTY_V:.1f} V in U_L "
        "are not resolved; sites are listed in a fixed geometric order.",
        f"U_L: CHE limiting potential of CO2 -> CO, variant {variant}, only for pathway_validity ok. "
        "NEB: barrier given only when resolved in neb_barriers.csv (fwd = from initial site, rev = from final site); "
        "otherwise below the barrier resolution, no saddle or endpoint not a minimum.",
        "MD: mean over the trajectories (n) of the production fractions adsorbed at the start site (lateral "
        "site assignment, persistently desorbed frames excluded) and persistently desorbed (CO2-slab distance "
        "above the threshold). TDDFT: isolated CO2 in the adsorbed geometry.",
        f"Material: {material}; DFT: GPAW PBE.",
    ]


def write_csv(path: Path, table: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in table:
            writer.writerow({key: ("" if isinstance(row[key], float) and not math.isfinite(row[key]) else row[key])
                             for key in CSV_COLUMNS})


def write_markdown(path: Path, header: list[str], body: list[list[str]], notes: list[str]) -> None:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell.replace("|", "/") for cell in row) + " |" for row in body]
    lines.append("")
    lines += [f"Note: {note}" if i == 0 else note for i, note in enumerate(notes)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                 "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}


def latex_escape(text: str) -> str:
    out = "".join(LATEX_SPECIAL.get(char, char) for char in text)
    return out.replace("->", r"$\rightarrow$").replace("< ", r"$<$ ")


def write_latex(path: Path, body: list[list[str]], notes: list[str], variant: str) -> None:
    header = [
        "Site", r"$E_\mathrm{ads}$ (eV)", "Final state", r"O--C--O ($^\circ$)", r"C--slab (\AA)",
        rf"$U_\mathrm{{L}}$ ({latex_escape(variant)}, V)", "Limiting step", "NEB paths",
        "MD ads. at start", "MD desorbed", r"TDDFT $E_1$ (eV)",
    ]
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * 7 + "p{4.5cm}" + "l" * (len(header) - 8) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in body:
        cells = [latex_escape(cell) for cell in row]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip"]
    lines += [r"{\footnotesize " + latex_escape(note) + r"\par}" for note in notes]
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write table02_site_summary.{csv,md,tex} of one run.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Simulation run directory (RUN).")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Analysis run directory (A).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <analysis-dir>/paper.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--co2rr-variant", default="c", choices=("a", "b", "c"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"[paper] results directory not found: {results_dir}")
    output_dir = (args.output_dir or analysis_dir / "paper").expanduser().resolve()

    table = build_table(results_dir, analysis_dir, args.material, args.co2rr_variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    header, body = display_rows(table, args.co2rr_variant)
    notes = footnotes(args.material, args.co2rr_variant)
    paths = {ext: output_dir / f"{TABLE_NAME}.{ext}" for ext in ("csv", "md", "tex")}
    write_csv(paths["csv"], table)
    write_markdown(paths["md"], header, body, notes)
    write_latex(paths["tex"], body, notes, args.co2rr_variant)
    for path in paths.values():
        print(f"[paper] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
