"""Table 2 of the manuscript: one row per adsorption site of one material.

Inputs (paths of one simulation run RUN and one analysis run A):

* ``RUN/dft_summary.csv`` (required): E_ads, final state, O-C-O angle and the shortest
  C-surface distance (``c_min_slab_distance_ang``: minimum-image distance from C to any slab
  atom; not the C height of fig03(b)).
* ``A/science/co2rr/co2rr_corrected_summary.csv``: U_L and limiting step of one variant
  (default c); the validity of a row comes from ``co2rr_corrections.row_validity`` (the same
  function as everywhere else): only valid rows get a U_L, the others show their validity
  ("unbound", "wrong intermediate").
* ``A/science/neb/neb_barriers.csv``: every NEB path that starts or ends at the site.
  A barrier is given only when ``barrier_resolved`` / ``reverse_barrier_resolved`` is
  true; otherwise "endpoint not a minimum", "no saddle" or "< resolution".
* ``A/science/md/md_trajectory_analysis_summary.csv`` (the only MD input; nothing is read
  from RUN): ``fraction_production_adsorbed_at_start_site`` (site assignment within the site
  radius equal to the start site, persistently desorbed frames excluded) and
  ``fraction_production_desorbed_persistent`` of every trajectory with status ``complete`` or
  ``partial``. Printed as the mean with the per-seed range and the number of trajectories,
  e.g. "0.41 (0.08–1.00, n=2)"; partial trajectories are counted and marked
  ("n=3, 1 partial"). Other statuses (overfull, unreadable, error, ...) are left out and
  reported. A summary without the column leaves the MD cells "n/a".
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

from scripts.analysis.co2rr_corrections import row_validity

SITE_ORDER = ("top_metal", "top_oxygen", "bridge")
SITE_DISPLAY = {"top_metal": "Top metal", "top_oxygen": "Top oxygen", "bridge": "Bridge"}
EADS_UNCERTAINTY_EV = 0.05
UL_UNCERTAINTY_V = 0.1
MD_RUN_STATUSES = ("complete", "partial")
MD_START_COLUMN = "fraction_production_adsorbed_at_start_site"
MD_DESORBED_COLUMN = "fraction_production_desorbed_persistent"
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
    "md_n_partial",
    "md_fraction_adsorbed_at_start_site_mean",
    "md_fraction_adsorbed_at_start_site_min",
    "md_fraction_adsorbed_at_start_site_max",
    "md_fraction_desorbed_persistent_mean",
    "md_fraction_desorbed_persistent_min",
    "md_fraction_desorbed_persistent_max",
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


MD_STATS = ("mean", "min", "max")


def md_for(rows: list[dict[str, str]] | None, material: str, site: str) -> dict[str, float | int]:
    """MD statistics of one site from the analysis summary rows.

    Trajectories with status complete or partial and finite values of both columns are
    used (the same n for both means). Keys: n, n_partial, start_<mean|min|max>,
    desorbed_<mean|min|max>.
    """
    result: dict[str, float | int] = {"n": 0, "n_partial": 0}
    result.update({f"{key}_{stat}": float("nan") for key in ("start", "desorbed") for stat in MD_STATS})
    if rows is None:
        return result
    site_rows = [row for row in rows if row.get("material") == material and row.get("site") == site]
    if site_rows and MD_START_COLUMN not in site_rows[0]:
        print(f"[skip] MD columns of {site}: summary has no {MD_START_COLUMN} column "
              "(rerun md_trajectory_analysis)")
        return result
    used, left_out = [], []
    for row in site_rows:
        values = (safe_float(row.get(MD_START_COLUMN)), safe_float(row.get(MD_DESORBED_COLUMN)))
        if row.get("status") in MD_RUN_STATUSES and all(is_finite(v) for v in values):
            used.append((row.get("status"), *values))
        elif row.get("status") not in ("missing", "not_started", "no_production"):
            left_out.append(f"seed {row.get('seed', '')}: {row.get('status', '')}")
    if left_out:
        print(f"[paper] MD {site}: trajectories left out of Table 2 ({'; '.join(left_out)})")
    if not used:
        return result
    result["n"] = len(used)
    result["n_partial"] = sum(1 for status, _, _ in used if status == "partial")
    for index, key in ((1, "start"), (2, "desorbed")):
        values = [entry[index] for entry in used]
        result.update({f"{key}_mean": sum(values) / len(values), f"{key}_min": min(values),
                       f"{key}_max": max(values)})
    return result


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
        validity = row_validity(c_row) if c_row else ""
        md = md_for(md_rows, material, site)
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
            "md_n_trajectories": md["n"],
            "md_n_partial": md["n_partial"],
            **{f"md_fraction_adsorbed_at_start_site_{stat}": md[f"start_{stat}"] for stat in MD_STATS},
            **{f"md_fraction_desorbed_persistent_{stat}": md[f"desorbed_{stat}"] for stat in MD_STATS},
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


def md_text(row: dict[str, object], prefix: str) -> str:
    """'0.41 (0.08–1.00, n=2)', '0.41 (n=1)' or '0.41 (0.08–1.00, n=3, 1 partial)'."""
    n, n_partial = int(row["md_n_trajectories"]), int(row["md_n_partial"])
    mean = float(row[f"{prefix}_mean"])
    if n == 0 or not is_finite(mean):
        return "n/a"
    count = f"n={n}" + (f", {n_partial} partial" if n_partial else "")
    if n == 1:
        return f"{mean:.2f} ({count})"
    return f"{mean:.2f} ({float(row[f'{prefix}_min']):.2f}–{float(row[f'{prefix}_max']):.2f}, {count})"


def display_rows(table: list[dict[str, object]], variant: str) -> tuple[list[str], list[list[str]]]:
    header = [
        "Site",
        "E_ads (eV)",
        "Final state",
        "O-C-O (deg)",
        "Shortest C–surface distance (Å)",
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
            md_text(row, "md_fraction_adsorbed_at_start_site"),
            md_text(row, "md_fraction_desorbed_persistent"),
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
        "Shortest C–surface distance: minimum distance from C to any slab atom (not the C height above the "
        "topmost surface atom of Fig. 3b).",
        "MD: mean over the trajectories with the per-seed range and n (partial = trajectory not finished) of "
        "the production fractions adsorbed at the start site (site assignment within the site radius, "
        "persistently desorbed frames excluded) and persistently desorbed (shortest CO2–surface distance "
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
    out = out.replace("->", r"$\rightarrow$").replace("< ", r"$<$ ")
    return out.replace("–", "--").replace("Å", r"\AA{}")


def write_latex(path: Path, body: list[list[str]], notes: list[str], variant: str) -> None:
    header = [
        "Site", r"$E_\mathrm{ads}$ (eV)", "Final state", r"O--C--O ($^\circ$)",
        r"Shortest C--surface distance (\AA)",
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
