"""Post-processing corrections of the CO2 -> CO free-energy profile (no new DFT).

Recomputes the CHE free energies of a finished run from the checkpoints written by
run_co2rr_pathways.py (pathways/<material>/<site>/<profile>/<term>/energy.json and
pathways/_references/<profile>/<molecule>/energy.json: electronic energy and the
finite-difference vibrational energies in eV) in three variants:

  a  as computed: imaginary modes removed (what co2rr_pathway_summary.csv holds).
  b  quasi-harmonic floor: for the adsorbates (CO2*, COOH*, CO*, O*) every imaginary mode
     and every real mode below --low-mode-cm (default 50 cm^-1) is set to that value
     before HarmonicThermo; the gases are treated as in (a) (IdealGasThermo, which keeps
     the 3N-5/3N-6 highest modes and drops imaginary ones).
  c  (b) + semi-empirical gas-phase corrections for PBE of Granda-Marulanda et al.,
     ACS Catal. 10, 6900 (2020): G(CO2) + 0.19 eV and G(CO) - 0.24 eV.

Sign convention of (c): the corrections are the negative of the PBE gas-phase errors,
G_corr = G_PBE - epsilon with epsilon(CO2) = -0.19 eV and epsilon(CO) = +0.24 eV, so
--gas-correction CO2=0.19 CO=-0.24 is added to G of the gas. Their effect on the overall
CO2 + 2(H+ + e-) -> CO + H2O(g) is -0.24 - 0.19 = -0.43 eV; the experimental value is
+0.20 eV at 298 K.

The CHE arithmetic, stoichiometry, HarmonicThermo/IdealGasThermo handling and U_L are
those of run_co2rr_pathways.py (species_energies, che_reaction_energies,
limiting_potential), so variant (a) reproduces the summary of the run. Sites whose
geometry_flags contain "_desorbed" (COOH or CO not bound to the slab after relaxation)
get status "unbound": their steps are still listed but they have no U_L and no rank.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path

from scripts.common import project_config
from scripts.simulations.co2rr.run_co2rr_pathways import (
    DEFAULT_THERMO_SETTINGS,
    GAS_SPECIES,
    PATHWAY_STEPS,
    SPECIES_COLUMN,
    TermResult,
    che_reaction_energies,
    limiting_potential,
    molecule_atoms,
    safe_float,
    species_energies,
)

SUMMARY_CSV = "co2rr_pathway_summary.csv"
CORRECTED_CSV = "co2rr_corrected_summary.csv"
NOTE_MD = "co2rr_corrections_note.md"

VARIANTS = {
    "a": "as computed (imaginary modes removed)",
    "b": "quasi-harmonic floor on adsorbate modes",
    "c": "quasi-harmonic floor + PBE gas-phase corrections (Granda-Marulanda 2020)",
}
DEFAULT_VARIANT = "c"
DEFAULT_LOW_MODE_CM = 50.0
DEFAULT_GAS_CORRECTIONS = {"CO2": 0.19, "CO": -0.24}
EXPERIMENTAL_OVERALL_EV = 0.20
OVERALL_TOL_EV = 1.0e-6

# Directory of each species below pathways/ (see run_co2rr_pathways.py).
ADSORBATE_TERMS = {"CO2*": "co2_star", "COOH*": "cooh", "CO*": "co", "O*": "o"}
STEP_COLUMNS = ("CO2_adsorption", *PATHWAY_STEPS, "overall")


def parse_gas_corrections(items: list[str]) -> dict[str, float]:
    corrections: dict[str, float] = {}
    for item in items:
        name, sep, value = str(item).partition("=")
        if not sep or name not in ("CO2", "CO"):
            raise SystemExit(f"--gas-correction expects CO2=<eV> or CO=<eV>, got {item!r}")
        corrections[name] = float(value)
    return {"CO2": corrections.get("CO2", 0.0), "CO": corrections.get("CO", 0.0)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the CO2 -> CO CHE free energies of a finished run in three variants "
        "(as computed, quasi-harmonic floor, floor + PBE gas corrections)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_config.RESULTS_DIR,
        help="Run directory holding co2rr_pathway_summary.csv and pathways/ (default: QML_ZNO_RESULTS_DIR).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: --results-dir.")
    parser.add_argument(
        "--low-mode-cm",
        type=float,
        default=DEFAULT_LOW_MODE_CM,
        help="Quasi-harmonic floor (cm^-1) for adsorbate modes in variants b and c.",
    )
    parser.add_argument(
        "--gas-correction",
        nargs="*",
        default=[f"{name}={value:g}" for name, value in DEFAULT_GAS_CORRECTIONS.items()],
        metavar="NAME=EV",
        help="Additive corrections to G of the gases in variant c (default: CO2=0.19 CO=-0.24).",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- inputs


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_energy_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not math.isfinite(safe_float(payload.get("energy_ev"))):
        raise ValueError(f"no finite energy_ev in {path}")
    return payload


def decode_vibrations(payload: dict, path: Path) -> list[complex]:
    vibrations = payload.get("vibrations")
    if not isinstance(vibrations, dict) or "energies_ev" not in vibrations:
        raise ValueError(f"no vibrations in {path}")
    return [complex(float(real), float(imag)) for real, imag in vibrations["energies_ev"]]


def gas_atoms(folder: Path, name: str):
    """Relaxed molecule (moments of inertia for IdealGasThermo), else the built-in start."""
    from ase.io import read

    path = folder / f"{name.lower()}_optimized.xyz"
    if path.exists():
        return read(path)
    print(f"[co2rr-corrections] {path} not found; using the built-in {name} geometry for the rotations")
    return molecule_atoms(name)


def load_terms(pathways_dir: Path, material: str, site: str, profile: str) -> dict[str, TermResult]:
    """TermResults of one site rebuilt from its energy.json checkpoints."""
    terms: dict[str, TermResult] = {}

    def term(species: str, folder: Path, vibrations: bool, atoms=None) -> TermResult:
        path = folder / "energy.json"
        payload = read_energy_json(path)
        return TermResult(
            term=species,
            profile=profile,
            energy_ev=float(payload["energy_ev"]),
            atoms=atoms,
            metadata=dict(payload.get("metadata") or {}),
            source=str(payload.get("source", "")),
            vib_energies=decode_vibrations(payload, path) if vibrations else None,
        )

    terms["*"] = term("*", pathways_dir / material / "_clean" / profile, vibrations=False)
    for species, folder_name in ADSORBATE_TERMS.items():
        folder = pathways_dir / material / site / profile / folder_name
        if species == "O*" and not (folder / "energy.json").exists():
            continue  # O* is optional and never part of the profile
        terms[species] = term(species, folder, vibrations=True)
    for name in GAS_SPECIES:
        folder = pathways_dir / "_references" / profile / name.lower()
        terms[name] = term(name, folder, vibrations=True, atoms=gas_atoms(folder, name))
    return terms


# --------------------------------------------------------------------------- corrections


def floor_vib_energies(energies, floor_ev: float) -> tuple[list[complex], int]:
    """Imaginary modes and real modes below floor_ev set to floor_ev; returns (modes, count)."""
    floored: list[complex] = []
    count = 0
    for value in energies:
        value = complex(value)
        if abs(value.imag) > 0.0 or value.real < floor_ev:
            floored.append(complex(floor_ev, 0.0))
            count += 1
        else:
            floored.append(value)
    return floored, count


def floored_terms(terms: dict[str, TermResult], floor_ev: float) -> tuple[dict[str, TermResult], int]:
    """Copy of the terms with the adsorbate modes floored (gases and clean slab unchanged)."""
    out: dict[str, TermResult] = {}
    total = 0
    for species, term in terms.items():
        if species in ADSORBATE_TERMS and term.vib_energies is not None:
            modes, count = floor_vib_energies(term.vib_energies, floor_ev)
            total += count
            term = dataclasses.replace(term, vib_energies=modes)
        out[species] = term
    return out, total


def variant_energies(
    terms: dict[str, TermResult],
    variant: str,
    *,
    temperature: float,
    pressures_bar: dict[str, float],
    run_gas_corrections: dict[str, float],
    gas_corrections: dict[str, float],
    floor_ev: float,
) -> tuple[dict[str, float], int, int]:
    """(G per species, modes floored, imaginary modes removed) for one variant.

    Variants a and b keep the gas corrections the run was made with (zero in the
    production YAML); variant c replaces them by ``gas_corrections``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    n_floored = 0
    if variant in ("b", "c"):
        terms, n_floored = floored_terms(terms, floor_ev)
    energies, _, imaginary = species_energies(
        terms,
        vibrations=True,
        temperature=temperature,
        pressures_bar=pressures_bar,
        gas_corrections=gas_corrections if variant == "c" else run_gas_corrections,
    )
    n_imag = sum(count for species, count in imaginary.items() if species in ADSORBATE_TERMS)
    return energies, n_floored, n_imag


def is_unbound(geometry_flags: str) -> bool:
    return any(flag.strip().endswith("_desorbed") for flag in str(geometry_flags or "").split(";"))


def row_settings(row: dict[str, str]) -> tuple[float, dict[str, float], dict[str, float]]:
    """(T, partial pressures, run gas corrections) of a summary row, defaults when blank."""
    temperature = safe_float(row.get("temperature_k"))
    if not math.isfinite(temperature):
        temperature = float(DEFAULT_THERMO_SETTINGS["temperature_kelvin"])
    defaults = dict(DEFAULT_THERMO_SETTINGS["partial_pressures_bar"])
    pressures = {}
    for name in GAS_SPECIES:
        value = safe_float(row.get(f"p_{name}_bar"))
        pressures[name] = value if math.isfinite(value) else float(defaults[name])
    run_corrections = {}
    for name in ("CO2", "CO"):
        value = safe_float(row.get(f"gas_correction_{name}_ev"))
        run_corrections[name] = value if math.isfinite(value) else 0.0
    return temperature, pressures, run_corrections


def base_row(row: dict[str, str], variant: str) -> dict[str, float | str]:
    return {
        "material": row.get("material", ""),
        "site": row.get("site", ""),
        "variant": variant,
        "variant_description": VARIANTS[variant],
        "status": "",
        "profile": row.get("profile", ""),
        "geometry_flags": row.get("geometry_flags", ""),
    }


def correct_site(
    row: dict[str, str],
    pathways_dir: Path,
    *,
    floor_cm: float,
    gas_corrections: dict[str, float],
) -> list[dict[str, float | str]]:
    """One output row per variant for one summary row."""
    from ase.units import invcm

    floor_ev = float(floor_cm) * invcm
    material, site, profile = row.get("material", ""), row.get("site", ""), row.get("profile", "")
    out = [base_row(row, variant) for variant in VARIANTS]

    status = str(row.get("pathway_status", "")).strip()
    message = ""
    terms = None
    if status != "ok":
        message = str(row.get("error_message", "")).strip()
    elif row.get("energy_basis") != "free_energy_CHE":
        status, message = "no_vibrations", f"energy_basis={row.get('energy_basis', '')}: no free energies to correct"
    else:
        try:
            terms = load_terms(pathways_dir, material, site, profile)
        except (OSError, ValueError, KeyError) as exc:
            status, message = "missing_checkpoint", str(exc)
    if terms is None:
        for item in out:
            item.update({"status": status or "failed", "message": message})
        return out

    for species, term in terms.items():
        stored = safe_float(row.get(f"E_{SPECIES_COLUMN[species]}_ev"))
        if math.isfinite(stored) and abs(stored - term.energy_ev) > 1.0e-6:
            print(
                f"[co2rr-corrections] {material} {site}: E({species}) in energy.json differs from the summary "
                f"by {term.energy_ev - stored:+.6f} eV; energy.json is used"
            )

    temperature, pressures, run_corrections = row_settings(row)
    status = "unbound" if is_unbound(row.get("geometry_flags", "")) else "ok"
    for item in out:
        variant = str(item["variant"])
        energies, n_floored, n_imag = variant_energies(
            terms, variant, temperature=temperature, pressures_bar=pressures,
            run_gas_corrections=run_corrections, gas_corrections=gas_corrections, floor_ev=floor_ev,
        )
        steps = che_reaction_energies(energies, 0.0)
        u_l, limiting_step = limiting_potential(steps)
        item["status"] = status
        for step in STEP_COLUMNS:
            item[f"dG_{step}_ev"] = steps[step]
        if "O*" in terms:
            item["dG_O_formation_from_H2O_ev"] = steps["O_formation_from_H2O"]
        # Unbound sites have no meaningful adsorbed intermediate, so no U_L and no rank.
        item["limiting_step"] = limiting_step if status == "ok" else ""
        item["limiting_potential_v"] = u_l if status == "ok" else float("nan")
        item["limiting_potential_if_bound_v"] = u_l
        item["n_modes_floored"] = n_floored
        item["n_imag_modes_removed"] = n_imag
        item["low_mode_floor_cm"] = float(floor_cm) if variant in ("b", "c") else float("nan")
        applied = gas_corrections if variant == "c" else run_corrections
        item["gas_correction_CO2_ev"] = float(applied.get("CO2", 0.0))
        item["gas_correction_CO_ev"] = float(applied.get("CO", 0.0))
        item["temperature_k"] = temperature
        item["message"] = ""
    return out


def add_ranks(rows: list[dict[str, float | str]]) -> None:
    """Rank the ok sites of each material and variant by U_L (highest, least negative, first)."""
    groups: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for row in rows:
        row["rank_by_limiting_potential"] = ""
        value = row.get("limiting_potential_v", float("nan"))
        if row.get("status") == "ok" and isinstance(value, float) and math.isfinite(value):
            groups.setdefault((str(row["material"]), str(row["variant"])), []).append(row)
    for group in groups.values():
        group.sort(key=lambda item: float(item["limiting_potential_v"]), reverse=True)
        for rank, row in enumerate(group, start=1):
            row["rank_by_limiting_potential"] = rank


def check_overall(rows: list[dict[str, float | str]], tol: float = OVERALL_TOL_EV) -> dict[tuple[str, str, str], float]:
    """The overall reaction involves only gases: it must be the same for every site.

    Checked per material, profile and variant (sites recomputed in a fallback profile use
    other gas references). Returns the overall value per group.
    """
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        value = row.get("dG_overall_ev")
        if isinstance(value, float) and math.isfinite(value):
            key = (str(row["material"]), str(row["profile"]), str(row["variant"]))
            groups.setdefault(key, []).append(value)
    overall: dict[tuple[str, str, str], float] = {}
    for key, values in groups.items():
        spread = max(values) - min(values)
        if spread > tol:
            raise AssertionError(f"dG_overall differs between sites for {key}: spread {spread:.2e} eV")
        overall[key] = values[0]
    profiles = {(material, variant) for material, _, variant in overall}
    if len(profiles) < len(overall):
        print("[co2rr-corrections] sites use more than one GPAW profile; overall compared per profile")
    return overall


def compare_variant_a(rows: list[dict[str, float | str]], summary_rows: list[dict[str, str]]) -> float:
    """Largest |dG(a) - dG(summary)| over the PCET/chemical steps (should be ~0)."""
    summary = {(row.get("material", ""), row.get("site", "")): row for row in summary_rows}
    worst = 0.0
    for row in rows:
        if row["variant"] != "a" or row["status"] not in ("ok", "unbound"):
            continue
        original = summary.get((str(row["material"]), str(row["site"])), {})
        for step in STEP_COLUMNS:
            stored = safe_float(original.get(f"dG_{step}_ev"))
            potential = safe_float(original.get("potential_v_rhe"))
            if not math.isfinite(stored):
                continue
            # The summary is written at the configured U (each PCET step gains +eU);
            # variant (a) here is at U = 0.
            n_e = {"CO2_to_COOH": 1, "COOH_to_CO": 1, "overall": 2}.get(step, 0)
            stored -= n_e * (potential if math.isfinite(potential) else 0.0)
            worst = max(worst, abs(float(row[f"dG_{step}_ev"]) - stored))
    return worst


# --------------------------------------------------------------------------- readers


UNBOUND_STATUS = "unbound"


def load_corrected_pathway_rows(path: Path, variant: str = DEFAULT_VARIANT) -> list[dict[str, str]] | None:
    """Rows of one variant of co2rr_corrected_summary.csv in the co2rr_pathway_summary.csv layout.

    Returns None when the file does not exist. ``pathway_status`` is the corrected
    status (ok, unbound, failed, ...); unbound rows keep their dG_* values but have no
    limiting potential. All values are at U = 0 V vs RHE.
    """
    path = Path(path)
    if not path.exists():
        return None
    if variant not in VARIANTS:
        raise SystemExit(f"Unknown CO2RR variant {variant!r}; choose from {', '.join(VARIANTS)}.")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("variant") == variant]
    if not rows:
        raise SystemExit(f"Variant {variant!r} not found in {path}; rerun co2rr_corrections.py.")
    converted = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in ("status", "message", "variant_description")}
        item["pathway_status"] = row.get("status", "")
        item["energy_basis"] = "free_energy_CHE"
        item["co2rr_variant"] = variant
        item["potential_v_rhe"] = "0.0"
        item["error_message"] = row.get("message", "")
        item["notes"] = f"corrected variant {variant}: {VARIANTS[variant]}"
        converted.append(item)
    return converted


# --------------------------------------------------------------------------- outputs


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def note_text(floor_cm: float, gas_corrections: dict[str, float], overall: dict[tuple[str, str, str], float]) -> str:
    lines = [
        "# CO2RR free-energy corrections",
        "",
        f"File: `{CORRECTED_CSV}` (one row per site and variant; written by "
        "`scripts/analysis/co2rr_corrections.py` from the energy.json checkpoints, no new DFT).",
        "",
        "## Method",
        "",
        "Free energies follow the computational hydrogen electrode (Norskov et al., J. Phys. Chem. B 108, "
        "17886 (2004)) for * + CO2(g) -> COOH* -> CO* + H2O(g) -> * + CO(g) (Peterson et al., Energy Environ. "
        "Sci. 3, 1311 (2010)), at U = 0 V vs RHE. Adsorbates: harmonic approximation (ASE HarmonicThermo, "
        "slab frozen); gases: ideal gas, rigid rotor, harmonic oscillator (ASE IdealGasThermo) at the partial "
        "pressures of the run. U_L = -max(dG(CO2 -> COOH*), dG(COOH* -> CO* + H2O))/e.",
        "",
        "- **a** - as computed: imaginary adsorbate modes removed from the partition function.",
        f"- **b** - quasi-harmonic floor: imaginary adsorbate modes and real adsorbate modes below "
        f"{floor_cm:g} cm^-1 set to {floor_cm:g} cm^-1, which bounds their spurious entropy "
        "(the usual quasi-harmonic treatment of low-frequency modes; Ribeiro, Marenich, Cramer, Truhlar, "
        "J. Phys. Chem. B 115, 14556 (2011)). Gases are unchanged.",
        f"- **c** - (b) + semi-empirical gas-phase corrections for PBE (Granda-Marulanda, Rendon-Calle, "
        f"Builes, Illas, Koper, Calle-Vallejo, ACS Catal. 10, 6900 (2020)): "
        f"G(CO2) {gas_corrections['CO2']:+.2f} eV, G(CO) {gas_corrections['CO']:+.2f} eV.",
        "",
        "Sign convention of (c): G_corr = G_PBE - epsilon, with the PBE gas-phase errors "
        f"epsilon(CO2) = {-gas_corrections['CO2']:+.2f} eV and epsilon(CO) = {-gas_corrections['CO']:+.2f} eV. "
        "The overall reaction CO2(g) + 2(H+ + e-) -> CO(g) + H2O(g) changes by "
        f"{gas_corrections['CO'] - gas_corrections['CO2']:+.2f} eV; its experimental value is "
        f"{EXPERIMENTAL_OVERALL_EV:+.2f} eV.",
        "",
        "Sites whose relaxed COOH* or CO* is not bound to the slab (geometry flag `*_desorbed`) have status "
        "`unbound`: their steps are listed but they get no U_L (`limiting_potential_v` empty; the formal value "
        "is in `limiting_potential_if_bound_v`) and no rank.",
        "",
        "## Overall reaction (identical for every site)",
        "",
        "| material | profile | variant | dG_overall (eV) | experiment (eV) |",
        "|---|---|---|---|---|",
    ]
    for (material, profile, variant), value in sorted(overall.items()):
        lines.append(f"| {material} | {profile} | {variant} | {value:+.3f} | {EXPERIMENTAL_OVERALL_EV:+.2f} |")
    lines.append("")
    return "\n".join(lines)


def run_corrections(
    results_dir: Path,
    output_dir: Path | None = None,
    *,
    floor_cm: float = DEFAULT_LOW_MODE_CM,
    gas_corrections: dict[str, float] | None = None,
) -> list[dict[str, float | str]]:
    results_dir = Path(results_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else results_dir
    gas_corrections = dict(DEFAULT_GAS_CORRECTIONS if gas_corrections is None else gas_corrections)
    if floor_cm <= 0.0:
        raise SystemExit("--low-mode-cm must be positive.")

    summary_rows = [row for row in read_csv_rows(results_dir / SUMMARY_CSV) if "energy_basis" in row]
    if not summary_rows:
        raise SystemExit(f"No rows with the current schema (energy_basis) in {results_dir / SUMMARY_CSV}")

    rows: list[dict[str, float | str]] = []
    for row in summary_rows:
        rows.extend(
            correct_site(row, results_dir / "pathways", floor_cm=floor_cm, gas_corrections=gas_corrections)
        )
    add_ranks(rows)
    overall = check_overall(rows)

    deviation = compare_variant_a(rows, summary_rows)
    print(f"[co2rr-corrections] variant a vs {SUMMARY_CSV}: max |difference| = {deviation:.2e} eV")
    if deviation > 1.0e-4:
        print("[co2rr-corrections] WARNING: variant a does not reproduce the run summary")

    for (material, profile, variant), value in sorted(overall.items()):
        extra = f" (experiment {EXPERIMENTAL_OVERALL_EV:+.2f} eV)" if variant == "c" else ""
        print(f"[co2rr-corrections] {material} {profile} variant {variant}: dG_overall = {value:+.3f} eV{extra}")
    for row in rows:
        if row["status"] in ("ok", "unbound"):
            u_l = row["limiting_potential_v"]
            print(
                f"[co2rr-corrections] {row['material']} {row['site']} [{row['variant']}] {row['status']}: "
                + " ".join(f"{step}={float(row[f'dG_{step}_ev']):+.3f}" for step in PATHWAY_STEPS)
                + (f" U_L={u_l:+.3f} V" if isinstance(u_l, float) and math.isfinite(u_l) else " U_L=n/a")
                + f" floored={row['n_modes_floored']}"
            )
        else:
            print(f"[co2rr-corrections] {row['material']} {row['site']} [{row['variant']}] {row['status']}")

    csv_path = output_dir / CORRECTED_CSV
    write_csv(csv_path, rows)
    note_path = output_dir / NOTE_MD
    note_path.write_text(note_text(floor_cm, gas_corrections, overall), encoding="utf-8")
    print(f"[co2rr-corrections] wrote {csv_path}")
    print(f"[co2rr-corrections] wrote {note_path}")
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_corrections(
        args.results_dir,
        args.output_dir,
        floor_cm=float(args.low_mode_cm),
        gas_corrections=parse_gas_corrections(args.gas_correction),
    )


if __name__ == "__main__":
    main()
