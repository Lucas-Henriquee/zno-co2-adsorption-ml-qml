"""CO2 -> CO pathway on oxide slabs with the computational hydrogen electrode (CHE).

Pathway (one consistent sequence, Peterson et al., Energy Environ. Sci. 3, 1311 (2010)):

    *     + CO2(g) + (H+ + e-) -> COOH*              (PCET)
    COOH* + (H+ + e-)          -> CO* + H2O(g)       (PCET)
    CO*                        -> *   + CO(g)        (chemical, no electron)

CHE (Norskov et al., J. Phys. Chem. B 108, 17886 (2004)): G(H+ + e-) = 1/2 G(H2) - eU,
U in V vs RHE. Free energies are G = E + ZPE + int Cp dT - TS: adsorbates from
ase.thermochemistry.HarmonicThermo (adsorbate atoms vibrate, slab frozen), gases from
IdealGasThermo. With vibrations off every column is a dE_* electronic energy.

CO2* adsorption (* + CO2(g) -> CO2*) is reported as a separate reference step; the
pathway starts from * + CO2(g). O* (optional) is reported only as the separate
quantity "O* formation from H2O" and is never part of the CO2 -> CO profile or U_L.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from scripts.common.timer import timer, print0

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required. Install with: pip install numpy") from exc

from scripts.common.project_config import (
    ADSORPTION_SITES,
    DFT_RESULTS_DIR,
    GEOMETRY_DIR,
    MATERIALS,
    RESULTS_DIR,
    ROOT_DIR,
    ensure_directories,
)
from scripts.simulations.dft import run_dft
from scripts.simulations.dft.run_dft import (
    CEO2_EXTRA_PROFILES,
    OPTIMIZATION_PROFILES,
    apply_bottom_layer_constraint,
)


PATHWAY_RESULTS_DIR = RESULTS_DIR / "pathways"
PATHWAY_SUMMARY_CSV = RESULTS_DIR / "co2rr_pathway_summary.csv"
PATHWAY_SETTINGS_JSON = RESULTS_DIR / "co2rr_pathway_settings.json"
DEFAULT_CONFIG = ROOT_DIR / "configs" / "simulations" / "zno_production.yaml"

# Thermochemistry defaults; the YAML section methods.co2rr overrides them.
DEFAULT_THERMO_SETTINGS: dict[str, object] = {
    "vibrations": True,
    "vibration_displacement_angstrom": 0.01,
    "temperature_kelvin": 298.15,
    "electrode_potential_v_rhe": 0.0,
    # H2O at 0.035 bar: vapour in equilibrium with liquid water at 298 K (Peterson 2010).
    "partial_pressures_bar": {"CO2": 1.0, "CO": 1.0, "H2": 1.0, "H2O": 0.035},
    # Additive gas-phase corrections (eV) for the PBE error of CO2 and CO; zero by default.
    "gas_corrections_ev": {"CO2": 0.0, "CO": 0.0},
    "o_star_reference": False,
}
BAR_TO_PA = 1.0e5

# Gas-phase species: ideal-gas rigid-rotor/harmonic-oscillator parameters (closed shell).
GAS_SPECIES: dict[str, dict[str, object]] = {
    "CO2": {"geometry": "linear", "symmetrynumber": 2, "spin": 0},
    "CO": {"geometry": "linear", "symmetrynumber": 1, "spin": 0},
    "H2O": {"geometry": "nonlinear", "symmetrynumber": 2, "spin": 0},
    "H2": {"geometry": "linear", "symmetrynumber": 2, "spin": 0},
}

# Initial magnetic moments (mu_B) of adsorbate atoms; GPAW runs spin-polarised when any
# initial moment is non-zero. Closed-shell species (CO*, CO2*, gases) stay spin-paired.
ADSORBATE_MAGMOMS: dict[str, tuple[float, ...]] = {
    "cooh": (1.0, 0.0, 0.0, 0.0),  # C, O(=), O(H), H
    "co": (0.0, 0.0),
    "o": (2.0,),
    "co2": (0.0, 0.0, 0.0),
}

# Initial height (Ang) of the binding atom above the top-plane site position. These are
# only starting guesses; bonds are checked after relaxation.
INTERMEDIATE_HEIGHTS = {
    "cooh": {"top_metal": 2.00, "top_oxygen": 2.60, "bridge": 2.10},
    "co": {"top_metal": 2.00, "top_oxygen": 2.60, "bridge": 2.10},
    "o": {"top_metal": 1.85, "top_oxygen": 1.50, "bridge": 1.60},
}
DESORBED_C_SLAB_DISTANCE_ANG = 2.5

# Stoichiometry of each step: species -> coefficient (products > 0, reactants < 0).
# "H+e-" is a proton-electron pair, G(H+ + e-) = 1/2 G(H2) - eU (CHE).
SPECIES_COMPOSITION: dict[str, dict[str, int]] = {
    "*": {},
    "CO2*": {"C": 1, "O": 2},
    "COOH*": {"C": 1, "O": 2, "H": 1},
    "CO*": {"C": 1, "O": 1},
    "O*": {"O": 1},
    "CO2": {"C": 1, "O": 2},
    "CO": {"C": 1, "O": 1},
    "H2O": {"H": 2, "O": 1},
    "H2": {"H": 2},
    "H+e-": {"H": 1},
}
STEP_STOICHIOMETRY: dict[str, dict[str, int]] = {
    "CO2_adsorption": {"CO2*": 1, "*": -1, "CO2": -1},
    "CO2_to_COOH": {"COOH*": 1, "*": -1, "CO2": -1, "H+e-": -1},
    "COOH_to_CO": {"CO*": 1, "H2O": 1, "COOH*": -1, "H+e-": -1},
    "CO_desorption": {"*": 1, "CO": 1, "CO*": -1},
    "overall": {"CO": 1, "H2O": 1, "CO2": -1, "H+e-": -2},
    "O_formation_from_H2O": {"O*": 1, "H+e-": 2, "*": -1, "H2O": -1},
}
PCET_STEPS = ("CO2_to_COOH", "COOH_to_CO")
CHEMICAL_STEPS = ("CO_desorption",)
PATHWAY_STEPS = PCET_STEPS + CHEMICAL_STEPS
PATHWAY_STATES = ("* + CO2(g)", "COOH*", "CO* + H2O(g)", "* + CO(g) + H2O(g)")

# Every metadata key must match for a checkpoint to be reused; these compare as floats.
METADATA_FLOAT_KEYS = ("ecut_ev", "fmax_ev_per_ang", "initial_magmom_total", "box_vacuum_ang")


class InconsistentInputs(RuntimeError):
    """Inputs of one site cannot be combined into a single-profile pathway."""


# --------------------------------------------------------------------------- MPI helpers


def mpi_rank() -> int:
    try:
        from gpaw.mpi import world
    except ImportError:
        return 0
    return int(world.rank)


def is_rank0() -> bool:
    return mpi_rank() == 0


def mpi_barrier() -> None:
    try:
        from gpaw.mpi import world
    except ImportError:
        return
    world.barrier()


def broadcast_from_rank0(value):
    """Rank 0's value on every rank (identity without MPI)."""
    from ase.parallel import broadcast, world

    return broadcast(value, root=0, comm=world)


def write_json(path: Path, payload: dict) -> None:
    # Every rank calls this; the barrier keeps the others from reading the file before
    # rank 0 has finished writing it.
    if is_rank0():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    mpi_barrier()


# --------------------------------------------------------------------------- arguments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CO2 -> CO pathway calculations (COOH*, CO*) with the computational hydrogen electrode."
    )
    parser.add_argument("--materials", nargs="*", default=sorted(MATERIALS.keys()))
    parser.add_argument("--sites", nargs="*", default=list(ADSORPTION_SITES))
    parser.add_argument("--fmax", type=float, default=0.05, help="Optimization force threshold (eV/Ang).")
    parser.add_argument("--steps", type=int, default=260, help="Maximum BFGS steps.")
    parser.add_argument("--ecut", type=float, default=500.0, help="Plane-wave cutoff (eV).")
    parser.add_argument(
        "--kpts",
        nargs=3,
        type=int,
        default=(2, 3, 1),
        metavar=("KX", "KY", "KZ"),
        help="Monkhorst-Pack k-point mesh for slab calculations.",
    )
    parser.add_argument(
        "--molecule-box-vacuum",
        type=float,
        default=8.0,
        help="Vacuum padding (Ang) for gas-phase molecule references.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML file whose methods.co2rr section sets the thermochemistry options.",
    )
    parser.add_argument(
        "--vibrations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute vibrations for ZPE/entropy (default from YAML, else on).",
    )
    parser.add_argument("--vib-delta", type=float, default=None, help="Finite-difference displacement (Ang).")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (K).")
    parser.add_argument("--potential", type=float, default=None, help="Electrode potential U (V vs RHE).")
    parser.add_argument(
        "--o-star",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also compute O* and report O* formation from H2O as a separate quantity.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute every term even if a checkpoint with matching parameters exists.",
    )
    return parser.parse_args(argv)


def load_co2rr_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    path = Path(path).expanduser()
    if not path.is_file():
        print0(f"[pathways] config not found, using built-in thermochemistry defaults: {path}")
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read the configuration file.") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    section = payload.get("methods", {}).get("co2rr", {}) if isinstance(payload, dict) else {}
    return dict(section or {})


def resolve_thermo_settings(args: argparse.Namespace, config_section: dict[str, object]) -> dict[str, object]:
    settings: dict[str, object] = {
        key: (dict(value) if isinstance(value, dict) else value) for key, value in DEFAULT_THERMO_SETTINGS.items()
    }
    for key in DEFAULT_THERMO_SETTINGS:
        if key not in config_section:
            continue
        value = config_section[key]
        if isinstance(settings[key], dict):
            merged = dict(settings[key])
            merged.update({str(k): float(v) for k, v in dict(value or {}).items()})
            settings[key] = merged
        else:
            settings[key] = value

    overrides = {
        "vibrations": getattr(args, "vibrations", None),
        "vibration_displacement_angstrom": getattr(args, "vib_delta", None),
        "temperature_kelvin": getattr(args, "temperature", None),
        "electrode_potential_v_rhe": getattr(args, "potential", None),
        "o_star_reference": getattr(args, "o_star", None),
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value

    settings["vibrations"] = bool(settings["vibrations"])
    settings["o_star_reference"] = bool(settings["o_star_reference"])
    for key in ("vibration_displacement_angstrom", "temperature_kelvin", "electrode_potential_v_rhe"):
        settings[key] = float(settings[key])
    missing = [name for name in GAS_SPECIES if name not in settings["partial_pressures_bar"]]
    if missing:
        raise SystemExit(f"partial_pressures_bar is missing: {', '.join(missing)}")
    return settings


def validate_inputs(materials: Iterable[str], sites: Iterable[str]) -> tuple[list[str], list[str]]:
    unknown_materials = [name for name in materials if name not in MATERIALS]
    if unknown_materials:
        raise SystemExit(f"Unknown materials: {', '.join(unknown_materials)}")

    unknown_sites = [name for name in sites if name not in ADSORPTION_SITES]
    if unknown_sites:
        raise SystemExit(f"Unknown sites: {', '.join(unknown_sites)}")

    return list(materials), list(sites)


# --------------------------------------------------------------------------- CSV helpers


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows or not is_rank0():
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, float | str]) -> tuple[str, str]:
    return str(row.get("material", "")), str(row.get("site", ""))


def ordered_rows(rows_by_key: dict[tuple[str, str], dict[str, float | str]]) -> list[dict[str, float | str]]:
    material_order = {name: idx for idx, name in enumerate(MATERIALS.keys())}
    site_order = {name: idx for idx, name in enumerate(ADSORPTION_SITES)}
    rows = list(rows_by_key.values())
    rows.sort(
        key=lambda row: (
            material_order.get(str(row.get("material", "")), 999),
            site_order.get(str(row.get("site", "")), 999),
            str(row.get("material", "")),
            str(row.get("site", "")),
        )
    )
    return rows


# --------------------------------------------------------------------------- profiles


def profile_catalog() -> dict[str, dict[str, object]]:
    return {str(profile["name"]): profile for profile in (OPTIMIZATION_PROFILES + CEO2_EXTRA_PROFILES)}


def profiles_for_case(material: str, preferred_profile: str, catalog: dict[str, dict[str, object]]) -> tuple[dict[str, object], ...]:
    base_profiles = OPTIMIZATION_PROFILES
    if material == "CeO2":
        base_profiles = CEO2_EXTRA_PROFILES + OPTIMIZATION_PROFILES

    ordered_names: list[str] = []
    if preferred_profile in catalog:
        ordered_names.append(preferred_profile)
    for profile in base_profiles:
        name = str(profile["name"])
        if name not in ordered_names:
            ordered_names.append(name)
    return tuple(catalog[name] for name in ordered_names)


def make_calculator(profile: dict[str, object], ecut: float, kpts, txt_file: Path):
    """Calculator with exactly the settings optimize_structure uses for this profile."""
    return run_dft.build_calculator(
        ecut=ecut,
        kpts=tuple(int(v) for v in kpts),
        txt_file=txt_file,
        mode=str(profile.get("mode", "pw")),
        smearing=float(profile["smearing"]),
        maxiter=int(profile["maxiter"]),
        mixer=profile["mixer"],
    )


# --------------------------------------------------------------------------- metadata / checkpoints


def cell_hash(atoms) -> str:
    cell = np.round(np.array(atoms.get_cell(), dtype=float), 4) + 0.0  # +0.0 drops -0.0
    pbc = "".join("1" if flag else "0" for flag in atoms.get_pbc())
    digest = hashlib.sha256((np.array2string(cell, precision=4, floatmode="fixed") + pbc).encode("utf-8"))
    return digest.hexdigest()[:16]


def positions_hash(atoms) -> str:
    positions = np.round(np.array(atoms.get_positions(), dtype=float), 4) + 0.0
    return hashlib.sha256(np.array2string(positions, precision=4, floatmode="fixed").encode("utf-8")).hexdigest()[:10]


def calc_metadata(
    *,
    term: str,
    profile_name: str,
    ecut: float,
    kpts,
    fmax: float,
    atoms,
    n_slab_atoms: int,
    box_vacuum: float | None = None,
) -> dict[str, object]:
    magmoms = atoms.get_initial_magnetic_moments()
    metadata: dict[str, object] = {
        "term": term,
        "profile": str(profile_name),
        "ecut_ev": float(ecut),
        "kpts": [int(v) for v in kpts],
        "fmax_ev_per_ang": float(fmax),
        "n_slab_atoms": int(n_slab_atoms),
        "n_atoms": int(len(atoms)),
        "formula": atoms.get_chemical_formula(),
        "cell_hash": cell_hash(atoms),
        "initial_magmom_total": float(np.sum(np.abs(magmoms))) if len(magmoms) else 0.0,
    }
    if box_vacuum is not None:
        metadata["box_vacuum_ang"] = float(box_vacuum)
    return metadata


def metadata_mismatches(stored: dict[str, object] | None, expected: dict[str, object]) -> list[str]:
    """Keys whose stored value differs from the expected one (missing keys count as different)."""
    stored = stored or {}
    mismatches: list[str] = []
    for key, value in expected.items():
        if key not in stored:
            mismatches.append(key)
            continue
        old = stored[key]
        if key in METADATA_FLOAT_KEYS:
            try:
                if not math.isclose(float(old), float(value), rel_tol=0.0, abs_tol=1.0e-8):
                    mismatches.append(key)
            except (TypeError, ValueError):
                mismatches.append(key)
        elif isinstance(value, list):
            if not isinstance(old, list) or list(old) != list(value):
                mismatches.append(key)
        elif old != value:
            mismatches.append(key)
    return mismatches


def read_checkpoint(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_reusable_checkpoint(path: Path, expected: dict[str, object], label: str) -> dict[str, object] | None:
    """Checkpoint payload if all calculation parameters match, else None (and say why)."""
    payload = read_checkpoint(path)
    if payload is None:
        return None
    mismatches = metadata_mismatches(payload.get("metadata"), expected)
    if mismatches:
        print0(f"[pathways] {label}: checkpoint not reused, parameters differ: {', '.join(mismatches)}")
        return None
    if not math.isfinite(safe_float(payload.get("energy_ev"))):
        return None
    return payload


def _encode_vib_energies(energies) -> list[list[float]]:
    return [[float(np.real(value)), float(np.imag(value))] for value in energies]


def _decode_vib_energies(encoded) -> list[complex]:
    return [complex(real, imag) for real, imag in encoded]


# --------------------------------------------------------------------------- relaxations


@dataclass
class TermResult:
    term: str
    profile: str
    energy_ev: float
    atoms: object
    metadata: dict[str, object]
    source: str
    adsorbate_indices: list[int] = field(default_factory=list)
    magmom_final: float = float("nan")
    n_steps: int | None = None
    fmax_final: float = float("nan")
    vib_energies: list[complex] | None = None


def final_magnetic_moment(atoms) -> float:
    try:
        return float(atoms.get_magnetic_moment())
    except Exception:
        return 0.0 if not np.any(atoms.get_initial_magnetic_moments()) else float("nan")


def compute_term(
    *,
    term: str,
    label: str,
    out_dir: Path,
    atoms,
    adsorbate_indices: list[int],
    n_slab_atoms: int,
    profile: dict[str, object],
    ecut: float,
    kpts,
    fmax: float,
    steps: int,
    overwrite: bool,
    vib_delta: float | None,
    box_vacuum: float | None = None,
    known_energy: float | None = None,
    known_source: str = "",
) -> TermResult:
    """Relax one structure in one profile (or take a known energy), then its vibrations.

    Checkpoints are reused only when every calculation parameter matches. The stored
    energy is always the one returned by the calculator, never a value parsed from a log.
    """
    profile_name = str(profile["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "energy.json"
    xyz_path = out_dir / f"{label}_optimized.xyz"
    metadata = calc_metadata(
        term=term,
        profile_name=profile_name,
        ecut=ecut,
        kpts=kpts,
        fmax=fmax,
        atoms=atoms,
        n_slab_atoms=n_slab_atoms,
        box_vacuum=box_vacuum,
    )
    initial_magmoms = np.array(atoms.get_initial_magnetic_moments(), dtype=float)

    payload: dict[str, object] | None = None
    relaxed = None
    just_relaxed = False
    if not overwrite:
        # Only rank 0 looks at the files and every rank follows its decision: a rank that
        # checked after rank 0 had written energy.json would otherwise reuse it alone and
        # skip a collective call (MPI deadlock).
        decision = None
        if is_rank0():
            decision = (
                load_reusable_checkpoint(checkpoint_path, metadata, f"{label} ({profile_name})"),
                xyz_path.exists(),
            )
        payload, xyz_ready = broadcast_from_rank0(decision)
        if payload is not None and known_energy is not None:
            if not math.isclose(float(payload["energy_ev"]), float(known_energy), abs_tol=1.0e-6):
                print0(f"[pathways] {label}: checkpoint energy differs from the DFT summary; not reused")
                payload = None
        if payload is not None:
            if known_energy is not None:
                relaxed = atoms.copy()
            elif xyz_ready:
                from ase.io import read

                relaxed = read(xyz_path)
                if len(relaxed) != len(atoms) or relaxed.get_chemical_formula() != atoms.get_chemical_formula():
                    print0(f"[pathways] {label}: optimized structure does not match checkpoint; not reused")
                    payload, relaxed = None, None
                else:
                    relaxed.set_cell(atoms.get_cell())
                    relaxed.set_pbc(atoms.get_pbc())
                    relaxed.set_initial_magnetic_moments(initial_magmoms)
            else:
                payload = None
        if payload is not None:
            print0(
                f"[pathways] {label}: reuse checkpoint (E={float(payload['energy_ev']):.6f} eV, "
                f"profile={profile_name})"
            )

    if payload is None:
        if known_energy is not None:
            relaxed = atoms.copy()
            payload = {
                "metadata": metadata,
                "energy_ev": float(known_energy),
                "source": known_source or "known",
                "n_steps": None,
                "fmax_final_ev_per_ang": None,
                "magmom_final": None,
            }
        else:
            trial_in = atoms.copy()
            if term not in GAS_SPECIES:
                apply_bottom_layer_constraint(trial_in, adsorbate_atoms=len(adsorbate_indices))
            with timer(f"Pathways {label} ({profile_name})"):
                energy, _, relaxed, used_profile = run_dft.optimize_structure(
                    atoms=trial_in,
                    output_dir=out_dir,
                    label=label,
                    ecut=ecut,
                    kpts=tuple(int(v) for v in kpts),
                    fmax=fmax,
                    steps=steps,
                    profiles=(profile,),
                )
            just_relaxed = True
            if used_profile != profile_name:
                raise InconsistentInputs(f"{label}: converged in {used_profile}, expected {profile_name}")
            payload = {
                "metadata": metadata,
                "energy_ev": float(energy),
                "source": "optimize",
                "n_steps": relaxed.info.get("n_steps"),
                "fmax_final_ev_per_ang": relaxed.info.get("fmax_final"),
                "magmom_final": final_magnetic_moment(relaxed),
            }
        write_json(checkpoint_path, payload)

    result = TermResult(
        term=term,
        profile=profile_name,
        energy_ev=float(payload["energy_ev"]),
        atoms=relaxed,
        metadata=metadata,
        source=str(payload.get("source", "")),
        adsorbate_indices=list(adsorbate_indices),
        magmom_final=safe_float(payload.get("magmom_final")),
        n_steps=payload.get("n_steps"),
        fmax_final=safe_float(payload.get("fmax_final_ev_per_ang")),
    )

    if vib_delta is not None and adsorbate_indices:
        stored_vib = payload.get("vibrations") if isinstance(payload.get("vibrations"), dict) else None
        if (
            stored_vib is not None
            and math.isclose(float(stored_vib.get("delta_ang", -1.0)), float(vib_delta), abs_tol=1.0e-12)
            and list(stored_vib.get("indices", [])) == list(adsorbate_indices)
        ):
            result.vib_energies = _decode_vib_energies(stored_vib["energies_ev"])
        else:
            result.vib_energies = run_vibrations(
                relaxed,
                adsorbate_indices,
                out_dir=out_dir,
                label=label,
                profile=profile,
                ecut=ecut,
                kpts=kpts,
                delta=vib_delta,
                fresh_calculator=not just_relaxed,
            )
            payload["vibrations"] = {
                "delta_ang": float(vib_delta),
                "indices": list(adsorbate_indices),
                "energies_ev": _encode_vib_energies(result.vib_energies),
            }
            write_json(checkpoint_path, payload)
    return result


def run_vibrations(atoms, indices, *, out_dir: Path, label: str, profile, ecut, kpts, delta: float, fresh_calculator: bool):
    """Finite-difference vibrations of the given atoms only (the rest stays frozen).

    ase.vibrations caches each displacement and writes only on MPI rank 0.
    """
    from ase.vibrations import Vibrations

    if fresh_calculator:
        atoms.calc = make_calculator(profile, ecut, kpts, out_dir / f"{label}.{profile['name']}.vib.txt")
    # The cache name carries the geometry hash, so a new relaxation never reuses stale displacements.
    name = out_dir / f"vib_{profile['name']}_d{delta:g}_{positions_hash(atoms)}"
    vib = Vibrations(atoms, indices=list(indices), name=str(name), delta=float(delta), nfree=2)
    vib.run()
    # Only rank 0 writes the displacement cache, and it may still be writing the last file
    # when the other ranks leave run(). Reading it on every rank raced and made some ranks
    # fail alone (MPI deadlock), so rank 0 reads it and broadcasts the energies.
    mpi_barrier()
    energies = [complex(value) for value in vib.get_energies()] if is_rank0() else None
    return broadcast_from_rank0(energies)


# --------------------------------------------------------------------------- structures


def molecule_atoms(name: str):
    from ase import Atoms

    if name == "H2":
        return Atoms("H2", positions=[(0.0, 0.0, -0.37), (0.0, 0.0, 0.37)])
    if name == "CO":
        return Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.15)])
    if name == "CO2":
        return Atoms("OCO", positions=[(-1.16, 0.0, 0.0), (0.0, 0.0, 0.0), (1.16, 0.0, 0.0)])
    if name == "H2O":
        return Atoms(
            "OH2",
            positions=[
                (0.0, 0.0, 0.0),
                (0.757, 0.586, 0.0),
                (-0.757, 0.586, 0.0),
            ],
        )
    raise ValueError(f"Unsupported reference molecule: {name}")


def surface_axis(slab) -> np.ndarray:
    """Unit in-plane vector along the longer surface cell vector (room for the adsorbate)."""
    cell = np.array(slab.get_cell(), dtype=float)
    a, b = cell[0].copy(), cell[1].copy()
    vec = a if np.linalg.norm(a[:2]) >= np.linalg.norm(b[:2]) else b
    vec[2] = 0.0
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def build_intermediate(clean_slab, site: str, site_position, intermediate: str):
    """Adsorbate on a top-plane site from build_geometries.select_adsorption_sites.

    CO: upright, C down. COOH: C bonded to the surface (C down), planar sp2 carbon with
    C=O and C-OH tilted 60 deg from the surface normal. O*: single O on the site.
    Initial magnetic moments are set from ADSORBATE_MAGMOMS.
    """
    from ase import Atoms

    z_up = np.array([0.0, 0.0, 1.0])
    axis = surface_axis(clean_slab)
    anchor = np.array(site_position, dtype=float)
    binding = anchor + INTERMEDIATE_HEIGHTS[intermediate][site] * z_up

    if intermediate == "co":
        symbols = "CO"
        positions = [binding, binding + 1.15 * z_up]
    elif intermediate == "cooh":
        cos60, sin60 = 0.5, math.sqrt(3.0) / 2.0
        o_double = binding + 1.22 * (cos60 * z_up + sin60 * axis)
        o_hydroxyl = binding + 1.36 * (cos60 * z_up - sin60 * axis)
        hydrogen = o_hydroxyl + 0.98 * z_up
        symbols = "COOH"
        positions = [binding, o_double, o_hydroxyl, hydrogen]
    elif intermediate == "o":
        symbols = "O"
        positions = [binding]
    else:
        raise ValueError(f"Unsupported intermediate: {intermediate}")

    combined = clean_slab.copy()
    combined.set_constraint()
    fragment = Atoms(symbols=symbols, positions=[tuple(float(v) for v in pos) for pos in positions])
    combined.extend(fragment)
    magmoms = np.zeros(len(combined))
    magmoms[len(clean_slab):] = ADSORBATE_MAGMOMS[intermediate]
    combined.set_initial_magnetic_moments(magmoms)
    return combined, len(fragment)


def check_adsorbate_bonds(atoms, n_slab_atoms: int, intermediate: str) -> tuple[list[str], float]:
    """Geometry flags after relaxation and the C (or O) to slab minimum distance (Ang)."""
    slab_indices = list(range(n_slab_atoms))
    first = n_slab_atoms
    flags: list[str] = []

    def dist(i: int, j: int) -> float:
        return float(atoms.get_distance(i, j, mic=True))

    binding_to_slab = float(atoms.get_distances(first, slab_indices, mic=True).min())
    lattice_oxygen = [i for i in slab_indices if atoms[i].symbol == "O"]
    if intermediate in ("co", "cooh") and binding_to_slab > DESORBED_C_SLAB_DISTANCE_ANG:
        flags.append(f"{intermediate}_desorbed")
    if intermediate == "co" and dist(first, first + 1) > 1.5:
        flags.append("co_dissociated")
    if intermediate == "cooh":
        if dist(first, first + 1) > 1.6 or dist(first, first + 2) > 1.6:
            flags.append("cooh_dissociated")
        if dist(first + 2, first + 3) > 1.3:
            flags.append("cooh_deprotonated")
        if lattice_oxygen and float(atoms.get_distances(first, lattice_oxygen, mic=True).min()) < 1.6:
            flags.append("cooh_c_bonded_to_lattice_o")
    if intermediate == "o" and lattice_oxygen:
        if float(atoms.get_distances(first, lattice_oxygen, mic=True).min()) < 1.6:
            flags.append("o_star_peroxo")
    return flags, binding_to_slab


# --------------------------------------------------------------------------- thermochemistry


def adsorbate_free_energy_correction(vib_energies, temperature: float) -> tuple[float, float, int]:
    """(G - E, ZPE, n_imag) of an adsorbate from HarmonicThermo (U - TS, slab frozen)."""
    from ase.thermochemistry import HarmonicThermo

    # The removed imaginary modes are counted in the imaginary_modes column, so ASE's
    # per-call "imag modes removed" warning is silenced.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*imag modes removed")
        thermo = HarmonicThermo(vib_energies=list(vib_energies), potentialenergy=0.0, ignore_imag_modes=True)
        correction = float(thermo.get_helmholtz_energy(temperature, verbose=False))
        zpe = float(thermo.get_ZPE_correction())
    return correction, zpe, int(thermo.n_imag)


def gas_free_energy_correction(name: str, atoms, vib_energies, temperature: float, pressure_pa: float) -> tuple[float, float, int]:
    """(G - E, ZPE, n_imag) of a gas from IdealGasThermo (H - TS at the given pressure)."""
    from ase.thermochemistry import IdealGasThermo

    spec = GAS_SPECIES[name]
    molecule = atoms.copy()
    molecule.pbc = False  # only the moments of inertia and the mass are used
    molecule.calc = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*imag modes removed")
        thermo = IdealGasThermo(
            vib_energies=list(vib_energies),
            geometry=str(spec["geometry"]),
            potentialenergy=0.0,
            atoms=molecule,
            symmetrynumber=int(spec["symmetrynumber"]),
            spin=float(spec["spin"]),
            ignore_imag_modes=True,
        )
        correction = float(thermo.get_gibbs_energy(temperature=temperature, pressure=pressure_pa, verbose=False))
        zpe = float(thermo.get_ZPE_correction())
    return correction, zpe, int(getattr(thermo, "n_imag", 0))


def che_reaction_energies(
    energies: dict[str, float],
    potential_v_rhe: float = 0.0,
    steps: Iterable[str] | None = None,
) -> dict[str, float]:
    """Reaction (free) energies of each step with G(H+ + e-) = 1/2 G(H2) - eU.

    `energies` maps species names of SPECIES_COMPOSITION (without "H+e-") to G or E.
    A step whose species are missing is NaN.
    """
    values = dict(energies)
    if "H2" in values:
        values["H+e-"] = 0.5 * float(values["H2"]) - float(potential_v_rhe)
    results: dict[str, float] = {}
    for step in steps if steps is not None else STEP_STOICHIOMETRY:
        total = 0.0
        for species, coefficient in STEP_STOICHIOMETRY[step].items():
            value = values.get(species)
            if value is None or not math.isfinite(float(value)):
                total = float("nan")
                break
            total += coefficient * float(value)
        results[step] = total
    return results


def electrons_transferred(step: str) -> int:
    return -int(STEP_STOICHIOMETRY[step].get("H+e-", 0))


def limiting_potential(step_energies_at_zero: dict[str, float]) -> tuple[float, str]:
    """U_L = -max(dG of the PCET steps at U = 0)/e, in V vs RHE (one electron per step)."""
    candidates = [(step_energies_at_zero[step], step) for step in PCET_STEPS if math.isfinite(step_energies_at_zero.get(step, float("nan")))]
    if len(candidates) != len(PCET_STEPS):
        return float("nan"), ""
    value, step = max(candidates)
    return -float(value), step


def energy_prefix(vibrations: bool) -> str:
    return "dG" if vibrations else "dE"


def energy_basis(vibrations: bool) -> str:
    return "free_energy_CHE" if vibrations else "electronic_energies_only"


def pathway_profile(step_values: dict[str, float]) -> list[float]:
    """Cumulative energies of PATHWAY_STATES relative to * + CO2(g)."""
    profile = [0.0]
    for step in PATHWAY_STEPS:
        profile.append(profile[-1] + float(step_values[step]))
    return profile


def species_energies(
    terms: dict[str, TermResult],
    *,
    vibrations: bool,
    temperature: float,
    pressures_bar: dict[str, float],
    gas_corrections: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """G (or E) of every species, the thermal corrections G - E and imaginary-mode counts.

    The clean slab is frozen in the vibrations, so G(*) = E(*).
    """
    energies: dict[str, float] = {}
    corrections: dict[str, float] = {}
    imaginary: dict[str, int] = {}
    for species, term in terms.items():
        correction = 0.0
        if vibrations and species != "*":
            if term.vib_energies is None:
                raise RuntimeError(f"Missing vibrations for {species}")
            if species in GAS_SPECIES:
                correction, _, n_imag = gas_free_energy_correction(
                    species, term.atoms, term.vib_energies, temperature, float(pressures_bar[species]) * BAR_TO_PA
                )
                correction += float(gas_corrections.get(species, 0.0))
            else:
                correction, _, n_imag = adsorbate_free_energy_correction(term.vib_energies, temperature)
            imaginary[species] = n_imag
        energies[species] = term.energy_ev + correction
        corrections[species] = correction
    return energies, corrections, imaginary


SPECIES_COLUMN = {
    "*": "clean_slab",
    "CO2*": "CO2_star",
    "COOH*": "COOH_star",
    "CO*": "CO_star",
    "O*": "O_star",
    "CO2": "CO2_gas",
    "CO": "CO_gas",
    "H2O": "H2O_gas",
    "H2": "H2_gas",
}


def summary_step_columns(vibrations: bool, o_star: bool) -> list[str]:
    prefix = energy_prefix(vibrations)
    steps = ["CO2_adsorption", *PATHWAY_STEPS, "overall"]
    if o_star:
        steps.append("O_formation_from_H2O")
    return [f"{prefix}_{step}_ev" for step in steps]


def build_result_row(
    *,
    material: str,
    site: str,
    terms: dict[str, TermResult],
    settings: dict[str, object],
    args: argparse.Namespace,
    flags: list[str],
    distances: dict[str, float],
) -> dict[str, float | str]:
    vibrations = bool(settings["vibrations"])
    potential = float(settings["electrode_potential_v_rhe"])
    gas_corrections = dict(settings["gas_corrections_ev"]) if vibrations else {}
    energies, corrections, imaginary = species_energies(
        terms,
        vibrations=vibrations,
        temperature=float(settings["temperature_kelvin"]),
        pressures_bar=dict(settings["partial_pressures_bar"]),
        gas_corrections=gas_corrections,
    )
    at_u = che_reaction_energies(energies, potential)
    at_zero = che_reaction_energies(energies, 0.0)
    u_limiting, limiting_step = limiting_potential(at_zero)
    prefix = energy_prefix(vibrations)
    clean = terms["*"]
    profile = clean.profile

    row: dict[str, float | str] = {
        "material": material,
        "site": site,
        "pathway_status": "ok",
        "energy_basis": energy_basis(vibrations),
        "error_message": "",
        "geometry_flags": ";".join(flags),
        "profile": profile,
        "ecut_ev": float(args.ecut),
        "kpts": "x".join(str(int(v)) for v in args.kpts),
        "fmax_ev_per_ang": float(args.fmax),
        "n_slab_atoms": int(clean.metadata["n_slab_atoms"]),
        "cell_hash": str(clean.metadata["cell_hash"]),
        "vibrations": "on" if vibrations else "off",
        "vib_delta_ang": float(settings["vibration_displacement_angstrom"]) if vibrations else float("nan"),
        "temperature_k": float(settings["temperature_kelvin"]) if vibrations else float("nan"),
        "potential_v_rhe": potential,
    }
    for name in GAS_SPECIES:
        row[f"p_{name}_bar"] = float(settings["partial_pressures_bar"][name]) if vibrations else float("nan")
    for name in ("CO2", "CO"):
        row[f"gas_correction_{name}_ev"] = float(gas_corrections.get(name, 0.0)) if vibrations else 0.0
    for species, term in terms.items():
        row[f"E_{SPECIES_COLUMN[species]}_ev"] = term.energy_ev
    if vibrations:
        for species in terms:
            if species != "*":
                row[f"Gcorr_{SPECIES_COLUMN[species]}_ev"] = corrections[species]
        row["imaginary_modes"] = ";".join(f"{name}:{count}" for name, count in imaginary.items() if count)
        if row["imaginary_modes"]:
            print0(f"[pathways] imaginary modes removed from the thermochemistry: {row['imaginary_modes']}")
    for species in ("COOH*", "CO*", "O*"):
        if species in terms:
            row[f"magmom_{SPECIES_COLUMN[species]}_final"] = terms[species].magmom_final
    for name, value in distances.items():
        row[name] = value

    for step in ("CO2_adsorption", *PATHWAY_STEPS, "overall"):
        row[f"{prefix}_{step}_ev"] = at_u[step]
    row["limiting_step"] = limiting_step
    row["limiting_potential_v"] = u_limiting
    if "O*" in terms:
        row[f"{prefix}_O_formation_from_H2O_ev"] = at_u["O_formation_from_H2O"]
    row["notes"] = (
        "CHE free energies (ZPE, int Cp dT, -TS; slab frozen)"
        if vibrations
        else "electronic energies only (no ZPE, entropy or gas corrections)"
    )
    return row


def pathway_failure_row(material: str, site: str, message: str, status: str = "failed", vibrations: bool = True) -> dict[str, float | str]:
    return {
        "material": material,
        "site": site,
        "pathway_status": status,
        "energy_basis": energy_basis(vibrations),
        "error_message": message,
        "geometry_flags": "",
        "profile": "none",
        "limiting_step": "",
        "limiting_potential_v": float("nan"),
        "notes": "",
    }


# --------------------------------------------------------------------------- workflow


@dataclass
class RunContext:
    args: argparse.Namespace
    settings: dict[str, object]
    catalog: dict[str, dict[str, object]]
    kpts: tuple[int, int, int]
    references: dict[tuple[str, str], TermResult] = field(default_factory=dict)
    clean_slabs: dict[tuple[str, str], TermResult] = field(default_factory=dict)

    @property
    def vib_delta(self) -> float | None:
        if not self.settings["vibrations"]:
            return None
        return float(self.settings["vibration_displacement_angstrom"])


def reference_molecule(name: str, profile: dict[str, object], ctx: RunContext) -> TermResult:
    key = (str(profile["name"]), name)
    if key in ctx.references:
        return ctx.references[key]
    atoms = molecule_atoms(name)
    atoms.center(vacuum=ctx.args.molecule_box_vacuum)
    atoms.pbc = True
    atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))  # closed shell
    result = compute_term(
        term=name,
        label=name.lower(),
        out_dir=PATHWAY_RESULTS_DIR / "_references" / str(profile["name"]) / name.lower(),
        atoms=atoms,
        adsorbate_indices=list(range(len(atoms))),
        n_slab_atoms=0,
        profile=profile,
        ecut=ctx.args.ecut,
        kpts=(1, 1, 1),
        fmax=ctx.args.fmax,
        steps=max(ctx.args.steps, 180),
        overwrite=ctx.args.overwrite,
        vib_delta=ctx.vib_delta,
        box_vacuum=ctx.args.molecule_box_vacuum,
    )
    ctx.references[key] = result
    print0(f"[pathways] reference {name} ({profile['name']}): E = {result.energy_ev:.6f} eV")
    return result


def dft_clean_slab_path(material: str, profile_name: str) -> Path:
    """Optimized clean slab written by run_dft.py for this profile."""
    alternate = DFT_RESULTS_DIR / material / f"clean_{profile_name}" / "clean_slab_optimized.xyz"
    if alternate.exists():
        return alternate
    return DFT_RESULTS_DIR / material / "clean" / "clean_slab_optimized.xyz"


def clean_slab_term(material: str, profile: dict[str, object], dft_row: dict[str, str], ctx: RunContext) -> TermResult:
    from ase.io import read

    profile_name = str(profile["name"])
    key = (material, profile_name)
    if key in ctx.clean_slabs:
        return ctx.clean_slabs[key]

    out_dir = PATHWAY_RESULTS_DIR / material / "_clean" / profile_name
    if profile_name == str(dft_row.get("dft_profile", "")):
        # Same profile as the CO2* adsorption run: reuse the DFT clean slab and its energy.
        path = dft_clean_slab_path(material, profile_name)
        if not path.exists():
            raise InconsistentInputs(f"Missing optimized clean slab: {path}")
        atoms = read(path)
        atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))
        energy = safe_float(dft_row.get("clean_slab_energy_ev"))
        if not math.isfinite(energy):
            raise InconsistentInputs("dft_summary.csv has no clean_slab_energy_ev")
        result = compute_term(
            term="clean_slab", label="clean_slab", out_dir=out_dir, atoms=atoms, adsorbate_indices=[],
            n_slab_atoms=len(atoms), profile=profile, ecut=ctx.args.ecut, kpts=ctx.kpts, fmax=ctx.args.fmax,
            steps=ctx.args.steps, overwrite=ctx.args.overwrite, vib_delta=None,
            known_energy=energy, known_source="dft_summary",
        )
    else:
        atoms = read(GEOMETRY_DIR / material / "clean_slab.traj")
        atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))
        result = compute_term(
            term="clean_slab", label="clean_slab", out_dir=out_dir, atoms=atoms, adsorbate_indices=[],
            n_slab_atoms=len(atoms), profile=profile, ecut=ctx.args.ecut, kpts=ctx.kpts, fmax=ctx.args.fmax,
            steps=ctx.args.steps, overwrite=ctx.args.overwrite, vib_delta=None,
        )
    ctx.clean_slabs[key] = result
    return result


def co2_star_term(material: str, site: str, profile: dict[str, object], dft_row: dict[str, str], clean: TermResult, ctx: RunContext) -> TermResult:
    from ase.io import read

    profile_name = str(profile["name"])
    path = DFT_RESULTS_DIR / material / site / f"{material}_{site}_optimized.xyz"
    if not path.exists():
        raise InconsistentInputs(f"Missing optimized CO2* structure: {path}")
    atoms = read(path)
    n_slab = int(clean.metadata["n_slab_atoms"])
    if len(atoms) - 3 != n_slab:
        raise InconsistentInputs(f"CO2* has {len(atoms) - 3} slab atoms, clean slab has {n_slab}")
    if cell_hash(atoms) != clean.metadata["cell_hash"]:
        raise InconsistentInputs("CO2* and clean slab cells differ")
    atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))
    out_dir = PATHWAY_RESULTS_DIR / material / site / profile_name / "co2_star"
    common = dict(
        term="co2_star", label=f"{material}_{site}_co2_star", out_dir=out_dir, atoms=atoms,
        adsorbate_indices=list(range(n_slab, n_slab + 3)), n_slab_atoms=n_slab, profile=profile,
        ecut=ctx.args.ecut, kpts=ctx.kpts, fmax=ctx.args.fmax, steps=ctx.args.steps,
        overwrite=ctx.args.overwrite, vib_delta=ctx.vib_delta,
    )
    if profile_name == str(dft_row.get("dft_profile", "")):
        energy = safe_float(dft_row.get("adsorbed_energy_ev"))
        if not math.isfinite(energy):
            raise InconsistentInputs("dft_summary.csv has no adsorbed_energy_ev")
        return compute_term(**common, known_energy=energy, known_source="dft_summary")
    # Fallback profile: relax CO2* again in this profile from the DFT geometry.
    return compute_term(**common)


def intermediate_term(material: str, site: str, intermediate: str, profile: dict[str, object], clean: TermResult, site_position, ctx: RunContext) -> TermResult:
    atoms, n_ads = build_intermediate(clean.atoms, site, site_position, intermediate)
    n_slab = int(clean.metadata["n_slab_atoms"])
    return compute_term(
        term=intermediate,
        label=f"{material}_{site}_{intermediate}",
        out_dir=PATHWAY_RESULTS_DIR / material / site / str(profile["name"]) / intermediate,
        atoms=atoms,
        adsorbate_indices=list(range(n_slab, n_slab + n_ads)),
        n_slab_atoms=n_slab,
        profile=profile,
        ecut=ctx.args.ecut,
        kpts=ctx.kpts,
        fmax=ctx.args.fmax,
        steps=ctx.args.steps,
        overwrite=ctx.args.overwrite,
        vib_delta=ctx.vib_delta,
    )


def compute_site_in_profile(material: str, site: str, profile: dict[str, object], dft_row: dict[str, str], ctx: RunContext) -> dict[str, float | str]:
    """Every term of one site in a single profile; raises if any term fails."""
    from scripts.simulations.dft.build_geometries import select_adsorption_sites

    clean = clean_slab_term(material, profile, dft_row, ctx)
    site_position = select_adsorption_sites(clean.atoms, MATERIALS[material].metal_symbol)[site]
    terms: dict[str, TermResult] = {"*": clean}
    terms["CO2*"] = co2_star_term(material, site, profile, dft_row, clean, ctx)
    terms["COOH*"] = intermediate_term(material, site, "cooh", profile, clean, site_position, ctx)
    terms["CO*"] = intermediate_term(material, site, "co", profile, clean, site_position, ctx)
    if ctx.settings["o_star_reference"]:
        terms["O*"] = intermediate_term(material, site, "o", profile, clean, site_position, ctx)
    for name in GAS_SPECIES:
        terms[name] = reference_molecule(name, profile, ctx)

    profiles_used = {term.profile for term in terms.values()}
    if profiles_used != {str(profile["name"])}:
        raise InconsistentInputs(f"mixed profiles: {sorted(profiles_used)}")

    n_slab = int(clean.metadata["n_slab_atoms"])
    flags: list[str] = []
    distances: dict[str, float] = {}
    for species, key in (("COOH*", "cooh"), ("CO*", "co"), ("O*", "o")):
        if species not in terms:
            continue
        term_flags, distance = check_adsorbate_bonds(terms[species].atoms, n_slab, key)
        flags.extend(term_flags)
        distances[f"{'c' if key != 'o' else 'o'}_slab_min_dist_{SPECIES_COLUMN[species]}_ang"] = distance

    return build_result_row(
        material=material, site=site, terms=terms, settings=ctx.settings, args=ctx.args,
        flags=flags, distances=distances,
    )


def run_site(material: str, site: str, dft_row: dict[str, str] | None, ctx: RunContext) -> dict[str, float | str]:
    vibrations = bool(ctx.settings["vibrations"])
    if not dft_row:
        return pathway_failure_row(material, site, "Missing DFT row for case.", vibrations=vibrations)
    if str(dft_row.get("dft_status", "")) != "ok":
        return pathway_failure_row(
            material, site, f"DFT case not usable (status={dft_row.get('dft_status', '')}).", vibrations=vibrations
        )
    preferred = str(dft_row.get("dft_profile", ""))
    if preferred not in ctx.catalog:
        return pathway_failure_row(material, site, f"Unknown DFT profile '{preferred}'.", "inconsistent", vibrations)
    # E(*) and E(CO2*) come from dft_summary.csv: they must be converged to this fmax.
    for column in ("clean_fmax_final_ev_per_ang", "adsorbed_fmax_final_ev_per_ang"):
        value = safe_float(dft_row.get(column))
        if not math.isfinite(value) or value > float(ctx.args.fmax) + 1.0e-9:
            return pathway_failure_row(
                material, site,
                f"dft_summary.csv {column}={dft_row.get(column, 'missing')} is not <= fmax {ctx.args.fmax}; rerun run_dft.py",
                "inconsistent", vibrations,
            )

    errors: list[str] = []
    for profile in profiles_for_case(material, preferred, ctx.catalog):
        name = str(profile["name"])
        print0(f"[pathways] start {material} {site} (profile={name}, overwrite={ctx.args.overwrite})")
        try:
            row = compute_site_in_profile(material, site, profile, dft_row, ctx)
        except InconsistentInputs as exc:
            return pathway_failure_row(material, site, f"{name}: {exc}", "inconsistent", vibrations)
        except Exception as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print0(f"[pathways] {material} {site} failed in profile {name}; every term is recomputed in the next profile")
            continue
        if name != preferred:
            row["notes"] = f"{row['notes']}; all terms recomputed in fallback profile {name}"
        return row
    return pathway_failure_row(material, site, " | ".join(errors), vibrations=vibrations)


def run_pathways(args: argparse.Namespace) -> list[tuple[str, str]]:
    ensure_directories()
    PATHWAY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    materials, sites = validate_inputs(args.materials, args.sites)
    settings = resolve_thermo_settings(args, load_co2rr_config(args.config))
    ctx = RunContext(args=args, settings=settings, catalog=profile_catalog(), kpts=tuple(int(v) for v in args.kpts))

    if not settings["vibrations"]:
        print0("[pathways] vibrations off: electronic energies only, all step columns are dE_* (not dG)")
        if any(float(v) for v in dict(settings["gas_corrections_ev"]).values()):
            print0("[pathways] gas_corrections_ev ignored without vibrations")

    dft_rows = {(row["material"], row["site"]): row for row in read_csv_rows(RESULTS_DIR / "dft_summary.csv")}
    if not dft_rows:
        raise SystemExit("Missing dft_summary.csv rows. Run run_dft.py first.")

    # Rows of other cases are kept only if they use the current schema; requested cases are recomputed.
    rows_by_key: dict[tuple[str, str], dict[str, float | str]] = {
        row_key(row): row for row in read_csv_rows(PATHWAY_SUMMARY_CSV) if "energy_basis" in row
    }

    write_json(
        PATHWAY_SETTINGS_JSON,
        {
            "energy_basis": energy_basis(bool(settings["vibrations"])),
            "fmax_ev_per_ang": args.fmax,
            "max_steps": args.steps,
            "ecut_ev": args.ecut,
            "kpts": list(ctx.kpts),
            "molecule_box_vacuum_ang": args.molecule_box_vacuum,
            **settings,
            "pathway": " -> ".join(PATHWAY_STATES),
            "limiting_potential": "U_L = -max(dG(CO2->COOH*), dG(COOH*->CO*+H2O)) / e at U = 0, V vs RHE",
        },
    )

    for material in materials:
        for site in sites:
            key = (material, site)
            rows_by_key[key] = run_site(material, site, dft_rows.get(key), ctx)
            row = rows_by_key[key]
            if row["pathway_status"] == "ok":
                prefix = energy_prefix(bool(settings["vibrations"]))
                print0(
                    f"[pathways] {material} {site} done ({row['energy_basis']}): "
                    + " ".join(f"{step}={float(row[f'{prefix}_{step}_ev']):.3f}" for step in PATHWAY_STEPS)
                    + f" eV, U_L={float(row['limiting_potential_v']):.3f} V vs RHE"
                    + (f" flags={row['geometry_flags']}" if row["geometry_flags"] else "")
                )
            else:
                print0(f"[pathways] {material} {site} {row['pathway_status']}: {row['error_message']}")
            write_csv(PATHWAY_SUMMARY_CSV, ordered_rows(rows_by_key))

    print0(f"[pathways] wrote summary: {PATHWAY_SUMMARY_CSV}")
    requested = [(material, site) for material in materials for site in sites]
    return [key for key in requested if rows_by_key[key].get("pathway_status") != "ok"]


def _require_ase_and_gpaw() -> None:
    try:
        import ase  # noqa: F401
        import gpaw  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ASE and GPAW are required for CO2RR pathway calculations."
        ) from exc


def main() -> None:
    _require_ase_and_gpaw()
    args = parse_args()
    failed = run_pathways(args)
    if failed:
        print0(f"[pathways] {len(failed)} case(s) did not finish ok: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
