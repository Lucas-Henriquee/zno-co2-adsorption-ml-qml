from __future__ import annotations
from scripts.common.timer import timer, print0

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required. Install with: pip install numpy") from exc

from scripts.common.project_config import ADSORPTION_SITES, DFT_SITES, DFT_RESULTS_DIR, GEOMETRY_DIR, MATERIALS, RESULTS_DIR, ensure_directories


# Sanity thresholds (audit D8).
MIN_INITIAL_DISTANCE_ANG = 1.5  # closest pair allowed in a start geometry (CO2 bonds exempt)
DISSOCIATED_CO_ANG = 1.6  # any C-O longer than this: CO2 broke apart
DESORBED_C_SLAB_ANG = 4.5  # C farther than this from every slab atom: CO2 left the surface
CHEMISORBED_OCO_DEG = 170.0  # O-C-O bent below this: activated, chemisorbed CO2
WARN_INITIAL_FMAX = 5.0  # eV/Ang; larger initial forces point to a poor start geometry
WARN_ABS_EADS = 3.0  # eV; |Eads| above this is unphysical for CO2 on an oxide


class GeometryError(RuntimeError):
    """Start geometry with atoms closer than MIN_INITIAL_DISTANCE_ANG."""


class RelaxationNotConverged(RuntimeError):
    """BFGS hit its step limit: a geometric failure that another SCF profile cannot fix."""


class CleanSlabEnergyError(RuntimeError):
    """Positive clean-slab energy: a broken slab (GPAW energies are relative to free atoms)."""


def mpi_rank() -> int:
    """MPI rank from GPAW, imported lazily so the module works without GPAW (rank 0)."""
    try:
        from gpaw.mpi import world
    except ImportError:
        return 0
    return int(world.rank)


def is_master() -> bool:
    return mpi_rank() == 0


def _require_ase_and_gpaw() -> None:
    try:
        import ase  # noqa: F401
        import gpaw  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ASE and GPAW are required for DFT runs. Install dependencies before running this script."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DFT optimization for CO2 adsorption cases.")
    parser.add_argument("--materials", nargs="*", default=sorted(MATERIALS.keys()))
    parser.add_argument("--sites", nargs="*", default=list(ADSORPTION_SITES))
    parser.add_argument("--fmax", type=float, default=0.05, help="Optimization force threshold (eV/Ang).")
    parser.add_argument("--steps", type=int, default=180, help="Maximum BFGS steps per block.")
    parser.add_argument(
        "--relax-extra-blocks",
        type=int,
        default=1,
        help="Extra blocks of --steps BFGS steps, continued from the last geometry, "
        "before a relaxation that hit the step limit is declared failed.",
    )
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
        "--co2-box-vacuum",
        type=float,
        default=8.0,
        help="Vacuum padding for gas-phase CO2 reference (Ang).",
    )
    parser.add_argument(
        "--scf-force-convergence",
        type=float,
        default=None,
        help="Optional GPAW SCF force criterion (eV/Ang); default: GPAW's own criteria.",
    )
    return parser.parse_args()


def validate_inputs(materials: Iterable[str], sites: Iterable[str]) -> tuple[list[str], list[str]]:
    unknown_materials = [name for name in materials if name not in MATERIALS]
    if unknown_materials:
        raise SystemExit(f"Unknown materials: {', '.join(unknown_materials)}")

    unknown_sites = [name for name in sites if name not in DFT_SITES]
    if unknown_sites:
        raise SystemExit(f"Unknown sites: {', '.join(unknown_sites)}")

    return list(materials), list(sites)


# Fallbacks only make the SCF more conservative (mixer, maxiter). The smearing stays at
# 0.05 eV: with a PBE gap of ~0.6-0.8 eV a wider one occupies the conduction band.
OPTIMIZATION_PROFILES = (
    {"name": "base", "mode": "pw", "smearing": 0.05, "maxiter": 180, "mixer": (0.05, 5, 50)},
    {"name": "robust", "mode": "pw", "smearing": 0.05, "maxiter": 280, "mixer": (0.03, 7, 100)},
    {"name": "very_robust", "mode": "pw", "smearing": 0.05, "maxiter": 420, "mixer": (0.02, 10, 150)},
)

CEO2_EXTRA_PROFILES = (
    {"name": "ce_lcao", "mode": "lcao", "smearing": 0.30, "maxiter": 500, "mixer": (0.02, 8, 180)},
    {"name": "ce_fd", "mode": "fd", "smearing": 0.35, "maxiter": 600, "mixer": (0.01, 12, 240)},
)


def build_calculator(
    ecut: float,
    kpts: tuple[int, int, int],
    txt_file: Path,
    mode: str = "pw",
    smearing: float = 0.05,
    maxiter: int = 180,
    mixer: tuple[float, int, float] | None = None,
    scf_force_convergence: float | None = None,
):
    from gpaw import FermiDirac, GPAW, Mixer, PW

    calc_kwargs = {}
    if scf_force_convergence is not None:
        # Added to GPAW's default criteria (energy, density, eigenstates).
        calc_kwargs["convergence"] = {"forces": float(scf_force_convergence)}
    if mixer is not None:
        calc_kwargs["mixer"] = Mixer(mixer[0], mixer[1], mixer[2])
    if mode == "lcao":
        calc_kwargs["basis"] = "dzp"
    elif mode == "fd":
        calc_kwargs["h"] = 0.24

    gpaw_mode = PW(ecut) if mode == "pw" else mode

    return GPAW(
        mode=gpaw_mode,
        xc="PBE",
        occupations=FermiDirac(smearing),
        kpts=kpts,
        symmetry="off",
        maxiter=maxiter,
        txt=str(txt_file),
        **calc_kwargs,
    )


def apply_bottom_layer_constraint(atoms, adsorbate_atoms: int = 0, slab_free_fraction: float = 0.60) -> None:
    from ase.constraints import FixAtoms

    slab_count = len(atoms) - adsorbate_atoms
    slab_indices = np.arange(slab_count, dtype=int)
    slab_z = atoms.positions[slab_indices, 2]
    z_min = float(np.min(slab_z))
    z_max = float(np.max(slab_z))
    threshold = z_min + (1.0 - slab_free_fraction) * (z_max - z_min)
    fixed_indices = [int(i) for i in slab_indices if atoms.positions[i, 2] <= threshold]
    atoms.set_constraint(FixAtoms(indices=fixed_indices))


def max_free_force(atoms) -> float:
    """Largest force norm on the free atoms (FixAtoms forces are zeroed by ASE)."""
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def check_initial_geometry(atoms, label: str, adsorbate_atoms: int = 0) -> float:
    """Raise GeometryError if two atoms overlap; bonds inside the adsorbate are exempt."""
    n_atoms = len(atoms)
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    if adsorbate_atoms:
        first = max(n_atoms - adsorbate_atoms, 0)
        distances[first:, first:] = np.inf
    i, j = np.unravel_index(int(np.argmin(distances)), distances.shape)
    d_min = float(distances[i, j])
    if d_min < MIN_INITIAL_DISTANCE_ANG:
        raise GeometryError(
            f"{label}: atoms {i} ({atoms[i].symbol}) and {j} ({atoms[j].symbol}) are "
            f"{d_min:.3f} Ang apart (< {MIN_INITIAL_DISTANCE_ANG} Ang) in the start geometry"
        )
    return d_min


def optimize_structure(
    atoms,
    output_dir: Path,
    label: str,
    ecut: float,
    kpts: tuple[int, int, int],
    fmax: float,
    steps: int,
    profiles=None,
    relax_extra_blocks: int = 1,
    scf_force_convergence: float | None = None,
) -> tuple[float, float, object, str]:
    """Relax with the first SCF profile that works.

    An SCF/GPAW exception moves to the next profile, from the initial geometry. A BFGS
    that hits the step limit is a geometric problem: it continues from its last geometry
    (same profile and Hessian) for up to `relax_extra_blocks` blocks of `steps`, then
    fails without trying other profiles.
    """
    from ase.io import write
    from ase.optimize import BFGS

    output_dir.mkdir(parents=True, exist_ok=True)
    write(output_dir / f"{label}_initial.cif", atoms)
    last_error: Exception | None = None

    if profiles is None:
        profiles = OPTIMIZATION_PROFILES

    for profile in profiles:
        profile_name = profile["name"]
        trial = atoms.copy()
        calc = build_calculator(
            ecut=ecut,
            kpts=kpts,
            txt_file=output_dir / f"{label}.{profile_name}.txt",
            mode=str(profile.get("mode", "pw")),
            smearing=float(profile["smearing"]),
            maxiter=int(profile["maxiter"]),
            mixer=profile["mixer"],
            scf_force_convergence=scf_force_convergence,
        )
        trial.calc = calc

        dyn = BFGS(
            trial,
            trajectory=str(output_dir / f"{label}.{profile_name}.traj"),
            logfile=str(output_dir / f"{label}.{profile_name}.opt.log"),
        )

        try:
            fmax_initial = max_free_force(trial)  # cached by the calculator for BFGS
            converged = dyn.run(fmax=fmax, steps=steps)
            restarts = 0
            # BFGS.run returns False when it stops at the step limit; it does not raise.
            # Another run() continues from the current geometry for `steps` more steps.
            while (not converged or max_free_force(trial) > fmax) and restarts < relax_extra_blocks:
                restarts += 1
                print0(
                    f"[dft] {label}: BFGS hit the step limit ({dyn.nsteps} steps, "
                    f"profile={profile_name}); continuing from the last geometry"
                )
                converged = dyn.run(fmax=fmax, steps=steps)
            fmax_final = max_free_force(trial)
            if not converged or fmax_final > fmax:
                raise RelaxationNotConverged(
                    f"{label}: BFGS did not converge in {dyn.nsteps} steps "
                    f"({restarts} continuation(s) of {steps}; "
                    f"fmax {fmax_final:.3f} > {fmax} eV/Ang, profile={profile_name})"
                )
            trial.info["n_steps"] = int(dyn.nsteps)
            trial.info["relax_restarts"] = restarts
            trial.info["fmax_initial"] = fmax_initial
            trial.info["fmax_final"] = fmax_final
            energy = float(trial.get_potential_energy())
            calc.write(str(output_dir / f"{label}.gpw"), mode="all")
            write(output_dir / f"{label}_optimized.xyz", trial)
            write(output_dir / f"{label}_optimized.cif", trial)
            gap = compute_band_gap_safe(calc)
            trial.info["band_gap_gamma"] = compute_band_gap_gamma(
                calc, kpts, output_dir / f"{label}.{profile_name}.gap_gamma.txt"
            )
            return energy, gap, trial, profile_name
        except Exception as exc:
            last_error = exc
            if is_master():
                (output_dir / f"{label}.{profile_name}.error.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
            if isinstance(exc, RelaxationNotConverged):
                # Restarting from the initial geometry with another SCF profile cannot help.
                raise

    raise RuntimeError(
        f"All optimization profiles failed for {label}. Last error: {last_error}"
    )


def compute_band_gap_safe(calc) -> float:
    """Kohn-Sham band gap via ase.dft.bandgap; a failure is reported, not hidden."""
    try:
        from ase.dft.bandgap import bandgap

        gap, _, _ = bandgap(calc, output=None)  # output=None: no print on every MPI rank
        return float(gap)
    except Exception as exc:
        print0(f"[dft] warning: band gap not computed ({type(exc).__name__}: {exc})")
        return float("nan")


def compute_band_gap_gamma(calc, kpts, txt_file) -> float:
    """Kohn-Sham gap on a Gamma-centred grid of the same size as the relaxation mesh.

    An even Monkhorst-Pack mesh skips Gamma, where the ZnO gap lies, so the gap on
    the relaxation mesh is only an upper bound. A non-self-consistent (fixed-density)
    calculation adds Gamma without changing the energies used for Eads.
    """
    size = tuple(int(n) for n in kpts)
    if all(n % 2 == 1 for n in size):
        return compute_band_gap_safe(calc)  # odd meshes already contain Gamma
    try:
        n_occupied = int(round(calc.get_number_of_electrons() / 2))
        gamma_calc = calc.fixed_density(
            kpts={"size": size, "gamma": True},
            symmetry="off",
            nbands=n_occupied + 12,
            convergence={"bands": n_occupied + 8},
            txt=str(txt_file),
        )
    except Exception as exc:
        print0(f"[dft] warning: Gamma-grid band gap not computed ({type(exc).__name__}: {exc})")
        return float("nan")
    return compute_band_gap_safe(gamma_calc)


def co2_reference_energy(
    output_dir: Path,
    ecut: float,
    vacuum: float,
    fmax: float,
    steps: int,
    profile: dict[str, object],
    scf_force_convergence: float | None = None,
) -> float:
    from ase import Atoms
    from ase.io import write
    from ase.optimize import BFGS

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_name = str(profile["name"])
    co2 = Atoms("OCO", positions=[(-1.16, 0.0, 0.0), (0.0, 0.0, 0.0), (1.16, 0.0, 0.0)])
    co2.center(vacuum=vacuum)
    co2.pbc = True
    write(output_dir / f"co2_ref.{profile_name}_initial.cif", co2)

    calc = build_calculator(
        ecut=ecut,
        kpts=(1, 1, 1),
        txt_file=output_dir / f"co2_ref.{profile_name}.txt",
        mode=str(profile.get("mode", "pw")),
        smearing=float(profile["smearing"]),
        maxiter=max(220, int(profile["maxiter"])),
        mixer=tuple(profile["mixer"]),
        scf_force_convergence=scf_force_convergence,
    )
    co2.calc = calc
    dyn = BFGS(
        co2,
        trajectory=str(output_dir / f"co2_ref.{profile_name}.traj"),
        logfile=str(output_dir / f"co2_ref.{profile_name}.opt.log"),
    )
    converged = dyn.run(fmax=fmax, steps=steps)
    fmax_final = max_free_force(co2)
    if not converged or fmax_final > fmax:
        raise RuntimeError(
            f"co2_ref.{profile_name}: BFGS did not converge in {steps} steps "
            f"(fmax {fmax_final:.3f} > {fmax} eV/Ang)"
        )

    energy = float(co2.get_potential_energy())
    calc.write(str(output_dir / f"co2_ref.{profile_name}.gpw"), mode="all")
    write(output_dir / f"co2_ref.{profile_name}_optimized.cif", co2)

    if is_master():
        with (output_dir / f"co2_reference.{profile_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "profile": profile_name,
                    "co2_reference_energy_ev": energy,
                },
                handle,
                indent=2,
            )

    return energy


def adsorption_geometry_metrics(atoms) -> dict[str, float]:
    # By construction in build_geometries.py the final three atoms are O, C, O for CO2.
    o1_idx = len(atoms) - 3
    c_idx = len(atoms) - 2
    o2_idx = len(atoms) - 1
    slab_indices = np.arange(len(atoms) - 3)
    metal_indices = [int(i) for i in slab_indices if atoms[int(i)].symbol != "O"]

    # mic=True: an O that crossed the cell boundary is measured to its nearest image.
    c_o_1 = float(atoms.get_distance(c_idx, o1_idx, mic=True))
    c_o_2 = float(atoms.get_distance(c_idx, o2_idx, mic=True))
    oco_angle = float(atoms.get_angle(o1_idx, c_idx, o2_idx, mic=True))

    # Height of C above the topmost slab atom; not a distance to any atom.
    slab_z_max = float(np.max(atoms.positions[slab_indices, 2]))
    carbon_z = float(atoms.positions[c_idx, 2])
    carbon_surface_distance = carbon_z - slab_z_max

    c_min_slab = float(atoms.get_distances(c_idx, slab_indices, mic=True).min())
    o_min_metal = float(min(
        atoms.get_distances(o_idx, metal_indices, mic=True).min() for o_idx in (o1_idx, o2_idx)
    ))

    return {
        "co_bond_1_ang": c_o_1,
        "co_bond_2_ang": c_o_2,
        "oco_angle_deg": oco_angle,
        "c_surface_distance_ang": carbon_surface_distance,
        "c_min_slab_distance_ang": c_min_slab,
        "o_min_metal_distance_ang": o_min_metal,
    }


def classify_final_state(metrics: dict[str, float]) -> str:
    """Relaxed CO2 state, checked in order: dissociated, desorbed, chemisorbed, physisorbed."""
    keys = ("co_bond_1_ang", "co_bond_2_ang", "c_min_slab_distance_ang", "oco_angle_deg")
    values = [float(metrics.get(key, float("nan"))) for key in keys]
    if any(math.isnan(value) for value in values):
        return "unknown"
    co_1, co_2, c_min_slab, oco_angle = values
    if max(co_1, co_2) > DISSOCIATED_CO_ANG:
        return "dissociated"
    if c_min_slab > DESORBED_C_SLAB_ANG:
        return "desorbed"
    if oco_angle < CHEMISORBED_OCO_DEG:
        return "chemisorbed"
    return "physisorbed"


def sanity_warnings(clean_atoms, ads_atoms, adsorption_energy: float) -> str:
    """Semicolon-separated warnings for a finished case; empty when nothing is suspicious."""
    warnings = []
    for name, atoms in (("clean", clean_atoms), ("adsorbed", ads_atoms)):
        fmax_initial = float(atoms.info.get("fmax_initial", float("nan")))
        if fmax_initial > WARN_INITIAL_FMAX:
            warnings.append(f"{name} initial fmax {fmax_initial:.2f} > {WARN_INITIAL_FMAX} eV/Ang")
    if abs(adsorption_energy) > WARN_ABS_EADS:
        warnings.append(f"|Eads| {abs(adsorption_energy):.2f} > {WARN_ABS_EADS} eV")
    return "; ".join(warnings)


def check_clean_slab_energy(energy: float, material: str, profile_name: str) -> None:
    # A bound slab is below its free atoms; the old, broken ZnO slab gave +317 eV.
    if energy > 0.0:
        raise CleanSlabEnergyError(
            f"{material} clean slab energy is positive ({energy:.3f} eV, profile={profile_name}); "
            "the slab geometry is unphysical, check build_geometries.py"
        )


def read_geometry(path: Path):
    from ase.io import read

    if not path.exists():
        raise SystemExit(
            f"Missing geometry file: {path}. Run build_geometries.py before DFT calculations."
        )
    return read(path)


def write_summary(rows: list[dict[str, float | str]]) -> None:
    if not rows or not is_master():
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output_csv = RESULTS_DIR / "dft_summary.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_summary(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def kpts_label(kpts: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in kpts)


def settings_match(row: dict[str, float | str], ecut: float, kpts: str, n_slab_atoms: int | None) -> bool:
    """True if an existing row was computed with the same cutoff, k-mesh and slab size."""
    try:
        return (
            math.isclose(float(row["ecut_ev"]), float(ecut))
            and str(row["kpts"]) == kpts
            and n_slab_atoms is not None
            and int(float(row["n_slab_atoms"])) == n_slab_atoms
        )
    except (KeyError, TypeError, ValueError):
        return False  # rows from older versions lack the settings columns


def compatible_existing_rows(
    rows: list[dict[str, float | str]],
    ecut: float,
    kpts: str,
    slab_atoms_by_material: dict[str, int],
) -> list[dict[str, float | str]]:
    """Keep only rows with the current physics settings; report the dropped ones."""
    kept = []
    dropped = []
    for row in rows:
        n_slab = slab_atoms_by_material.get(str(row.get("material", "")))
        if settings_match(row, ecut, kpts, n_slab):
            kept.append(row)
        else:
            dropped.append(row_key(row))
    if dropped:
        print0(
            f"[dft] warning: dropped {len(dropped)} row(s) from the existing summary computed "
            f"with other settings than ecut={ecut} eV, kpts={kpts} or the current slab: {dropped}"
        )
    return kept


def slab_atom_counts(materials: Iterable[str]) -> dict[str, int]:
    """Atoms in each material's current clean slab, for materials with a geometry file."""
    counts = {}
    for material in materials:
        path = GEOMETRY_DIR / material / "clean_slab.traj"
        if path.exists():
            counts[material] = len(read_geometry(path))
    return counts


def row_key(row: dict[str, float | str]) -> tuple[str, str]:
    return str(row.get("material", "")), str(row.get("site", ""))


def ordered_rows(rows_by_key: dict[tuple[str, str], dict[str, float | str]]) -> list[dict[str, float | str]]:
    material_order = {name: idx for idx, name in enumerate(MATERIALS.keys())}
    site_order = {name: idx for idx, name in enumerate(DFT_SITES)}
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


def make_failure_row(
    material: str,
    site: str,
    status: str,
    message: str,
    co2_energy: float = float("nan"),
) -> dict[str, float | str]:
    return {
        "material": material,
        "site": site,
        "dft_status": status,
        "dft_profile": "none",
        "error_message": message,
        "clean_slab_energy_ev": float("nan"),
        "adsorbed_energy_ev": float("nan"),
        "co2_reference_energy_ev": co2_energy,
        "adsorption_energy_ev": float("nan"),
        "clean_band_gap_ev": float("nan"),
        "adsorbed_band_gap_ev": float("nan"),
        "clean_band_gap_gamma_ev": float("nan"),
        "adsorbed_band_gap_gamma_ev": float("nan"),
        "co_bond_1_ang": float("nan"),
        "co_bond_2_ang": float("nan"),
        "oco_angle_deg": float("nan"),
        "c_surface_distance_ang": float("nan"),
        "c_min_slab_distance_ang": float("nan"),
        "o_min_metal_distance_ang": float("nan"),
        "final_state": "",
        "warnings": "",
    }


def run_dft_workflow(args: argparse.Namespace) -> None:
    ensure_directories()
    materials, sites = validate_inputs(args.materials, args.sites)
    kpts = tuple(int(value) for value in args.kpts)
    output_csv = RESULTS_DIR / "dft_summary.csv"
    # Absent from Namespaces built by older callers; defaults match parse_args.
    relax_extra_blocks = int(getattr(args, "relax_extra_blocks", 1))
    scf_force_convergence = getattr(args, "scf_force_convergence", None)
    n_slab_by_material: dict[str, int] = {}

    reference_dir = DFT_RESULTS_DIR / "_reference"
    profile_catalog = {
        str(profile["name"]): profile for profile in (OPTIMIZATION_PROFILES + CEO2_EXTRA_PROFILES)
    }
    co2_reference_cache: dict[str, float] = {}

    def get_co2_reference_for_profile(profile_name: str) -> float:
        cached = co2_reference_cache.get(profile_name)
        if cached is not None:
            return cached
        profile = profile_catalog[profile_name]
        with timer(f"CO2 reference → {profile_name}"):
            energy = co2_reference_energy(
                output_dir=reference_dir / profile_name,
                ecut=args.ecut,
                vacuum=args.co2_box_vacuum,
                fmax=args.fmax,
                steps=args.steps,
                profile=profile,
                scf_force_convergence=scf_force_convergence,
            )
        co2_reference_cache[profile_name] = energy
        print0(f"[dft] CO2 gas reference energy ({profile_name}) = {energy:.6f} eV")
        return energy

    # Merge rows from an earlier run only if they share cutoff, k-mesh and slab size.
    existing_rows = read_existing_summary(output_csv)
    existing_materials = {str(row.get("material", "")) for row in existing_rows}
    rows_by_key: dict[tuple[str, str], dict[str, float | str]] = {
        row_key(row): row
        for row in compatible_existing_rows(
            existing_rows,
            ecut=args.ecut,
            kpts=kpts_label(kpts),
            slab_atoms_by_material=slab_atom_counts(existing_materials & set(MATERIALS)),
        )
    }

    def upsert_row(row: dict[str, float | str]) -> None:
        material = str(row["material"])
        rows_by_key[row_key(row)] = {
            **row,
            "ecut_ev": float(args.ecut),
            "kpts": kpts_label(kpts),
            "n_slab_atoms": n_slab_by_material.get(material, ""),
        }

    def flush_summary() -> None:
        write_summary(ordered_rows(rows_by_key))

    for material in materials:
        material_out = DFT_RESULTS_DIR / material
        material_out.mkdir(parents=True, exist_ok=True)
        profiles_for_material = OPTIMIZATION_PROFILES
        if material == "CeO2":
            profiles_for_material = CEO2_EXTRA_PROFILES + OPTIMIZATION_PROFILES

        clean_atoms_in = read_geometry(GEOMETRY_DIR / material / "clean_slab.traj")
        apply_bottom_layer_constraint(clean_atoms_in, adsorbate_atoms=0)
        n_slab_by_material[material] = len(clean_atoms_in)

        try:
            check_initial_geometry(clean_atoms_in, f"{material} clean_slab")
            with timer(f"DFT clean slab → {material}"):
                clean_energy, clean_gap, clean_atoms_opt, clean_profile = optimize_structure(
                    atoms=clean_atoms_in,
                    output_dir=material_out / "clean",
                    label="clean_slab",
                    ecut=args.ecut,
                    kpts=kpts,
                    fmax=args.fmax,
                    steps=args.steps,
                    profiles=profiles_for_material,
                    relax_extra_blocks=relax_extra_blocks,
                    scf_force_convergence=scf_force_convergence,
                )
            print0(
                f"[dft] {material} clean slab energy = {clean_energy:.6f} eV "
                f"(profile={clean_profile})"
            )
            check_clean_slab_energy(clean_energy, material, clean_profile)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            status = "bad_initial_geometry" if isinstance(exc, GeometryError) else "clean_failed"
            print0(f"[dft] {material} clean slab failed: {message}")
            for site in sites:
                upsert_row(
                    make_failure_row(
                        material=material,
                        site=site,
                        status=status,
                        message=message,
                    )
                )
            flush_summary()
            continue

        # Clean slab per SCF profile: Eads subtracts terms of one profile only.
        clean_by_profile = {clean_profile: (clean_energy, clean_gap, clean_atoms_opt)}
        clean_errors: dict[str, Exception] = {}  # a failed fallback slab is not retried per site

        def clean_for_profile(profile_name: str):
            if profile_name in clean_errors:
                raise clean_errors[profile_name]
            if profile_name not in clean_by_profile:
                slab_in = read_geometry(GEOMETRY_DIR / material / "clean_slab.traj")
                apply_bottom_layer_constraint(slab_in, adsorbate_atoms=0)
                try:
                    with timer(f"DFT clean slab → {material} ({profile_name})"):
                        energy, gap, atoms_opt, _ = optimize_structure(
                            atoms=slab_in,
                            output_dir=material_out / f"clean_{profile_name}",
                            label="clean_slab",
                            ecut=args.ecut,
                            kpts=kpts,
                            fmax=args.fmax,
                            steps=args.steps,
                            profiles=(profile_catalog[profile_name],),
                            relax_extra_blocks=relax_extra_blocks,
                            scf_force_convergence=scf_force_convergence,
                        )
                    check_clean_slab_energy(energy, material, profile_name)
                except Exception as exc:
                    clean_errors[profile_name] = exc
                    raise
                clean_by_profile[profile_name] = (energy, gap, atoms_opt)
            return clean_by_profile[profile_name]

        try:
            get_co2_reference_for_profile(clean_profile)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print0(f"[dft] CO2 reference ({clean_profile}) failed: {message}")
            for site in sites:
                upsert_row(
                    {
                        **make_failure_row(
                            material=material,
                            site=site,
                            status="co2_ref_failed",
                            message=message,
                        ),
                        "clean_slab_energy_ev": clean_energy,
                        "clean_band_gap_ev": clean_gap,
                    }
                )
            flush_summary()
            continue

        # The adsorbate starts from the clean-slab profile and may fall back to later ones.
        profile_names = [str(profile["name"]) for profile in profiles_for_material]
        ads_profiles = tuple(profiles_for_material[profile_names.index(clean_profile):])

        for site in sites:
            case_dir = material_out / site
            ads_atoms_in = read_geometry(GEOMETRY_DIR / material / f"co2_{site}.traj")
            apply_bottom_layer_constraint(ads_atoms_in, adsorbate_atoms=3)
            clean_co2_energy = float("nan")

            try:
                check_initial_geometry(ads_atoms_in, f"{material}_{site}", adsorbate_atoms=3)
                with timer(f"DFT adsorption → {material} | {site}"):
                    ads_energy, ads_gap, ads_atoms_opt, ads_profile = optimize_structure(
                        atoms=ads_atoms_in,
                        output_dir=case_dir,
                        label=f"{material}_{site}",
                        ecut=args.ecut,
                        kpts=kpts,
                        fmax=args.fmax,
                        steps=args.steps,
                        profiles=ads_profiles,
                        relax_extra_blocks=relax_extra_blocks,
                        scf_force_convergence=scf_force_convergence,
                    )

                slab_energy, slab_gap, slab_atoms_opt = clean_for_profile(ads_profile)
                clean_co2_energy = get_co2_reference_for_profile(ads_profile)
                adsorption_energy = ads_energy - slab_energy - clean_co2_energy
                geometry_metrics = adsorption_geometry_metrics(ads_atoms_opt)
                final_state = classify_final_state(geometry_metrics)
                warnings = sanity_warnings(slab_atoms_opt, ads_atoms_opt, adsorption_energy)
                print0(
                    f"[dft] {material} {site} adsorption energy = {adsorption_energy:.6f} eV "
                    f"(profile={ads_profile}, final_state={final_state})"
                )
                if warnings:
                    print0(f"[dft] warning: {material} {site}: {warnings}")

                upsert_row(
                    {
                        "material": material,
                        "site": site,
                        "dft_status": "ok",
                        "dft_profile": ads_profile,
                        "error_message": "",
                        "clean_slab_energy_ev": slab_energy,
                        "adsorbed_energy_ev": ads_energy,
                        "co2_reference_energy_ev": clean_co2_energy,
                        "adsorption_energy_ev": adsorption_energy,
                        "clean_band_gap_ev": slab_gap,
                        "adsorbed_band_gap_ev": ads_gap,
                        "clean_band_gap_gamma_ev": slab_atoms_opt.info.get("band_gap_gamma", float("nan")),
                        "adsorbed_band_gap_gamma_ev": ads_atoms_opt.info.get("band_gap_gamma", float("nan")),
                        "clean_n_steps": slab_atoms_opt.info["n_steps"],
                        "clean_relax_restarts": slab_atoms_opt.info.get("relax_restarts", 0),
                        "clean_fmax_initial_ev_per_ang": slab_atoms_opt.info.get("fmax_initial", float("nan")),
                        "clean_fmax_final_ev_per_ang": slab_atoms_opt.info["fmax_final"],
                        "adsorbed_n_steps": ads_atoms_opt.info["n_steps"],
                        "adsorbed_relax_restarts": ads_atoms_opt.info.get("relax_restarts", 0),
                        "adsorbed_fmax_initial_ev_per_ang": ads_atoms_opt.info.get("fmax_initial", float("nan")),
                        "adsorbed_fmax_final_ev_per_ang": ads_atoms_opt.info["fmax_final"],
                        **geometry_metrics,
                        "final_state": final_state,
                        "warnings": warnings,
                    }
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print0(f"[dft] {material} {site} failed: {message}")
                if isinstance(exc, GeometryError):
                    status = "bad_initial_geometry"
                elif isinstance(exc, CleanSlabEnergyError):
                    status = "clean_failed"
                else:
                    status = "ads_failed"
                upsert_row(
                    {
                        **make_failure_row(
                            material=material,
                            site=site,
                            status=status,
                            message=message,
                            co2_energy=clean_co2_energy,
                        ),
                        "clean_slab_energy_ev": clean_energy,
                        "clean_band_gap_ev": clean_gap,
                    }
                )

            flush_summary()

    flush_summary()
    print0(f"[dft] wrote summary: {output_csv}")

    # A failed case must fail the workflow, so dependent methods do not run on it.
    requested = [(material, site) for material in materials for site in sites]
    failed = [key for key in requested if rows_by_key.get(key, {}).get("dft_status") != "ok"]
    if failed:
        print0(f"[dft] {len(failed)} case(s) did not finish ok: {failed}")
        raise SystemExit(1)


def main() -> None:
    _require_ase_and_gpaw()
    args = parse_args()
    run_dft_workflow(args)


if __name__ == "__main__":
    main()
