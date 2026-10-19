from __future__ import annotations
from scripts.common.timer import timer, print0

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required. Install with: pip install numpy") from exc

from scripts.common.project_config import ADSORPTION_SITES, DFT_RESULTS_DIR, GEOMETRY_DIR, MATERIALS, RESULTS_DIR, ensure_directories


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
    parser.add_argument("--steps", type=int, default=160, help="Maximum BFGS steps.")
    parser.add_argument("--ecut", type=float, default=500.0, help="Plane-wave cutoff (eV).")
    parser.add_argument(
        "--kpts",
        nargs=3,
        type=int,
        default=(3, 3, 1),
        metavar=("KX", "KY", "KZ"),
        help="Monkhorst-Pack k-point mesh for slab calculations.",
    )
    parser.add_argument(
        "--co2-box-vacuum",
        type=float,
        default=8.0,
        help="Vacuum padding for gas-phase CO2 reference (Ang).",
    )
    return parser.parse_args()


def validate_inputs(materials: Iterable[str], sites: Iterable[str]) -> tuple[list[str], list[str]]:
    unknown_materials = [name for name in materials if name not in MATERIALS]
    if unknown_materials:
        raise SystemExit(f"Unknown materials: {', '.join(unknown_materials)}")

    unknown_sites = [name for name in sites if name not in ADSORPTION_SITES]
    if unknown_sites:
        raise SystemExit(f"Unknown sites: {', '.join(unknown_sites)}")

    return list(materials), list(sites)


OPTIMIZATION_PROFILES = (
    {"name": "base", "mode": "pw", "smearing": 0.05, "maxiter": 180, "mixer": (0.05, 5, 50)},
    {"name": "robust", "mode": "pw", "smearing": 0.15, "maxiter": 280, "mixer": (0.03, 7, 100)},
    {"name": "very_robust", "mode": "pw", "smearing": 0.25, "maxiter": 420, "mixer": (0.02, 10, 150)},
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
):
    from gpaw import FermiDirac, GPAW, Mixer, PW

    calc_kwargs = {}
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


def optimize_structure(
    atoms,
    output_dir: Path,
    label: str,
    ecut: float,
    kpts: tuple[int, int, int],
    fmax: float,
    steps: int,
    profiles=None,
) -> tuple[float, float, object, str]:
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
        )
        trial.calc = calc

        dyn = BFGS(
            trial,
            trajectory=str(output_dir / f"{label}.{profile_name}.traj"),
            logfile=str(output_dir / f"{label}.{profile_name}.opt.log"),
        )

        try:
            converged = dyn.run(fmax=fmax, steps=steps)
            # BFGS.run returns False when it stops at the step limit; it does not raise.
            fmax_final = max_free_force(trial)
            if not converged or fmax_final > fmax:
                raise RuntimeError(
                    f"{label}: BFGS did not converge in {steps} steps "
                    f"(fmax {fmax_final:.3f} > {fmax} eV/Ang)"
                )
            trial.info["n_steps"] = int(dyn.nsteps)
            trial.info["fmax_final"] = fmax_final
            energy = float(trial.get_potential_energy())
            calc.write(str(output_dir / f"{label}.gpw"), mode="all")
            write(output_dir / f"{label}_optimized.xyz", trial)
            write(output_dir / f"{label}_optimized.cif", trial)
            gap = compute_band_gap_safe(calc)
            return energy, gap, trial, profile_name
        except Exception as exc:
            last_error = exc
            (output_dir / f"{label}.{profile_name}.error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )

    raise RuntimeError(
        f"All optimization profiles failed for {label}. Last error: {last_error}"
    )


def compute_band_gap_safe(calc) -> float:
    try:
        from gpaw.bandgap import bandgap
        gap_data = bandgap(calc)
    except Exception:
        return float("nan")

    if isinstance(gap_data, (tuple, list)) and gap_data:
        return float(gap_data[0])
    return float(gap_data)


def co2_reference_energy(
    output_dir: Path,
    ecut: float,
    vacuum: float,
    fmax: float,
    steps: int,
    profile: dict[str, object],
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

    c_o_1 = float(atoms.get_distance(c_idx, o1_idx))
    c_o_2 = float(atoms.get_distance(c_idx, o2_idx))
    oco_angle = float(atoms.get_angle(o1_idx, c_idx, o2_idx))

    slab_z_max = float(np.max(atoms.positions[slab_indices, 2]))
    carbon_z = float(atoms.positions[c_idx, 2])
    carbon_surface_distance = carbon_z - slab_z_max

    return {
        "co_bond_1_ang": c_o_1,
        "co_bond_2_ang": c_o_2,
        "oco_angle_deg": oco_angle,
        "c_surface_distance_ang": carbon_surface_distance,
    }


def read_geometry(path: Path):
    from ase.io import read

    if not path.exists():
        raise SystemExit(
            f"Missing geometry file: {path}. Run build_geometries.py before DFT calculations."
        )
    return read(path)


def write_summary(rows: list[dict[str, float | str]]) -> None:
    if not rows:
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
        "co_bond_1_ang": float("nan"),
        "co_bond_2_ang": float("nan"),
        "oco_angle_deg": float("nan"),
        "c_surface_distance_ang": float("nan"),
    }


def run_dft_workflow(args: argparse.Namespace) -> None:
    ensure_directories()
    materials, sites = validate_inputs(args.materials, args.sites)
    kpts = tuple(int(value) for value in args.kpts)
    output_csv = RESULTS_DIR / "dft_summary.csv"

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
            )
        co2_reference_cache[profile_name] = energy
        print0(f"[dft] CO2 gas reference energy ({profile_name}) = {energy:.6f} eV")
        return energy

    rows_by_key: dict[tuple[str, str], dict[str, float | str]] = {
        row_key(row): row for row in read_existing_summary(output_csv)
    }

    def upsert_row(row: dict[str, float | str]) -> None:
        rows_by_key[row_key(row)] = row

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

        try:
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
                )
            print0(
                f"[dft] {material} clean slab energy = {clean_energy:.6f} eV "
                f"(profile={clean_profile})"
            )
            clean_profile_cfg = profile_catalog[clean_profile]
            clean_co2_energy = get_co2_reference_for_profile(clean_profile)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print0(f"[dft] {material} clean slab failed: {message}")
            for site in sites:
                upsert_row(
                    make_failure_row(
                        material=material,
                        site=site,
                        status="clean_failed",
                        message=message,
                    )
                )
            flush_summary()
            continue

        for site in sites:
            case_dir = material_out / site
            ads_atoms_in = read_geometry(GEOMETRY_DIR / material / f"co2_{site}.traj")
            apply_bottom_layer_constraint(ads_atoms_in, adsorbate_atoms=3)

            try:
                with timer(f"DFT adsorption → {material} | {site}"):
                    ads_energy, ads_gap, ads_atoms_opt, ads_profile = optimize_structure(
                        atoms=ads_atoms_in,
                        output_dir=case_dir,
                        label=f"{material}_{site}",
                        ecut=args.ecut,
                        kpts=kpts,
                        fmax=args.fmax,
                        steps=args.steps,
                        profiles=(clean_profile_cfg,),
                    )

                adsorption_energy = ads_energy - clean_energy - clean_co2_energy
                geometry_metrics = adsorption_geometry_metrics(ads_atoms_opt)
                print0(
                    f"[dft] {material} {site} adsorption energy = {adsorption_energy:.6f} eV "
                    f"(profile={ads_profile})"
                )

                upsert_row(
                    {
                        "material": material,
                        "site": site,
                        "dft_status": "ok",
                        "dft_profile": ads_profile,
                        "error_message": "",
                        "clean_slab_energy_ev": clean_energy,
                        "adsorbed_energy_ev": ads_energy,
                        "co2_reference_energy_ev": clean_co2_energy,
                        "adsorption_energy_ev": adsorption_energy,
                        "clean_band_gap_ev": clean_gap,
                        "adsorbed_band_gap_ev": ads_gap,
                        "clean_n_steps": clean_atoms_opt.info["n_steps"],
                        "clean_fmax_final_ev_per_ang": clean_atoms_opt.info["fmax_final"],
                        "adsorbed_n_steps": ads_atoms_opt.info["n_steps"],
                        "adsorbed_fmax_final_ev_per_ang": ads_atoms_opt.info["fmax_final"],
                        **geometry_metrics,
                    }
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print0(f"[dft] {material} {site} failed: {message}")
                upsert_row(
                    {
                        **make_failure_row(
                            material=material,
                            site=site,
                            status="ads_failed",
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


def main() -> None:
    _require_ase_and_gpaw()
    args = parse_args()
    run_dft_workflow(args)


if __name__ == "__main__":
    main()
