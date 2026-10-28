"""Langevin NVT ab initio MD for CO2 on a slab, started from the DFT-optimized structure.

Each site runs one or more independent trajectories (one per seed). Every trajectory
has an equilibration phase (written separately, never used for descriptors) followed
by a production phase. The calculator uses the same GPAW profile (smearing, mixer,
maxiter) that the DFT run used for that site, read from dft_summary.csv.

Parameters are resolved as: command-line flag > methods.md in the run configuration
(RESULTS_DIR/config.yaml, written by run_simulations.py, or --config) > defaults below,
which match configs/simulations/zno_production.yaml.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scripts.common.timer import timer, print0

from scripts.common import project_config
from scripts.simulations.dft.run_dft import (
    CEO2_EXTRA_PROFILES,
    OPTIMIZATION_PROFILES,
    adsorption_geometry_metrics,
    apply_bottom_layer_constraint,
    build_calculator,
)


# Defaults mirror methods.md in configs/simulations/zno_production.yaml.
DEFAULTS = {
    "temperature_kelvin": 300.0,
    "timestep_fs": 1.0,
    "log_interval": 20,
    "ecut_ev": 500.0,
    "kpoints": [2, 3, 1],
    "friction_per_fs": 0.01,
    "equilibration_steps": 500,
    "production_steps": 1500,
    "seed": 1234,
    "seeds": None,
    "nve_test_steps": 200,
    "nve_test_timesteps_fs": [0.5, 1.0],
}

# Energy drift above which the NVE test suggests a smaller time step (audit M4).
NVE_DRIFT_THRESHOLD_MEV_PER_ATOM_PER_PS = 1.0

# Slab atoms within this distance of the topmost slab atom (initial frame) define the
# top layer; its mean z is the reference plane for the C height.
TOP_LAYER_TOLERANCE_ANG = 1.0

PROFILE_CATALOG = {str(profile["name"]): profile for profile in OPTIMIZATION_PROFILES + CEO2_EXTRA_PROFILES}

FRAME_FIELDS = [
    "seed",
    "phase",
    "step",
    "time_fs",
    "temperature_k",
    "epot_ev",
    "ekin_ev",
    "etot_ev",
    "co_bond_1_ang",
    "co_bond_2_ang",
    "co_bond_avg_ang",
    "oco_angle_deg",
    "c_height_above_slab_plane_ang",
    "c_surface_distance_ang",
    "c_min_slab_distance_ang",
    "o_min_metal_distance_ang",
]

# Columns averaged over the production frames in md_site_summary.csv.
SUMMARY_DESCRIPTORS = [
    "temperature_k",
    "epot_ev",
    "co_bond_avg_ang",
    "oco_angle_deg",
    "c_height_above_slab_plane_ang",
    "c_min_slab_distance_ang",
    "o_min_metal_distance_ang",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Molecular Dynamics for ZnO + CO2 adsorption system.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--site", "--sites", dest="sites", nargs="+", default=["bridge"])
    parser.add_argument("--config", type=Path, help="YAML with methods.md (default: RESULTS_DIR/config.yaml).")
    parser.add_argument("--temperature", type=float, help="Target temperature in K.")
    parser.add_argument(
        "--steps",
        type=int,
        help="Deprecated alias of --production-steps (used only when production_steps is not set).",
    )
    parser.add_argument("--equilibration-steps", type=int)
    parser.add_argument("--production-steps", type=int)
    parser.add_argument("--timestep", type=float, help="Time step in fs.")
    parser.add_argument("--friction", type=float, help="Langevin friction in 1/fs (converted with ase.units.fs).")
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--ecut", type=float, help="Plane-wave cutoff (eV); must match the DFT run.")
    parser.add_argument("--kpts", nargs=3, type=int, help="k-point mesh; must match the DFT run.")
    parser.add_argument("--seed", type=int, help="Seed for velocities and Langevin noise.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Several seeds: one independent trajectory each.")
    parser.add_argument(
        "--nve-test",
        action="store_true",
        help="Only run the NVE (VelocityVerlet) time-step test and report the energy drift.",
    )
    parser.add_argument("--nve-test-steps", type=int)
    parser.add_argument("--nve-test-timesteps", nargs="+", type=float, help="Time steps (fs) for --nve-test.")
    return parser.parse_args(argv)


def _world():
    # Lazy import: GPAW is only present on the simulation machine.
    try:
        from gpaw.mpi import world
    except ImportError:
        from ase.parallel import world
    return world


def is_master() -> bool:
    return _world().rank == 0


def load_run_config(path: Path | None) -> dict:
    if path is None:
        path = project_config.RESULTS_DIR / "config.yaml"
        if not path.is_file():
            return {}
    elif not path.is_file():
        raise SystemExit(f"[md] Configuration file does not exist: {path}")
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    print0(f"[md] configuration: {path}")
    return payload


def resolve_settings(args, config: dict) -> dict:
    md = (config.get("methods") or {}).get("md") or {}

    def pick(cli_value, key):
        if cli_value is not None:
            return cli_value
        if md.get(key) is not None:
            return md[key]
        return DEFAULTS[key]

    settings = {
        "temperature_kelvin": float(pick(args.temperature, "temperature_kelvin")),
        "timestep_fs": float(pick(args.timestep, "timestep_fs")),
        "log_interval": int(pick(args.log_interval, "log_interval")),
        "ecut_ev": float(pick(args.ecut, "ecut_ev")),
        "kpoints": tuple(int(k) for k in pick(args.kpts, "kpoints")),
        "friction_per_fs": float(pick(args.friction, "friction_per_fs")),
        "equilibration_steps": int(pick(args.equilibration_steps, "equilibration_steps")),
        "nve_test_steps": int(pick(args.nve_test_steps, "nve_test_steps")),
        "nve_test_timesteps_fs": [float(dt) for dt in pick(args.nve_test_timesteps, "nve_test_timesteps_fs")],
    }

    # production_steps replaces the old total `steps`; `steps` is kept as a fallback
    # so older configurations and the current workflow command line keep working.
    legacy_steps = args.steps if args.steps is not None else md.get("steps")
    explicit_production = args.production_steps if args.production_steps is not None else md.get("production_steps")
    if explicit_production is not None:
        production = int(explicit_production)
        if legacy_steps is not None and int(legacy_steps) != production:
            print0(f"[md] warning: steps={legacy_steps} is ignored; production_steps={production} is used.")
    elif legacy_steps is not None:
        production = int(legacy_steps)
        print0("[md] warning: 'steps' is deprecated; it is used as production_steps.")
    else:
        production = int(DEFAULTS["production_steps"])
    settings["production_steps"] = production

    seeds = args.seeds if args.seeds is not None else md.get("seeds")
    if seeds:
        settings["seeds"] = [int(seed) for seed in seeds]
    else:
        settings["seeds"] = [int(pick(args.seed, "seed"))]
    if len(set(settings["seeds"])) != len(settings["seeds"]):
        raise SystemExit(f"[md] Seeds must be distinct: {settings['seeds']}")

    for key in ("timestep_fs", "friction_per_fs", "temperature_kelvin"):
        if settings[key] <= 0.0:
            raise SystemExit(f"[md] {key} must be positive.")
    for key in ("log_interval", "production_steps", "nve_test_steps"):
        if settings[key] <= 0:
            raise SystemExit(f"[md] {key} must be a positive integer.")
    if settings["equilibration_steps"] < 0:
        raise SystemExit("[md] equilibration_steps must not be negative.")

    # M1: the MD must use the same plane-wave cutoff and k-points as the DFT relaxation.
    dft = (config.get("methods") or {}).get("dft") or {}
    if dft.get("ecut_ev") is not None and float(dft["ecut_ev"]) != settings["ecut_ev"]:
        raise SystemExit(f"[md] ecut {settings['ecut_ev']} eV differs from the DFT run ({dft['ecut_ev']} eV).")
    if dft.get("kpoints") is not None and tuple(int(k) for k in dft["kpoints"]) != settings["kpoints"]:
        raise SystemExit(f"[md] k-points {settings['kpoints']} differ from the DFT run ({dft['kpoints']}).")
    return settings


def read_dft_row(results_dir: Path, material: str, site: str) -> dict:
    path = results_dir / "dft_summary.csv"
    if not path.is_file():
        raise RuntimeError(f"Missing DFT summary: {path}. Run DFT first.")
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("material") == material and row.get("site") == site:
                return row
    raise RuntimeError(f"No row for {material} | {site} in {path}.")


def dft_profile_for(row: dict) -> dict:
    name = str(row.get("dft_profile", ""))
    if name not in PROFILE_CATALOG:
        raise RuntimeError(f"Unknown dft_profile '{name}' in dft_summary.csv.")
    return PROFILE_CATALOG[name]


def top_layer_indices(atoms, adsorbate_atoms: int = 3) -> list[int]:
    slab_indices = np.arange(len(atoms) - adsorbate_atoms)
    slab_z = atoms.positions[slab_indices, 2]
    return [int(i) for i in slab_indices if atoms.positions[i, 2] >= slab_z.max() - TOP_LAYER_TOLERANCE_ANG]


def frame_descriptors(atoms, top_layer: list[int]) -> dict[str, float]:
    """Minimum-image CO2 geometry; assumes the last three atoms are O, C, O."""
    metrics = adsorption_geometry_metrics(atoms)
    c_idx = len(atoms) - 2
    plane_z = float(np.mean(atoms.positions[top_layer, 2]))
    return {
        "co_bond_1_ang": metrics["co_bond_1_ang"],
        "co_bond_2_ang": metrics["co_bond_2_ang"],
        "co_bond_avg_ang": 0.5 * (metrics["co_bond_1_ang"] + metrics["co_bond_2_ang"]),
        "oco_angle_deg": metrics["oco_angle_deg"],
        # Mean z of the top layer (fixed set of atoms) is less noisy than the slab z maximum.
        "c_height_above_slab_plane_ang": float(atoms.positions[c_idx, 2]) - plane_z,
        "c_surface_distance_ang": metrics["c_surface_distance_ang"],
        "c_min_slab_distance_ang": metrics["c_min_slab_distance_ang"],
        "o_min_metal_distance_ang": metrics["o_min_metal_distance_ang"],
    }


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows or not is_master():
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def initialize_velocities(atoms, temperature_k: float, rng) -> None:
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, force_temperature

    # The Maxwell-Boltzmann draw is broadcast from rank 0, so all MPI ranks agree.
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    Stationary(atoms)

    # Remove momenta from fixed atoms
    momenta = atoms.get_momenta()
    for constraint in atoms.constraints:
        if hasattr(constraint, "get_indices"):
            momenta[constraint.get_indices()] = 0.0
    atoms.set_momenta(momenta)

    # Rescale temperature after removing constrained atom velocities
    force_temperature(atoms, temperature=temperature_k, unit="K")


def prepare_atoms(input_xyz: Path):
    from ase.io import read

    atoms = read(input_xyz)
    apply_bottom_layer_constraint(atoms, adsorbate_atoms=3)
    return atoms


def run_phase(dyn, atoms, steps, log_interval, timestep_fs, seed, phase, top_layer, traj, step_offset):
    """Run `steps` MD steps, recording a frame every `log_interval` steps (and at step 0)."""
    rows: list[dict] = []

    def record():
        local_step = dyn.nsteps
        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        temp = float(atoms.get_temperature())
        row = {
            "seed": seed,
            "phase": phase,
            "step": step_offset + local_step,
            "time_fs": local_step * timestep_fs,
            "temperature_k": temp,
            "epot_ev": epot,
            "ekin_ev": ekin,
            "etot_ev": epot + ekin,
            **frame_descriptors(atoms, top_layer),
        }
        rows.append(row)
        traj.write(atoms)
        print0(
            f"[md] seed={seed} {phase:13s} t={row['time_fs']:8.1f} fs "
            f"Epot={epot:12.6f} eV T={temp:8.2f} K "
            f"C-plane={row['c_height_above_slab_plane_ang']:7.3f} A "
            f"C-slab={row['c_min_slab_distance_ang']:7.3f} A OCO={row['oco_angle_deg']:7.2f} deg"
        )

    dyn.attach(record, interval=log_interval)
    dyn.run(steps)
    return rows


def run_trajectory(input_xyz, out_dir, settings, profile, seed, top_layer=None):
    from ase import units
    from ase.io import write
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin

    out_dir.mkdir(parents=True, exist_ok=True)
    atoms = prepare_atoms(input_xyz)
    top_layer = top_layer_indices(atoms) if top_layer is None else top_layer

    # One generator per trajectory feeds both the initial velocities and the Langevin noise.
    rng = np.random.default_rng(seed)
    initialize_velocities(atoms, settings["temperature_kelvin"], rng)

    atoms.calc = build_calculator(
        ecut=settings["ecut_ev"],
        kpts=settings["kpoints"],
        txt_file=out_dir / "md_gpaw.txt",
        mode=str(profile["mode"]),
        smearing=float(profile["smearing"]),
        maxiter=int(profile["maxiter"]),
        mixer=profile.get("mixer"),
    )

    timestep_fs = settings["timestep_fs"]
    # ASE expects friction in inverse ASE time units: 0.01 / units.fs is 0.01 fs^-1 (tau = 100 fs).
    friction = settings["friction_per_fs"] / units.fs

    def langevin():
        return Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=settings["temperature_kelvin"],
            friction=friction,
            fixcm=False,
            rng=rng,
        )

    equilibration_rows: list[dict] = []
    if settings["equilibration_steps"] > 0:
        traj = Trajectory(str(out_dir / "md_equilibration.traj"), "w", atoms)
        equilibration_rows = run_phase(
            langevin(), atoms, settings["equilibration_steps"], settings["log_interval"],
            timestep_fs, seed, "equilibration", top_layer, traj, step_offset=0,
        )
        traj.close()
        save_csv(out_dir / "md_equilibration_frames.csv", equilibration_rows, FRAME_FIELDS)
        write(out_dir / "md_equilibrated.xyz", atoms)

    traj = Trajectory(str(out_dir / "md_production.traj"), "w", atoms)
    production_rows = run_phase(
        langevin(), atoms, settings["production_steps"], settings["log_interval"],
        timestep_fs, seed, "production", top_layer, traj, step_offset=settings["equilibration_steps"],
    )
    traj.close()
    save_csv(out_dir / "md_production_frames.csv", production_rows, FRAME_FIELDS)

    write(out_dir / "md_final.xyz", atoms)
    write(out_dir / "md_final.cif", atoms)
    return equilibration_rows, production_rows


def summarize(production_rows: list[dict]) -> dict:
    summary = {"n_production_frames": len(production_rows)}
    for key in SUMMARY_DESCRIPTORS:
        values = np.array([float(row[key]) for row in production_rows])
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return summary


def force_evaluations_per_trajectory(settings: dict) -> int:
    # One evaluation per step plus the initial one (later run() calls reuse cached forces).
    return settings["equilibration_steps"] + settings["production_steps"] + 1


def run_site(material: str, site: str, settings: dict, results_dir: Path) -> list[dict]:
    input_xyz = results_dir / "dft" / material / site / f"{material}_{site}_optimized.xyz"
    site_dir = results_dir / "md" / material / site

    dft_row = read_dft_row(results_dir, material, site)
    if dft_row.get("dft_status") != "ok":
        raise RuntimeError(
            f"DFT status is '{dft_row.get('dft_status')}' for {material} | {site}; MD skipped "
            "(it must start from a converged DFT structure)."
        )
    profile = dft_profile_for(dft_row)
    if not input_xyz.exists():
        raise RuntimeError(f"Missing optimized DFT structure: {input_xyz}. Run DFT first for {material} | {site}.")

    print0(
        f"[md] {material} | {site}: profile={profile['name']} smearing={profile['smearing']} eV "
        f"ecut={settings['ecut_ev']} eV kpts={settings['kpoints']}"
    )

    site_dir.mkdir(parents=True, exist_ok=True)
    # Seeds from earlier invocations are kept, so a seed can be added later without
    # dropping the trajectories that already exist; the seeds run now are recomputed.
    current = {str(seed) for seed in settings["seeds"]}
    kept_summary = [r for r in read_rows(site_dir / "md_site_summary.csv") if r.get("seed") not in current]
    kept_frames = [r for r in read_rows(site_dir / "md_production_frames.csv") if r.get("seed") not in current]
    if kept_summary:
        print0(f"[md] {material} | {site}: keeping earlier seeds {sorted({r['seed'] for r in kept_summary})}")
    summary_rows = []
    all_production_rows = []
    for seed in settings["seeds"]:
        _, production_rows = run_trajectory(input_xyz, site_dir / f"seed_{seed}", settings, profile, seed)
        all_production_rows.extend(production_rows)
        summary_rows.append(
            {
                "material": material,
                "site": site,
                "seed": seed,
                "md_status": "ok",
                "dft_profile": profile["name"],
                "smearing_ev": profile["smearing"],
                "ecut_ev": settings["ecut_ev"],
                "kpoints": "x".join(str(k) for k in settings["kpoints"]),
                "target_temperature_k": settings["temperature_kelvin"],
                "timestep_fs": settings["timestep_fs"],
                "friction_per_fs": settings["friction_per_fs"],
                "equilibration_steps": settings["equilibration_steps"],
                "production_steps": settings["production_steps"],
                "production_time_fs": settings["production_steps"] * settings["timestep_fs"],
                "force_evaluations": force_evaluations_per_trajectory(settings),
                **summarize(production_rows),
            }
        )
        # Rewritten after each seed so finished trajectories are kept if a later one fails.
        save_csv(site_dir / "md_production_frames.csv", kept_frames + all_production_rows, FRAME_FIELDS)
        save_csv(site_dir / "md_site_summary.csv", kept_summary + summary_rows, list(summary_rows[0]))
    return summary_rows


def run_nve_test(material: str, site: str, settings: dict, results_dir: Path) -> list[dict]:
    """Short NVE runs (VelocityVerlet) to check energy conservation for each time step (audit M4)."""
    from ase import units
    from ase.md.verlet import VelocityVerlet

    dft_row = read_dft_row(results_dir, material, site)
    if dft_row.get("dft_status") != "ok":
        raise RuntimeError(f"DFT status is '{dft_row.get('dft_status')}' for {material} | {site}; NVE test skipped.")
    profile = dft_profile_for(dft_row)
    input_xyz = results_dir / "dft" / material / site / f"{material}_{site}_optimized.xyz"
    if not input_xyz.exists():
        raise RuntimeError(f"Missing optimized DFT structure: {input_xyz}.")

    out_dir = results_dir / "md" / material / site / "nve_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = settings["seeds"][0]
    steps = settings["nve_test_steps"]
    summary_rows, frame_rows = [], []

    for timestep_fs in settings["nve_test_timesteps_fs"]:
        atoms = prepare_atoms(input_xyz)
        initialize_velocities(atoms, settings["temperature_kelvin"], np.random.default_rng(seed))
        atoms.calc = build_calculator(
            ecut=settings["ecut_ev"],
            kpts=settings["kpoints"],
            txt_file=out_dir / f"nve_dt{timestep_fs:g}fs_gpaw.txt",
            mode=str(profile["mode"]),
            smearing=float(profile["smearing"]),
            maxiter=int(profile["maxiter"]),
            mixer=profile.get("mixer"),
        )
        dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
        times, energies = [], []

        def record():
            epot = float(atoms.get_potential_energy())
            ekin = float(atoms.get_kinetic_energy())
            times.append(dyn.nsteps * timestep_fs)
            energies.append(epot + ekin)
            frame_rows.append(
                {
                    "timestep_fs": timestep_fs,
                    "step": dyn.nsteps,
                    "time_fs": dyn.nsteps * timestep_fs,
                    "temperature_k": float(atoms.get_temperature()),
                    "epot_ev": epot,
                    "ekin_ev": ekin,
                    "etot_ev": epot + ekin,
                }
            )

        dyn.attach(record, interval=1)
        dyn.run(steps)

        n_atoms = len(atoms)
        etot_per_atom_mev = (np.array(energies) - energies[0]) / n_atoms * 1000.0
        times_ps = np.array(times) / 1000.0
        drift = float(np.polyfit(times_ps, etot_per_atom_mev, 1)[0])
        verdict = "ok" if abs(drift) <= NVE_DRIFT_THRESHOLD_MEV_PER_ATOM_PER_PS else "too_large"
        summary_rows.append(
            {
                "material": material,
                "site": site,
                "seed": seed,
                "dft_profile": profile["name"],
                "timestep_fs": timestep_fs,
                "steps": steps,
                "simulated_time_fs": steps * timestep_fs,
                "n_atoms": n_atoms,
                "force_evaluations": steps + 1,
                "drift_mev_per_atom_per_ps": drift,
                "max_abs_deviation_mev_per_atom": float(np.abs(etot_per_atom_mev).max()),
                "threshold_mev_per_atom_per_ps": NVE_DRIFT_THRESHOLD_MEV_PER_ATOM_PER_PS,
                "verdict": verdict,
            }
        )
        print0(
            f"[md:nve] {material} | {site} dt={timestep_fs:g} fs: drift={drift:.3f} meV/atom/ps "
            f"(threshold {NVE_DRIFT_THRESHOLD_MEV_PER_ATOM_PER_PS}) -> {verdict}"
        )
        save_csv(out_dir / "nve_test_frames.csv", frame_rows)
        save_csv(out_dir / "nve_test_summary.csv", summary_rows)

    return summary_rows


def run_md(args) -> int:
    results_dir = project_config.RESULTS_DIR
    settings = resolve_settings(args, load_run_config(args.config))
    material = args.material
    sites = list(dict.fromkeys(args.sites))

    if args.nve_test:
        per_trajectory = settings["nve_test_steps"] + 1
        total = per_trajectory * len(settings["nve_test_timesteps_fs"]) * len(sites)
        print0(
            f"[md:nve] planned force evaluations: {per_trajectory} per time step x "
            f"{len(settings['nve_test_timesteps_fs'])} time steps x {len(sites)} site(s) = {total}"
        )
    else:
        per_trajectory = force_evaluations_per_trajectory(settings)
        total = per_trajectory * len(settings["seeds"]) * len(sites)
        print0(
            f"[md] equilibration={settings['equilibration_steps']} + production={settings['production_steps']} steps "
            f"at {settings['timestep_fs']} fs, friction={settings['friction_per_fs']} 1/fs, "
            f"seeds={settings['seeds']}"
        )
        print0(
            f"[md] planned force evaluations: {per_trajectory} per trajectory x "
            f"{len(settings['seeds'])} seed(s) x {len(sites)} site(s) = {total}"
        )

    failures = []
    for site in sites:
        try:
            with timer(f"MD | Material={material} | Site={site}"):
                if args.nve_test:
                    run_nve_test(material, site, settings, results_dir)
                else:
                    run_site(material, site, settings, results_dir)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print0(f"[md] {material} | {site} failed: {message}")
            failures.append((site, message))

    if failures:
        print0(f"[md] {len(failures)} of {len(sites)} site(s) failed: {[site for site, _ in failures]}")
        return 1
    print0(f"[md] Done. Results saved in: {results_dir / 'md' / material}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    raise SystemExit(run_md(args))


if __name__ == "__main__":
    main()
