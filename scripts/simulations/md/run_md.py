"""Langevin NVT ab initio MD for CO2 on a slab, started from the DFT-optimized structure.

Each site runs one or more independent trajectories (one per seed). Every trajectory
has an equilibration phase (written separately, never used for descriptors) followed
by a production phase. The calculator uses the same GPAW profile (smearing, mixer,
maxiter) that the DFT run used for that site, read from dft_summary.csv.

Parameters are resolved as: command-line flag > methods.md in the run configuration
(RESULTS_DIR/config.yaml, written by run_simulations.py, or --config) > defaults below,
which match configs/simulations/zno_production.yaml.

Each trajectory holds an exclusive lock (seed_<n>/.run_md.lock) while it runs, so a
second run_md on the same seed stops before writing anything. A seed whose production
is complete in the site tables with the same settings is skipped; a seed directory
with an incomplete or different production is rerun only with --overwrite (there is no
mid-trajectory restart).

Random stream (methods.md.rng_stream): "shared" seeds one generator with the seed only,
so every site with the same seed and atom ordering gets the same velocity draw and
Langevin noise sequence; "per_site" derives it from SeedSequence([seed, SITE_RNG_INDEX
[site]]), which decorrelates the sites. Configurations without the key (runs made
before it existed) keep "shared", so their trajectories stay reproducible.
"""

from __future__ import annotations

import os

# Before GPAW (and NumPy's BLAS) is loaded: one OpenMP thread per MPI rank.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import contextlib
import csv
import fcntl
import math
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

# rng_stream when methods.md has no such key (runs made before it existed).
LEGACY_RNG_STREAM = "shared"
RNG_STREAMS = ("shared", "per_site")

# Fixed per-site index for rng_stream "per_site"; never renumber an existing site.
SITE_RNG_INDEX = {"top_metal": 0, "top_oxygen": 1, "bridge": 2, "carbonate": 3, "tridentate": 4}

LOCK_FILENAME = ".run_md.lock"

# Energy drift above which the NVE test suggests a smaller time step (audit M4).
NVE_DRIFT_THRESHOLD_MEV_PER_ATOM_PER_PS = 1.0

# Slab atoms within this distance of the topmost slab atom (initial frame) define the
# top layer; its mean z is the reference plane for the C height.
TOP_LAYER_TOLERANCE_ANG = 1.0

PROFILE_CATALOG = {str(profile["name"]): profile for profile in OPTIMIZATION_PROFILES + CEO2_EXTRA_PROFILES}

FRAME_FIELDS = [
    "seed",
    "rng_stream",
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
        "--overwrite",
        action="store_true",
        help="Rerun seeds whose production is already complete, incomplete or made with other settings.",
    )
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


def broadcast(value):
    from ase.parallel import broadcast as ase_broadcast

    return ase_broadcast(value, root=0, comm=_world())


class TrajectoryLocked(RuntimeError):
    """Another run_md holds the lock of this seed directory."""


@contextlib.contextmanager
def trajectory_lock(seed_dir: Path):
    """Exclusive, non-blocking flock on seed_dir/.run_md.lock for the whole trajectory.

    Rank 0 takes the lock and the outcome is broadcast, so all ranks stop together.
    """
    handle = None
    error = ""
    if is_master():
        seed_dir.mkdir(parents=True, exist_ok=True)
        handle = (seed_dir / LOCK_FILENAME).open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            owner = handle.read().strip() or "unknown"
            handle.close()
            handle = None
            error = (
                f"{seed_dir} is locked by another run_md (PID {owner}, {seed_dir / LOCK_FILENAME}); "
                "two processes must not write the same trajectory."
            )
        else:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
    error = broadcast(error)
    if error:
        raise TrajectoryLocked(error)
    try:
        yield
    finally:
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def make_rng(seed: int, site: str, rng_stream: str):
    if rng_stream == "shared":
        return np.random.default_rng(seed)
    if site not in SITE_RNG_INDEX:
        raise RuntimeError(f"Site '{site}' has no SITE_RNG_INDEX entry for rng_stream=per_site.")
    return np.random.default_rng(np.random.SeedSequence([int(seed), SITE_RNG_INDEX[site]]))


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

    settings["rng_stream"] = str(md.get("rng_stream") or LEGACY_RNG_STREAM)
    if settings["rng_stream"] not in RNG_STREAMS:
        raise SystemExit(f"[md] rng_stream must be one of {RNG_STREAMS}, not '{settings['rng_stream']}'.")

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
                dispersion = str(row.get("dispersion", "") or "none").strip().lower()
                if dispersion not in ("none", "nan"):
                    # The MD calculator is plain PBE; a D3 geometry would start out of equilibrium.
                    raise RuntimeError(
                        f"{material} | {site} was relaxed with dispersion {dispersion!r}; "
                        "the MD supports PBE without dispersion only."
                    )
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
    fieldnames = list(fieldnames or rows[0].keys())
    # Rows of earlier versions may carry columns that are no longer written.
    for row in rows:
        fieldnames += [key for key in row if key not in fieldnames]
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


def run_phase(dyn, atoms, steps, log_interval, timestep_fs, seed, phase, top_layer, traj, step_offset, rng_stream):
    """Run `steps` MD steps, recording a frame every `log_interval` steps (and at step 0)."""
    rows: list[dict] = []

    def record():
        local_step = dyn.nsteps
        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        temp = float(atoms.get_temperature())
        row = {
            "seed": seed,
            "rng_stream": rng_stream,
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


def run_trajectory(input_xyz, out_dir, settings, profile, seed, site, top_layer=None):
    from ase import units
    from ase.io import write
    from ase.io.trajectory import Trajectory
    from ase.md.langevin import Langevin

    out_dir.mkdir(parents=True, exist_ok=True)
    atoms = prepare_atoms(input_xyz)
    top_layer = top_layer_indices(atoms) if top_layer is None else top_layer

    # One generator per trajectory feeds both the initial velocities and the Langevin noise.
    rng = make_rng(seed, site, settings["rng_stream"])
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
            rng_stream=settings["rng_stream"],
        )
        traj.close()
        save_csv(out_dir / "md_equilibration_frames.csv", equilibration_rows, FRAME_FIELDS)
        write(out_dir / "md_equilibrated.xyz", atoms)

    traj = Trajectory(str(out_dir / "md_production.traj"), "w", atoms)
    production_rows = run_phase(
        langevin(), atoms, settings["production_steps"], settings["log_interval"],
        timestep_fs, seed, "production", top_layer, traj, step_offset=settings["equilibration_steps"],
        rng_stream=settings["rng_stream"],
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


class SeedNeedsOverwrite(RuntimeError):
    """A seed directory holds an incomplete production or one made with other settings."""


def recorded_setting_differences(row: dict, settings: dict, profile_name: str) -> list[str]:
    """Columns of an md_site_summary.csv row that differ from the settings of this run."""
    numeric = {
        "timestep_fs": settings["timestep_fs"],
        "friction_per_fs": settings["friction_per_fs"],
        "target_temperature_k": settings["temperature_kelvin"],
        "equilibration_steps": settings["equilibration_steps"],
        "production_steps": settings["production_steps"],
        "ecut_ev": settings["ecut_ev"],
    }
    differences = []
    for column, value in numeric.items():
        try:
            same = math.isclose(float(row.get(column)), float(value), rel_tol=1e-9)
        except (TypeError, ValueError):
            same = False
        if not same:
            differences.append(column)
    if row.get("kpoints") != "x".join(str(k) for k in settings["kpoints"]):
        differences.append("kpoints")
    if row.get("dft_profile") != profile_name:
        differences.append("dft_profile")
    # Rows written before rng_stream existed used the shared stream.
    if (row.get("rng_stream") or LEGACY_RNG_STREAM) != settings["rng_stream"]:
        differences.append("rng_stream")
    return differences


def seed_state(site_dir: Path, seed: int, settings: dict, profile_name: str) -> tuple[str, str]:
    """("absent" | "complete" | "incomplete" | "different", detail) of one seed.

    Complete: one ok row in md_site_summary.csv with the settings of this run and
    production_steps // log_interval + 1 production frames in md_production_frames.csv.
    """
    key = str(seed)
    summary = [row for row in read_rows(site_dir / "md_site_summary.csv") if row.get("seed") == key]
    frames = [
        row for row in read_rows(site_dir / "md_production_frames.csv")
        if row.get("seed") == key and row.get("phase", "production") == "production"
    ]
    seed_dir = site_dir / f"seed_{seed}"
    outputs = [path for path in seed_dir.iterdir() if path.name != LOCK_FILENAME] if seed_dir.is_dir() else []
    if not summary and not frames and not outputs:
        return "absent", ""
    expected = settings["production_steps"] // settings["log_interval"] + 1
    if len(summary) != 1 or summary[0].get("md_status") != "ok" or len(frames) != expected:
        return "incomplete", f"{len(frames)} of {expected} production frames in md_production_frames.csv"
    differences = recorded_setting_differences(summary[0], settings, profile_name)
    if differences:
        return "different", "recorded with different " + ", ".join(differences)
    return "complete", f"{expected} production frames, same settings"


def replace_seed_rows(path: Path, seed: int, rows: list[dict], fieldnames: list[str]) -> None:
    """Rewrite a site table with the rows of `seed` replaced by `rows` (rank 0 only)."""
    if not is_master():
        return
    kept = [row for row in read_rows(path) if row.get("seed") != str(seed)]
    if kept or rows:
        save_csv(path, kept + rows, fieldnames)
    elif path.is_file():
        path.unlink()


def run_site(material: str, site: str, settings: dict, results_dir: Path, overwrite: bool = False) -> list[dict]:
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
        f"ecut={settings['ecut_ev']} eV kpts={settings['kpoints']} rng_stream={settings['rng_stream']}"
    )

    site_dir.mkdir(parents=True, exist_ok=True)
    frames_csv = site_dir / "md_production_frames.csv"
    summary_csv = site_dir / "md_site_summary.csv"
    # Rows of other seeds are kept, so a seed can be added later without dropping the
    # trajectories that already exist.
    summary_rows = []
    problems = []
    for seed in settings["seeds"]:
        seed_dir = site_dir / f"seed_{seed}"
        try:
            with trajectory_lock(seed_dir):
                state, detail = broadcast(seed_state(site_dir, seed, settings, profile["name"]) if is_master() else None)
                if state == "complete" and not overwrite:
                    print0(f"[md] {material} | {site} seed {seed}: complete ({detail}); skipped (--overwrite reruns it).")
                    continue
                if state in ("incomplete", "different") and not overwrite:
                    raise SeedNeedsOverwrite(
                        f"seed {seed}: {seed_dir} holds an {state} production ({detail}); its files were not touched. "
                        "There is no mid-trajectory restart: rerun with --overwrite to recompute this seed."
                    )
                if state != "absent":
                    print0(f"[md] {material} | {site} seed {seed}: {state} ({detail}); recomputed (--overwrite).")
                    # A rerun that fails must not leave the old rows looking complete.
                    replace_seed_rows(frames_csv, seed, [], FRAME_FIELDS)
                    replace_seed_rows(summary_csv, seed, [], [])

                _, production_rows = run_trajectory(input_xyz, seed_dir, settings, profile, seed, site)
                row = {
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
                    "rng_stream": settings["rng_stream"],
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
                    **summarize(production_rows),
                }
                summary_rows.append(row)
                # Written after each seed so finished trajectories are kept if a later one fails.
                replace_seed_rows(frames_csv, seed, production_rows, FRAME_FIELDS)
                replace_seed_rows(summary_csv, seed, [row], list(row))
        except (TrajectoryLocked, SeedNeedsOverwrite) as exc:
            print0(f"[md] {material} | {site}: {exc}")
            problems.append(str(exc))
    if problems:
        raise RuntimeError("; ".join(problems))
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
                    run_site(material, site, settings, results_dir, overwrite=args.overwrite)
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
