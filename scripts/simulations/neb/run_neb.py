from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.common.timer import timer, print0
from scripts.common import project_config
from scripts.common.project_config import ADSORPTION_SITES, RESULTS_DIR, ensure_directories
from scripts.simulations.dft import run_dft


NEB_RESULTS_DIR = RESULTS_DIR / "neb"
PROJECT_CONFIG_PATH = project_config.ROOT_DIR / "configs" / "simulations" / "zno_production.yaml"

DEFAULT_SITES = list(ADSORPTION_SITES)

# Elementary hops only. On the 2x1 cell the three sites are collinear along y
# (top_metal y=2.60, bridge y=3.60, top_oxygen y=4.59 A), so top_metal -> top_oxygen
# passes over bridge and is a composite path.
DEFAULT_PAIRS = (("top_metal", "bridge"), ("bridge", "top_oxygen"))

DEFAULT_FMAX_CI = 0.05
DEFAULT_DFT_FMAX = 0.05
ADSORBATE_ATOMS = 3  # CO2 = last three atoms (O, C, O)
# The endpoint single points run on all MPI ranks (no image sub-communicator) with the
# image calculator settings; a fresh SCF still does not reproduce the DFT energy to the
# last meV (SCF convergence is per electron), so the provenance check needs some slack.
DEFAULT_ENDPOINT_ENERGY_TOL_EV = 0.02
DEFAULT_ENDPOINT_FORCE_FACTOR = 1.5
MIN_ENDPOINT_DISPLACEMENT_ANG = 0.3
# Smallest forward barrier (eV) reported as resolved; methods.neb.barrier_resolution_ev.
DEFAULT_BARRIER_RESOLUTION_EV = 0.03

SUMMARY_JSON = "neb_run_summary.json"
FINAL_IMAGES_TRAJ = "neb_final_images.traj"
PATHS_SUMMARY_CSV = "neb_paths_summary.csv"

STATUS_CONVERGED = "converged"
STATUS_NOT_CONVERGED = "max_steps_reached"
STATUS_IDENTICAL_ENDPOINTS = "skipped_identical_endpoints"
STATUS_PROFILE_MISMATCH = "skipped_profile_mismatch"
STATUS_DFT_NOT_OK = "aborted_dft_not_ok"
STATUS_MISSING_INPUT = "aborted_missing_input"
STATUS_INCOMPATIBLE = "aborted_incompatible_endpoints"
STATUS_PROVENANCE = "aborted_provenance"
STATUS_ENDPOINT_FORCES = "aborted_endpoint_forces"
STATUS_INVALID_ENERGIES = "aborted_invalid_energies"

# Two identical minima leave no path to compute; that is a result, not a failure.
NON_FAILING_STATUSES = {STATUS_CONVERGED, STATUS_IDENTICAL_ENDPOINTS}

PATHS_SUMMARY_FIELDS = [
    "material",
    "initial_site",
    "final_site",
    "path",
    "status",
    "converged",
    "dft_profile",
    "internal_images",
    "total_images",
    "fmax",
    "fmax_ci",
    "max_steps",
    "n_steps_plain",
    "n_steps_ci",
    "climb_activated",
    "converged_plain",
    "converged_ci",
    "ecut",
    "kpts",
    "interpolation",
    "optimizer",
    "parallel_images",
    "mpi_size",
    "forward_barrier_eV",
    "reverse_barrier_eV",
    "reaction_energy_eV",
    "highest_image_index",
    "highest_internal_image_index",
    "no_internal_saddle",
    "barrier_resolution_eV",
    "barrier_resolved",
    "message",
]


class PairAborted(Exception):
    """A pair that cannot be run; ``status`` is written to its summary."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run NEB calculations between ZnO + CO2 adsorption sites.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument(
        "--sites",
        nargs="*",
        default=DEFAULT_SITES,
        help="Only pairs whose two sites are both listed here are run.",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        metavar="INITIAL:FINAL",
        help="Explicit pathways, e.g. top_metal:bridge. Default: methods.neb.pairs or adjacent sites.",
    )
    parser.add_argument("--initial-site", default=None,)
    parser.add_argument("--final-site", default=None,)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML configuration. Default: <results>/config.yaml, else the production YAML.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Run results directory holding dft_summary.csv and dft/. Default: QML_ZNO_RESULTS_DIR.",
    )
    parser.add_argument("--images", type=int, default=3,)
    parser.add_argument("--fmax", type=float, default=0.08, help="Force threshold of the plain NEB phase.")
    parser.add_argument(
        "--fmax-ci",
        type=float,
        default=None,
        help=f"Force threshold of the climbing-image phase (default: YAML or {DEFAULT_FMAX_CI}).",
    )
    parser.add_argument(
        "--dft-fmax",
        type=float,
        default=None,
        help=f"DFT force threshold the endpoints must satisfy (default: YAML or {DEFAULT_DFT_FMAX}).",
    )
    parser.add_argument("--steps", type=int, default=300, help="Total optimizer step budget (both phases).")
    parser.add_argument("--ecut", type=float, default=500.0)
    parser.add_argument("--kpts", nargs=3, type=int, default=(2, 3, 1), metavar=("KX", "KY", "KZ"),)
    parser.add_argument("--interpolation", choices=["linear", "idpp"], default="idpp",)
    parser.add_argument(
        "--climb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a climbing-image phase after the plain phase (default: on).",
    )
    parser.add_argument("--optimizer", choices=["FIRE", "BFGS"], default="FIRE",)
    parser.add_argument(
        "--climb-start-fraction",
        type=float,
        default=0.30,
        help="Fraction of --steps given to the plain phase; the CI phase gets the remaining steps.",
    )
    parser.add_argument("--parallel-images", action="store_true",)
    parser.add_argument("--overwrite", action="store_true",)

    return parser.parse_args(argv)


def get_world():
    try:
        from gpaw.mpi import world
    except ImportError:
        from ase.parallel import world
    return world


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# CONFIGURATION
def load_config(config_path: Path | None, results_dir: Path) -> dict:
    if config_path is not None:
        candidates = [Path(config_path)]
    else:
        # The workflow copies the YAML into the run directory.
        candidates = [results_dir / "config.yaml", PROJECT_CONFIG_PATH]

    for candidate in candidates:
        if candidate.is_file():
            import yaml

            with candidate.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            if not isinstance(payload, dict):
                raise SystemExit(f"Configuration must contain a YAML mapping: {candidate}")
            payload["_config_path"] = str(candidate)
            return payload

    if config_path is not None:
        raise SystemExit(f"Configuration file does not exist: {config_path}")
    return {}


def method_settings(config: dict, method: str) -> dict:
    methods = config.get("methods") or {}
    settings = methods.get(method) or {}
    return settings if isinstance(settings, dict) else {}


def parse_pair(value) -> tuple[str, str]:
    if isinstance(value, str):
        parts = value.replace("->", ":").split(":")
    else:
        parts = list(value)
    parts = [str(part).strip() for part in parts]

    if len(parts) != 2 or not all(parts):
        raise SystemExit(f"Invalid NEB pair {value!r}; use INITIAL:FINAL or [INITIAL, FINAL].")

    initial_site, final_site = parts
    unknown = [site for site in parts if site not in ADSORPTION_SITES]
    if unknown:
        raise SystemExit(f"Unknown NEB site(s) {unknown} in pair {value!r}.")
    if initial_site == final_site:
        raise SystemExit(f"Initial and final sites must be different: {value!r}.")

    return initial_site, final_site


def get_pairs(args, config: dict | None = None) -> list[tuple[str, str]]:
    if args.initial_site or args.final_site:
        if not args.initial_site or not args.final_site:
            raise SystemExit("Use both --initial-site and --final-site together.")

        return [parse_pair((args.initial_site, args.final_site))]

    if args.pairs:
        pairs = [parse_pair(item) for item in args.pairs]
    else:
        configured = method_settings(config or {}, "neb").get("pairs")
        if configured:
            if not isinstance(configured, list):
                raise SystemExit("methods.neb.pairs must be a list of [initial, final] pairs.")
            pairs = [parse_pair(item) for item in configured]
        else:
            pairs = list(DEFAULT_PAIRS)

    selected_sites = set(args.sites or [])
    kept = [pair for pair in pairs if set(pair) <= selected_sites]
    dropped = [pair for pair in pairs if pair not in kept]
    if dropped:
        print0(f"[neb] Pairs outside --sites are not run: {dropped}")

    if not kept:
        raise SystemExit("No NEB pair left to run; check --sites and methods.neb.pairs.")

    return list(dict.fromkeys(kept))


def resolve_settings(args, config: dict) -> None:
    neb_cfg = method_settings(config, "neb")
    dft_cfg = method_settings(config, "dft")

    if args.fmax_ci is None:
        args.fmax_ci = float(neb_cfg.get("fmax_ci_ev_per_angstrom", DEFAULT_FMAX_CI))
    if args.dft_fmax is None:
        args.dft_fmax = float(dft_cfg.get("fmax_ev_per_angstrom", DEFAULT_DFT_FMAX))
    args.endpoint_energy_tol_ev = float(neb_cfg.get("endpoint_energy_tol_ev", DEFAULT_ENDPOINT_ENERGY_TOL_EV))
    args.barrier_resolution_ev = float(neb_cfg.get("barrier_resolution_ev", DEFAULT_BARRIER_RESOLUTION_EV))
    if args.barrier_resolution_ev < 0.0:
        raise SystemExit("methods.neb.barrier_resolution_ev must be non-negative.")
    args.endpoint_force_tol = args.dft_fmax * float(
        neb_cfg.get("endpoint_force_factor", DEFAULT_ENDPOINT_FORCE_FACTOR)
    )

    if args.images < 1:
        raise SystemExit("--images must be at least 1.")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1.")
    if not 0.0 <= args.climb_start_fraction <= 1.0:
        raise SystemExit("--climb-start-fraction must be between 0 and 1.")
    if args.climb and plain_step_budget(args.steps, args.climb_start_fraction, True) >= args.steps:
        raise SystemExit(
            "--climb-start-fraction leaves no step budget for the climbing-image phase; use a value below 1."
        )

    # Images must use the DFT settings of the endpoints.
    if dft_cfg:
        dft_ecut = dft_cfg.get("ecut_ev")
        dft_kpts = dft_cfg.get("kpoints")
        if dft_ecut is not None and not math.isclose(float(dft_ecut), float(args.ecut)):
            raise SystemExit(f"NEB ecut {args.ecut} eV differs from methods.dft.ecut_ev {dft_ecut} eV.")
        if dft_kpts is not None and [int(v) for v in dft_kpts] != [int(v) for v in args.kpts]:
            raise SystemExit(f"NEB kpts {list(args.kpts)} differ from methods.dft.kpoints {list(dft_kpts)}.")


def check_parallel_layout(internal_images: int, world_size: int, parallel_images: bool) -> None:
    """Image parallelism gives each internal image world_size // internal_images ranks."""
    if parallel_images and world_size > 1 and world_size % internal_images != 0:
        raise SystemExit(
            f"Parallel NEB over images requires the number of MPI processes to be divisible by "
            f"the number of internal images: {world_size} % {internal_images} != 0. "
            f"Use --images that divides the process count (e.g. 3 images with 6 processes) "
            f"or run without --parallel-images."
        )


# BARRIER
def barrier_metrics(energies, resolution_ev: float = DEFAULT_BARRIER_RESOLUTION_EV) -> dict | None:
    """Barriers from the highest energy of the whole band, endpoints included.

    E_max = max over all images, Ea_fwd = E_max - E_IS and Ea_rev = E_max - E_FS, so
    Ea_fwd >= 0 and Ea_rev >= 0 always hold and, for a downhill path without a saddle,
    Ea_fwd = 0 and Ea_rev = |dE|. ``highest_image_index`` is the image of E_max (an
    endpoint when there is no internal maximum). ``no_internal_saddle`` is True when the
    highest internal image is not above both endpoints (monotonic profile).
    ``barrier_resolved`` is True only for an internal saddle with Ea_fwd above
    ``resolution_ev``; smaller values are below what a 3-image band and the SCF/fmax
    tolerances can resolve and are reported as no resolved barrier.

    Returns None when there is no internal image or an energy is not finite.
    """
    energies = [float(value) for value in energies]

    if len(energies) < 3 or not all(math.isfinite(value) for value in energies):
        return None

    e_is = energies[0]
    e_fs = energies[-1]
    internal = energies[1:-1]
    highest_internal = 1 + int(np.argmax(internal))
    e_internal_max = energies[highest_internal]
    highest = int(np.argmax(energies))
    e_max = energies[highest]

    no_internal_saddle = not (e_internal_max > e_is and e_internal_max > e_fs)
    forward = e_max - e_is

    return {
        "energies": energies,
        "relative_energies": [value - e_is for value in energies],
        "forward_barrier_eV": forward,
        "reverse_barrier_eV": e_max - e_fs,
        "reaction_energy_eV": e_fs - e_is,
        "highest_image_index": highest,
        "highest_internal_image_index": highest_internal,
        "no_internal_saddle": no_internal_saddle,
        "barrier_resolution_eV": float(resolution_ev),
        "barrier_resolved": bool(forward > float(resolution_ev) and not no_internal_saddle),
        "n_total_images": len(energies),
        "n_internal_images": len(energies) - 2,
    }


# DFT INPUTS
def read_dft_summary(results_dir: Path) -> dict[tuple[str, str], dict]:
    path = results_dir / "dft_summary.csv"

    if not path.exists():
        raise SystemExit(f"Missing DFT summary (run DFT first): {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (str(row.get("material", "")), str(row.get("site", ""))): row
            for row in csv.DictReader(handle)
        }


def dft_structure_path(results_dir: Path, material: str, site: str) -> Path:
    return results_dir / "dft" / material / site / f"{material}_{site}_optimized.xyz"


def fixed_indices(atoms) -> list[int]:
    from ase.constraints import FixAtoms

    indices: set[int] = set()
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices.update(int(i) for i in constraint.get_indices())
    return sorted(indices)


def read_endpoint(path: Path):
    from ase.io import read

    atoms = read(path, parallel=False)
    # extxyz keeps the DFT FixAtoms as move_mask; rebuild it only if absent.
    if not fixed_indices(atoms):
        run_dft.apply_bottom_layer_constraint(atoms, adsorbate_atoms=ADSORBATE_ATOMS)
    atoms.calc = None
    return atoms


def check_compatible_endpoints(initial, final, initial_site: str, final_site: str):
    if len(initial) != len(final):
        raise PairAborted(
            STATUS_INCOMPATIBLE,
            f"Initial and final structures have different atom counts: "
            f"{initial_site}={len(initial)}, {final_site}={len(final)}",
        )

    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise PairAborted(
            STATUS_INCOMPATIBLE,
            "Initial and final structures have different atom ordering. "
            "NEB requires the same atoms in the same order.",
        )

    if fixed_indices(initial) != fixed_indices(final):
        raise PairAborted(
            STATUS_INCOMPATIBLE,
            f"Initial and final structures have different FixAtoms indices: "
            f"{fixed_indices(initial)} vs {fixed_indices(final)}",
        )

    if not np.allclose(initial.cell.array, final.cell.array, atol=1e-6):
        raise PairAborted(STATUS_INCOMPATIBLE, "Initial and final structures have different cells.")


def adsorbate_displacement(initial, final, adsorbate_atoms: int = ADSORBATE_ATOMS) -> float:
    """Largest minimum-image displacement of the adsorbate atoms between endpoints."""
    from ase.geometry import find_mic

    delta = final.positions[-adsorbate_atoms:] - initial.positions[-adsorbate_atoms:]
    _, lengths = find_mic(delta, initial.cell, initial.pbc)
    return float(np.max(lengths))


# CALCULATORS
def calculator_settings(profile_name: str, ecut: float, kpts) -> dict:
    catalog = {
        str(profile["name"]): profile
        for profile in (run_dft.OPTIMIZATION_PROFILES + run_dft.CEO2_EXTRA_PROFILES)
    }

    if profile_name not in catalog:
        raise PairAborted(STATUS_DFT_NOT_OK, f"Unknown DFT profile {profile_name!r} in dft_summary.csv.")

    profile = catalog[profile_name]
    mixer = profile.get("mixer")

    return {
        "builder": "scripts.simulations.dft.run_dft.build_calculator",
        "profile": profile_name,
        "ecut": float(ecut),
        "kpts": [int(value) for value in kpts],
        "mode": str(profile.get("mode", "pw")),
        "smearing": float(profile["smearing"]),
        "maxiter": int(profile["maxiter"]),
        "mixer": list(mixer) if mixer is not None else None,
    }


@contextlib.contextmanager
def _gpaw_communicator(communicator):
    """Make run_dft.build_calculator create GPAW on an image sub-communicator."""
    import gpaw

    original = gpaw.GPAW
    gpaw.GPAW = functools.partial(original, communicator=communicator)
    try:
        yield
    finally:
        gpaw.GPAW = original


def make_calculator(settings: dict, txt_file: Path, communicator=None):
    kwargs = {
        "ecut": settings["ecut"],
        "kpts": tuple(settings["kpts"]),
        "txt_file": txt_file,
        "mode": settings["mode"],
        "smearing": settings["smearing"],
        "maxiter": settings["maxiter"],
        "mixer": tuple(settings["mixer"]) if settings["mixer"] is not None else None,
    }

    if communicator is None:
        return run_dft.build_calculator(**kwargs)

    with _gpaw_communicator(communicator):
        return run_dft.build_calculator(**kwargs)


def endpoint_single_point(atoms, settings: dict, txt_file: Path) -> tuple[float, float]:
    """Recompute energy and forces with the image calculator, then freeze them.

    No communicator is passed, so the single point runs on all MPI ranks (world).
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms.calc = make_calculator(settings, txt_file)
    energy = float(atoms.get_potential_energy())
    fmax = run_dft.max_free_force(atoms)
    forces = atoms.get_forces(apply_constraint=False)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    return energy, fmax


# OPTIMIZATION PHASES
def plain_step_budget(steps: int, climb_start_fraction: float, climb: bool) -> int:
    if not climb:
        return int(steps)
    return int(round(climb_start_fraction * steps))


def run_neb_phases(neb, optimizer, *, fmax: float, fmax_ci: float, steps: int,
                   climb: bool, climb_start_fraction: float) -> dict:
    """Plain NEB, then (always, if climb) a climbing-image phase on the remaining budget.

    ASE optimizers accumulate ``nsteps`` across ``run`` calls and each call
    allows ``steps`` more, so the CI phase gets ``steps - n_steps_plain``.
    """
    start = int(optimizer.nsteps)
    plain_budget = plain_step_budget(steps, climb_start_fraction, climb)

    print0(f"[neb] Phase 1: plain NEB, fmax={fmax}, up to {plain_budget} steps")
    converged_plain = bool(optimizer.run(fmax=fmax, steps=plain_budget))
    n_steps_plain = int(optimizer.nsteps) - start

    record = {
        "fmax_plain": float(fmax),
        "plain_step_budget": plain_budget,
        "n_steps_plain": n_steps_plain,
        "converged_plain": converged_plain,
        "climb_activated": False,
        "fmax_ci": float(fmax_ci) if climb else None,
        "ci_step_budget": 0,
        "n_steps_ci": 0,
        "converged_ci": None,
    }

    if climb:
        ci_budget = int(steps) - n_steps_plain
        print0(f"[neb] Phase 2: climbing image, fmax={fmax_ci}, up to {ci_budget} steps")
        neb.climb = True
        converged_ci = bool(optimizer.run(fmax=fmax_ci, steps=ci_budget))
        record.update(
            {
                "climb_activated": True,
                "ci_step_budget": ci_budget,
                "n_steps_ci": int(optimizer.nsteps) - start - n_steps_plain,
                "converged_ci": converged_ci,
            }
        )
        converged = converged_ci
    else:
        converged = converged_plain

    record["n_steps_total"] = record["n_steps_plain"] + record["n_steps_ci"]
    record["converged"] = converged
    return record


# OUTPUT
def write_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def discover_neb_runs(base: Path) -> list[dict]:
    """Pair folders under results/neb/<material> written by this script."""
    runs = []

    if not base.exists():
        return runs

    for summary_path in sorted(base.glob(f"*/{SUMMARY_JSON}")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARNING] Could not read {summary_path}: {exc}")
            continue

        folder = summary_path.parent
        runs.append(
            {
                "folder": folder,
                "folder_name": folder.name,
                "material": summary.get("material", ""),
                "initial_site": summary.get("initial_site", ""),
                "final_site": summary.get("final_site", ""),
                "status": summary.get("status", ""),
                "summary": summary,
            }
        )

    return runs


def read_final_images(folder: Path) -> list:
    """Final band frames written by run_neb (traj with energies and forces, else image_XX.xyz)."""
    from ase.io import read

    traj = folder / FINAL_IMAGES_TRAJ
    if traj.exists():
        return list(read(traj, index=":"))
    return [read(path) for path in sorted(folder.glob("image_[0-9][0-9].xyz"))]


def read_image_energies(folder: Path) -> list[float]:
    """Energies of the final band from the frames written by run_neb (not GPAW logs)."""
    energies = []
    for atoms in read_final_images(folder):
        try:
            energies.append(float(atoms.get_potential_energy()))
        except Exception:
            energies.append(float("nan"))
    return energies


def band_fit(images) -> dict | None:
    """Smooth band through the images from their energies and stored forces.

    Uses ase.utils.forcecurve.fit_images (what ase.mep.NEBTools.plot_band draws): a cubic
    spline per segment matching the energies and the forces projected on the path.
    Returns None when the frames have no energies or forces.
    """
    from ase.utils.forcecurve import fit_images

    try:
        fit = fit_images(images)
    except Exception as exc:
        print(f"[WARNING] smooth band fit not available: {type(exc).__name__}: {exc}")
        return None
    e_is = float(fit.energies[0])
    return {
        "path_ang": [float(value) for value in fit.path],
        "relative_energies": [float(value) - e_is for value in fit.energies],
        "fit_path_ang": [float(value) for value in fit.fit_path],
        "fit_relative_energies": [float(value) - e_is for value in fit.fit_energies],
    }


def resolve_barrier_resolution(cli_value: float | None, base: Path) -> float:
    """Barrier resolution (eV) for re-analysis of a run: CLI, else YAML, else the default.

    ``base`` is <results>/neb/<material>; the YAML is <results>/config.yaml (copied there by
    the workflow), else the production YAML.
    """
    if cli_value is not None:
        value = float(cli_value)
    else:
        results_dir = Path(base).resolve().parents[1]
        try:
            config = load_config(None, results_dir)
        except Exception as exc:  # unreadable YAML: fall back to the default
            print(f"[WARNING] could not read the NEB configuration: {exc}")
            config = {}
        value = float(method_settings(config, "neb").get("barrier_resolution_ev", DEFAULT_BARRIER_RESOLUTION_EV))
    if value < 0.0:
        raise SystemExit("The barrier resolution must be non-negative.")
    return value


def write_paths_summary(material_dir: Path):
    rows = []
    for run in discover_neb_runs(material_dir):
        summary = run["summary"]
        row = {key: summary.get(key, "") for key in PATHS_SUMMARY_FIELDS}
        if isinstance(row["kpts"], list):
            row["kpts"] = " ".join(str(value) for value in row["kpts"])
        rows.append(row)

    if not rows:
        return

    with (material_dir / PATHS_SUMMARY_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATHS_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# SINGLE PAIR
def _run_pair(record: dict, *, material: str, initial_site: str, final_site: str,
              args, results_dir: Path, dft_rows: dict, out_dir: Path, world):
    from ase.io import write
    from ase.mep import NEB
    from ase.optimize import BFGS, FIRE

    rows = {}
    for site in (initial_site, final_site):
        row = dft_rows.get((material, site))
        if row is None:
            raise PairAborted(STATUS_MISSING_INPUT, f"No dft_summary.csv row for {material} | {site}.")
        if row.get("dft_status") != "ok":
            raise PairAborted(
                STATUS_DFT_NOT_OK,
                f"DFT for {material} | {site} has status {row.get('dft_status')!r}, not 'ok'.",
            )
        rows[site] = row

    profile_initial = rows[initial_site].get("dft_profile", "")
    profile_final = rows[final_site].get("dft_profile", "")
    record["dft_profile_initial"] = profile_initial
    record["dft_profile_final"] = profile_final
    if profile_initial != profile_final:
        raise PairAborted(
            STATUS_PROFILE_MISMATCH,
            f"Endpoints were optimized with different DFT profiles "
            f"({initial_site}={profile_initial}, {final_site}={profile_final}); "
            f"their energies are not comparable.",
        )
    record["dft_profile"] = profile_initial

    paths = {site: dft_structure_path(results_dir, material, site) for site in (initial_site, final_site)}
    for site, path in paths.items():
        if not path.exists():
            raise PairAborted(STATUS_MISSING_INPUT, f"Missing optimized DFT structure for {material} | {site}: {path}")

    record["endpoint_files"] = {
        "initial": {"path": str(paths[initial_site]), "sha256": sha256_file(paths[initial_site])},
        "final": {"path": str(paths[final_site]), "sha256": sha256_file(paths[final_site])},
    }
    record["dft_summary_sha256"] = sha256_file(results_dir / "dft_summary.csv")

    initial = read_endpoint(paths[initial_site])
    final = read_endpoint(paths[final_site])

    check_compatible_endpoints(initial, final, initial_site, final_site)
    record["fixed_indices"] = fixed_indices(initial)

    displacement = adsorbate_displacement(initial, final)
    record["endpoint_max_adsorbate_displacement_ang"] = displacement
    if displacement <= MIN_ENDPOINT_DISPLACEMENT_ANG:
        raise PairAborted(
            STATUS_IDENTICAL_ENDPOINTS,
            f"CO2 moves at most {displacement:.3f} A between {initial_site} and {final_site} "
            f"(<= {MIN_ENDPOINT_DISPLACEMENT_ANG} A); both relaxed to the same minimum.",
        )

    settings = calculator_settings(profile_initial, args.ecut, args.kpts)
    record["calculator"] = settings

    endpoints = {}
    for label, site, atoms in (("initial", initial_site, initial), ("final", final_site, final)):
        energy, fmax = endpoint_single_point(atoms, settings, out_dir / f"endpoint_{label}.txt")
        stored = float(rows[site].get("adsorbed_energy_ev", "nan"))
        endpoints[label] = {
            "site": site,
            "energy_ev": energy,
            "dft_summary_energy_ev": stored,
            "energy_difference_ev": energy - stored,
            "max_free_force_ev_per_ang": fmax,
        }
    record["endpoints"] = endpoints

    for label, info in endpoints.items():
        if not math.isfinite(info["energy_difference_ev"]) or abs(info["energy_difference_ev"]) > args.endpoint_energy_tol_ev:
            raise PairAborted(
                STATUS_PROVENANCE,
                f"{label} endpoint ({info['site']}) single-point energy {info['energy_ev']:.6f} eV differs from "
                f"dft_summary adsorbed_energy_ev {info['dft_summary_energy_ev']:.6f} eV by more than "
                f"{args.endpoint_energy_tol_ev} eV; endpoint and calculator do not match the DFT run.",
            )
        if info["max_free_force_ev_per_ang"] > args.endpoint_force_tol:
            raise PairAborted(
                STATUS_ENDPOINT_FORCES,
                f"{label} endpoint ({info['site']}) max free-atom force "
                f"{info['max_free_force_ev_per_ang']:.4f} eV/A exceeds {args.endpoint_force_tol:.3f} eV/A "
                f"({args.dft_fmax} eV/A DFT fmax with SCF slack); "
                f"it is not a converged minimum.",
            )

    if world.rank == 0:
        write(out_dir / "initial.xyz", initial, parallel=False)
        write(out_dir / "final.xyz", final, parallel=False)

    # Copies keep the endpoint constraint, so every image has the same FixAtoms.
    images = [initial]
    for _ in range(args.images):
        image = initial.copy()
        image.calc = None
        images.append(image)
    images.append(final)

    use_parallel = args.parallel_images and world.size > 1
    record["parallel_images"] = bool(use_parallel)

    if use_parallel:
        n_internal = args.images

        check_parallel_layout(n_internal, world.size, True)

        ranks_per_image = world.size // n_internal

        for i, image in enumerate(images[1:-1]):
            ranks = range(i * ranks_per_image, (i + 1) * ranks_per_image)

            if world.rank in ranks:
                communicator = world.new_communicator(ranks)
                image.calc = make_calculator(
                    settings,
                    out_dir / f"neb_image_{i + 1}.txt",
                    communicator=communicator,
                )
    else:
        for i, image in enumerate(images[1:-1]):
            image.calc = make_calculator(settings, out_dir / f"neb_image_{i + 1}.txt")

    neb = NEB(
        images,
        parallel=bool(use_parallel),
        climb=False,
        method="improvedtangent",
        allow_shared_calculator=False,
    )

    neb.interpolate(method=args.interpolation, mic=True, apply_constraint=False)

    optimizer_cls = FIRE if args.optimizer == "FIRE" else BFGS

    optimizer = optimizer_cls(
        neb,
        logfile=str(out_dir / "neb_opt.log"),
        trajectory=str(out_dir / "neb.traj"),
    )

    with timer(
        f"NEB | {material} | {initial_site} -> {final_site} | "
        f"images={args.images}"
    ):
        phases = run_neb_phases(
            neb,
            optimizer,
            fmax=args.fmax,
            fmax_ci=args.fmax_ci,
            steps=args.steps,
            climb=args.climb,
            climb_start_fraction=args.climb_start_fraction,
        )
    record.update(phases)

    # iterimages() attaches the broadcast energies/forces to every image, also
    # for images computed on other MPI ranks.
    final_images = list(neb.iterimages())
    energies = [float(atoms.get_potential_energy()) for atoms in final_images]

    if world.rank == 0:
        for i, atoms in enumerate(final_images):
            write(out_dir / f"image_{i:02d}.xyz", atoms, parallel=False)

        write(out_dir / FINAL_IMAGES_TRAJ, final_images, parallel=False)

    metrics = barrier_metrics(
        energies, resolution_ev=getattr(args, "barrier_resolution_ev", DEFAULT_BARRIER_RESOLUTION_EV)
    )
    record["energies_ev"] = energies

    if metrics is None:
        raise PairAborted(STATUS_INVALID_ENERGIES, f"Non-finite image energies: {energies}")

    record.update({key: value for key, value in metrics.items() if key != "energies"})
    record["status"] = STATUS_CONVERGED if phases["converged"] else STATUS_NOT_CONVERGED


def run_single_neb(
    *,
    material: str,
    initial_site: str,
    final_site: str,
    args,
    results_dir: Path,
    dft_rows: dict,
    world=None,
) -> dict:
    world = world if world is not None else get_world()

    pair_name = f"{initial_site}_to_{final_site}"
    out_dir = results_dir / "neb" / material / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / SUMMARY_JSON

    if summary_path.exists() and not args.overwrite:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        print0(
            f"[neb] Skipping existing path: {material} | {pair_name} "
            f"(status={existing.get('status')}; use --overwrite to rerun)"
        )
        return existing

    print0(
        f"\n[neb] Start | material={material} | "
        f"path={initial_site} -> {final_site}"
    )

    record = {
        "material": material,
        "initial_site": initial_site,
        "final_site": final_site,
        "path": pair_name,
        "internal_images": args.images,
        "total_images": args.images + 2,
        "fmax": args.fmax,
        "fmax_ci": args.fmax_ci if args.climb else None,
        "dft_fmax": args.dft_fmax,
        "endpoint_energy_tol_ev": args.endpoint_energy_tol_ev,
        "endpoint_force_tol": args.endpoint_force_tol,
        "max_steps": args.steps,
        "ecut": args.ecut,
        "kpts": [int(value) for value in args.kpts],
        "interpolation": args.interpolation,
        "climb": bool(args.climb),
        "climb_start_fraction": args.climb_start_fraction,
        "optimizer": args.optimizer,
        "parallel_images": False,
        "mpi_size": int(world.size),
        "config_path": getattr(args, "config_path", None),
        "status": "",
        "message": "",
    }

    try:
        _run_pair(
            record,
            material=material,
            initial_site=initial_site,
            final_site=final_site,
            args=args,
            results_dir=results_dir,
            dft_rows=dft_rows,
            out_dir=out_dir,
            world=world,
        )
    except PairAborted as exc:
        record["status"] = exc.status
        record["message"] = str(exc)
        print0(f"[neb] {material} | {pair_name}: {exc.status}: {exc}")

    record["converged"] = record["status"] == STATUS_CONVERGED

    if world.rank == 0:
        write_json(summary_path, record)

    print0(
        f"[neb] Done | status={record['status']} "
        f"| results saved in: {out_dir}"
    )

    return record


def run_neb(args) -> list[dict]:
    world = get_world()

    results_dir = Path(args.results_dir).resolve() if args.results_dir else project_config.RESULTS_DIR
    if args.results_dir is None:
        ensure_directories()

    config = load_config(args.config, results_dir)
    args.config_path = config.get("_config_path")
    resolve_settings(args, config)
    # Fail before any SCF if the ranks cannot be split evenly over the images.
    check_parallel_layout(args.images, int(world.size), args.parallel_images)

    material_dir = results_dir / "neb" / args.material
    material_dir.mkdir(parents=True, exist_ok=True)

    pairs = get_pairs(args, config)
    dft_rows = read_dft_summary(results_dir)

    if world.rank == 0:
        print0("\n=== Nudged Elastic Band calculations ===")
        print0(f"[neb] material: {args.material}")
        print0(f"[neb] config: {args.config_path}")
        print0(f"[neb] pairs: {pairs}")
        print0(f"[neb] internal images: {args.images}")
        print0(f"[neb] interpolation: {args.interpolation}")
        print0(f"[neb] climb: {args.climb} (fmax plain={args.fmax}, fmax CI={args.fmax_ci})")
        print0(f"[neb] optimizer: {args.optimizer}")
        print0(f"[neb] ecut={args.ecut} kpts={list(args.kpts)}")
        print0(f"[neb] parallel images: {args.parallel_images}")

    outcomes = []
    for initial_site, final_site in pairs:
        outcomes.append(
            run_single_neb(
                material=args.material,
                initial_site=initial_site,
                final_site=final_site,
                args=args,
                results_dir=results_dir,
                dft_rows=dft_rows,
                world=world,
            )
        )

    if world.rank == 0:
        write_paths_summary(material_dir)

    failed = [
        f"{outcome.get('path')}={outcome.get('status')}"
        for outcome in outcomes
        if outcome.get("status") not in NON_FAILING_STATUSES
    ]

    if failed:
        print0(f"\n[neb] {len(failed)} pair(s) did not converge or were aborted: {failed}")
        raise SystemExit(1)

    print0("\n[neb] All NEB paths finished.")
    return outcomes


def main():
    args = parse_args()
    run_neb(args)


if __name__ == "__main__":
    main()
