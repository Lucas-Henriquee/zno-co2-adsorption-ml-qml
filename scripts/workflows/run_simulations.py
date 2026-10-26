from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from scripts.common.project_config import DFT_SITES, EXPLORATORY_SITES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "simulations" / "zno_production.yaml"
METHOD_ORDER = ("geometry", "dft", "tddft", "co2rr", "md", "neb")
METHOD_ALIASES = {"pathways": "co2rr"}
DEPENDENCIES = {
    "geometry": (),
    "dft": ("geometry",),
    "tddft": ("dft",),
    "co2rr": ("dft",),
    "md": ("dft",),
    "neb": ("dft",),
}
MODULES = {
    "geometry": "scripts.simulations.dft.build_geometries",
    "dft": "scripts.simulations.dft.run_dft",
    "tddft": "scripts.simulations.tddft.run_tddft",
    "co2rr": "scripts.simulations.co2rr.run_co2rr_pathways",
    "md": "scripts.simulations.md.run_md",
    "neb": "scripts.simulations.neb.run_neb",
}
VALID_MATERIALS = {"ZnO", "TiO2", "CeO2"}
VALID_SITES = set(DFT_SITES)
# Methods that accept the DFT-only exploratory sites (e.g. carbonate).
EXPLORATORY_SITE_METHODS = ("geometry", "dft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated DFT, TDDFT, CO2RR, MD, and NEB workflows from YAML configuration."
    )
    parser.add_argument(
        "methods",
        nargs="*",
        help="One or more methods, or 'all'. Available: geometry, dft, tddft, co2rr, md, neb.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", help="Safe directory name for this isolated run.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing run and skip methods already marked completed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print commands without creating files or running methods.",
    )
    parser.add_argument(
        "--no-dependencies",
        action="store_true",
        help="Do not add prerequisite methods automatically.",
    )
    parser.add_argument("--materials", help="Comma-separated material override.")
    parser.add_argument("--sites", help="Comma-separated adsorption-site override.")
    parser.add_argument("--mpi-processes", type=int, help="Override the configured MPI process count.")
    parser.add_argument("--list-methods", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[error] {message}")


def load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        fail(f"Configuration file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        fail(f"Configuration must contain a YAML mapping: {resolved}")
    if payload.get("schema_version") != 1:
        fail("Unsupported or missing schema_version; expected 1.")
    payload["_config_path"] = str(resolved)
    return payload


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        fail(f"Configuration field '{key}' must be a mapping.")
    return value


def parse_csv_override(value: str | None, fallback: Any, label: str) -> list[str]:
    if value is None:
        if not isinstance(fallback, list) or not fallback:
            fail(f"Configuration field 'selection.{label}' must be a non-empty list.")
        values = [str(item) for item in fallback]
    else:
        values = [item.strip() for item in value.split(",") if item.strip()]
        if not values:
            fail(f"--{label} must contain at least one value.")
    return list(dict.fromkeys(values))


def normalize_methods(raw_methods: list[str]) -> list[str]:
    if not raw_methods:
        fail("Select at least one method or use 'all'.")
    normalized = [METHOD_ALIASES.get(name.lower(), name.lower()) for name in raw_methods]
    if "all" in normalized:
        if len(normalized) != 1:
            fail("Use 'all' by itself, without additional method names.")
        return list(METHOD_ORDER)
    unknown = [name for name in normalized if name not in METHOD_ORDER]
    if unknown:
        fail(f"Unknown methods: {', '.join(unknown)}")
    return list(dict.fromkeys(normalized))


def resolve_methods(selected: list[str], include_dependencies: bool) -> list[str]:
    required = set(selected)

    def add_dependencies(method: str) -> None:
        for dependency in DEPENDENCIES[method]:
            if dependency not in required:
                required.add(dependency)
                add_dependencies(dependency)

    if include_dependencies:
        for method in selected:
            add_dependencies(method)
    return [method for method in METHOD_ORDER if method in required]


def validate_choice_list(values: list[str], valid: set[str], label: str) -> None:
    unknown = [value for value in values if value not in valid]
    if unknown:
        fail(f"Unknown {label}: {', '.join(unknown)}")


def validate_exploratory_sites(sites: list[str], methods: list[str]) -> None:
    """Exploratory starts are DFT-only; other methods would crash on them, so fail early."""
    exploratory = [site for site in sites if site in EXPLORATORY_SITES]
    other_methods = [method for method in methods if method not in EXPLORATORY_SITE_METHODS]
    if exploratory and other_methods:
        fail(
            f"{', '.join(exploratory)} is a DFT-only exploratory site; it cannot run with "
            f"{', '.join(other_methods)}. Run it separately with: dft --sites {','.join(exploratory)}"
        )


def validate_triplet(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        fail(f"Configuration field '{label}' must contain three integers.")
    triplet = [int(item) for item in value]
    if any(item <= 0 for item in triplet):
        fail(f"Configuration field '{label}' must contain positive integers.")
    return triplet


def method_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    methods = require_mapping(config, "methods")
    value = methods.get(method)
    if not isinstance(value, dict):
        fail(f"Configuration field 'methods.{method}' must be a mapping.")
    return value


def required_value(settings: dict[str, Any], key: str, method: str) -> Any:
    if key not in settings:
        fail(f"Missing configuration field 'methods.{method}.{key}'.")
    return settings[key]


def positive_float(settings: dict[str, Any], key: str, method: str) -> float:
    value = float(required_value(settings, key, method))
    if value <= 0.0:
        fail(f"methods.{method}.{key} must be positive.")
    return value


def positive_int(settings: dict[str, Any], key: str, method: str) -> int:
    value = int(required_value(settings, key, method))
    if value <= 0:
        fail(f"methods.{method}.{key} must be a positive integer.")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ["gpaw", "ase", "numpy", "scipy", "mpi4py", "pyyaml"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def command_prefix(method: str, config: dict[str, Any], mpi_processes: int) -> list[str]:
    execution = require_mapping(config, "execution")
    mpi = execution.get("mpi", {})
    if not isinstance(mpi, dict):
        fail("Configuration field 'execution.mpi' must be a mapping.")
    mpi_methods = mpi.get("methods", [])
    if not isinstance(mpi_methods, list):
        fail("Configuration field 'execution.mpi.methods' must be a list.")
    use_mpi = bool(mpi.get("enabled", False)) and method in mpi_methods and mpi_processes > 1
    if not use_mpi:
        return [sys.executable, "-m", MODULES[method]]
    launcher = str(mpi.get("launcher", "mpiexec"))
    extra_args = mpi.get("extra_args", [])
    if not isinstance(extra_args, list):
        fail("Configuration field 'execution.mpi.extra_args' must be a list.")
    return [launcher, *[str(item) for item in extra_args], "-n", str(mpi_processes), "gpaw", "python", "-m", MODULES[method]]


def build_commands(
    method: str,
    config: dict[str, Any],
    materials: list[str],
    sites: list[str],
    mpi_processes: int,
) -> list[list[str]]:
    settings = method_config(config, method)
    prefix = command_prefix(method, config, mpi_processes)

    if method == "geometry":
        return [[*prefix, "--materials", *materials]]

    if method == "dft":
        kpoints = validate_triplet(required_value(settings, "kpoints", method), "methods.dft.kpoints")
        command = [
            *prefix,
            "--materials", *materials,
            "--sites", *sites,
            "--fmax", str(positive_float(settings, "fmax_ev_per_angstrom", method)),
            "--steps", str(positive_int(settings, "max_steps", method)),
            "--ecut", str(positive_float(settings, "ecut_ev", method)),
            "--kpts", *map(str, kpoints),
            "--co2-box-vacuum", str(positive_float(settings, "co2_box_vacuum_angstrom", method)),
        ]
        relax_extra_blocks = int(settings.get("relax_extra_blocks", 1))
        if relax_extra_blocks < 0:
            fail("methods.dft.relax_extra_blocks must be zero or a positive integer.")
        command += ["--relax-extra-blocks", str(relax_extra_blocks)]
        if settings.get("scf_force_convergence") is not None:
            command += ["--scf-force-convergence", str(positive_float(settings, "scf_force_convergence", method))]
        return [command]

    if method == "tddft":
        return [[
            *prefix,
            "--materials", *materials,
            "--sites", *sites,
            "--max-transitions", str(positive_int(settings, "max_transitions", method)),
            "--osc-threshold", str(positive_float(settings, "oscillator_strength_threshold", method)),
            "--grid-spacing", str(positive_float(settings, "grid_spacing_angstrom", method)),
            "--box-size", str(positive_float(settings, "box_size_angstrom", method)),
            "--smearing-width", str(positive_float(settings, "smearing_width_ev", method)),
            "--nbands", str(positive_int(settings, "nbands", method)),
            "--converged-bands", str(positive_int(settings, "converged_bands", method)),
            "--energy-window", str(positive_float(settings, "energy_window_ev", method)),
        ]]

    if method == "co2rr":
        kpoints = validate_triplet(required_value(settings, "kpoints", method), "methods.co2rr.kpoints")
        return [[
            *prefix,
            "--materials", *materials,
            "--sites", *sites,
            "--fmax", str(positive_float(settings, "fmax_ev_per_angstrom", method)),
            "--steps", str(positive_int(settings, "max_steps", method)),
            "--ecut", str(positive_float(settings, "ecut_ev", method)),
            "--kpts", *map(str, kpoints),
            "--molecule-box-vacuum", str(positive_float(settings, "molecule_box_vacuum_angstrom", method)),
        ]]

    if method == "md":
        kpoints = validate_triplet(required_value(settings, "kpoints", method), "methods.md.kpoints")
        commands = []
        for material in materials:
            for site in sites:
                commands.append([
                    *prefix,
                    "--material", material,
                    "--site", site,
                    "--temperature", str(positive_float(settings, "temperature_kelvin", method)),
                    "--steps", str(positive_int(settings, "steps", method)),
                    "--timestep", str(positive_float(settings, "timestep_fs", method)),
                    "--log-interval", str(positive_int(settings, "log_interval", method)),
                    "--ecut", str(positive_float(settings, "ecut_ev", method)),
                    "--kpts", *map(str, kpoints),
                ])
        return commands

    if method == "neb":
        kpoints = validate_triplet(required_value(settings, "kpoints", method), "methods.neb.kpoints")
        internal_images = positive_int(settings, "internal_images", method)
        if len(sites) < 2:
            fail("NEB requires at least two adsorption sites.")
        interpolation = str(required_value(settings, "interpolation", method))
        if interpolation not in {"linear", "idpp"}:
            fail("methods.neb.interpolation must be 'linear' or 'idpp'.")
        optimizer = str(required_value(settings, "optimizer", method))
        if optimizer not in {"FIRE", "BFGS"}:
            fail("methods.neb.optimizer must be 'FIRE' or 'BFGS'.")
        climb_start_fraction = float(required_value(settings, "climb_start_fraction", method))
        if not 0.0 <= climb_start_fraction <= 1.0:
            fail("methods.neb.climb_start_fraction must be between 0 and 1.")
        if bool(settings.get("parallel_images", False)) and mpi_processes > 1:
            if mpi_processes % internal_images != 0:
                fail(
                    "NEB image parallelism requires MPI processes divisible by internal images "
                    f"({mpi_processes} is not divisible by {internal_images})."
                )
        commands = []
        for material in materials:
            command = [
                *prefix,
                "--material", material,
                "--sites", *sites,
                "--images", str(internal_images),
                "--fmax", str(positive_float(settings, "fmax_ev_per_angstrom", method)),
                "--steps", str(positive_int(settings, "max_steps", method)),
                "--ecut", str(positive_float(settings, "ecut_ev", method)),
                "--kpts", *map(str, kpoints),
                "--interpolation", interpolation,
                "--optimizer", optimizer,
                "--climb-start-fraction", str(climb_start_fraction),
            ]
            if bool(settings.get("climb", False)):
                command.append("--climb")
            if bool(settings.get("parallel_images", False)):
                command.append("--parallel-images")
            commands.append(command)
        return commands

    fail(f"No command builder is available for method: {method}")






def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_id(output_root: Path | None = None) -> str:
    from scripts.common.run_names import next_run_name
    return next_run_name(output_root or PROJECT_ROOT / "results/simulations", "simulation")


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        fail("Run ID may only contain letters, numbers, periods, underscores, and hyphens.")
    return run_id


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def run_command(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    print(f"[command] {shlex.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] {shlex.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def verify_method_outputs(method: str, run_dir: Path, materials: list[str], sites: list[str]) -> None:
    """Second check after a zero exit code: every requested DFT case must be 'ok'.

    The exit code is the primary failure signal for every method (run_command raises on
    a non-zero one). Only the DFT summary is cross-checked here, because its per-case
    status column is cheap to read; the other methods rely on their exit codes.
    """
    if method != "dft":
        return
    summary_path = run_dir / "dft_summary.csv"
    if not summary_path.is_file():
        raise RuntimeError(f"dft exited 0 but wrote no summary: {summary_path}")
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        status = {(row.get("material"), row.get("site")): row.get("dft_status") for row in csv.DictReader(handle)}
    not_ok = [f"{material}/{site}" for material in materials for site in sites
              if status.get((material, site)) != "ok"]
    if not_ok:
        raise RuntimeError(f"dft exited 0 but these cases are not ok: {', '.join(not_ok)}")


def main() -> None:
    args = parse_args()
    if args.list_methods:
        print("\n".join(METHOD_ORDER))
        return

    config = load_config(args.config)
    if args.resume and not args.run_id:
        fail("--resume requires an explicit --run-id.")
    selected_methods = normalize_methods(args.methods)
    resolved_methods = resolve_methods(selected_methods, include_dependencies=not args.no_dependencies)

    selection = require_mapping(config, "selection")
    materials = parse_csv_override(args.materials, selection.get("materials"), "materials")
    sites = parse_csv_override(args.sites, selection.get("sites"), "sites")
    validate_choice_list(materials, VALID_MATERIALS, "materials")
    validate_choice_list(sites, VALID_SITES, "sites")
    validate_exploratory_sites(sites, resolved_methods)

    execution = require_mapping(config, "execution")
    configured_environment = str(execution.get("environment", "qml-zno-simulation"))
    environment_name = os.environ.get("QML_ZNO_ACTIVE_ENVIRONMENT", configured_environment)
    omp_threads = int(execution.get("omp_num_threads", 1))
    if omp_threads <= 0:
        fail("execution.omp_num_threads must be positive.")
    mpi = execution.get("mpi", {})
    if not isinstance(mpi, dict):
        fail("Configuration field 'execution.mpi' must be a mapping.")
    mpi_processes = args.mpi_processes if args.mpi_processes is not None else int(mpi.get("processes", 1))
    if mpi_processes <= 0:
        fail("MPI process count must be positive.")
    if args.mpi_processes is not None and mpi_processes > 1:
        mpi["enabled"] = True
        if not mpi.get("methods"):
            mpi["methods"] = list(METHOD_ORDER[1:])

    output = require_mapping(config, "output")
    output_root = Path(str(output.get("root", "results/simulations"))).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_id = validate_run_id(args.run_id or default_run_id(output_root))
    run_dir = (output_root / run_id).resolve()
    protected_root = (PROJECT_ROOT / "results").resolve()
    if not run_dir.is_relative_to(protected_root):
        fail(f"Simulation output must stay below the project results directory: {protected_root}")
    manifest_path = run_dir / "run_manifest.json"
    config_path = Path(config["_config_path"])
    config_hash = file_sha256(config_path)
    current_versions = runtime_versions()

    if args.no_dependencies and not args.resume:
        dependent = [method for method in selected_methods if DEPENDENCIES[method]]
        if dependent:
            fail("--no-dependencies is only safe when resuming an existing isolated run.")

    commands_by_method = {
        method: build_commands(method, config, materials, sites, mpi_processes)
        for method in resolved_methods
    }

    print(f"[plan] config: {config['_config_path']}")
    print(f"[plan] run ID: {run_id}")
    print(f"[plan] output: {run_dir}")
    print(f"[plan] selected methods: {', '.join(selected_methods)}")
    print(f"[plan] resolved methods: {', '.join(resolved_methods)}")
    print(f"[plan] materials: {', '.join(materials)}")
    print(f"[plan] sites: {', '.join(sites)}")
    print(f"[plan] MPI processes: {mpi_processes}")
    for method in resolved_methods:
        for command in commands_by_method[method]:
            print(f"[plan:{method}] {shlex.join(command)}")

    if args.dry_run:
        print("[dry-run] Validation completed; no directories or result files were created.")
        return

    if args.resume:
        if not manifest_path.is_file():
            fail(f"Cannot resume because the manifest does not exist: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        immutable_context = {
            "config_sha256": config_hash,
            "environment": environment_name,
            "versions": current_versions,
            "materials": materials,
            "sites": sites,
            "mpi_processes": mpi_processes,
            "omp_num_threads": omp_threads,
        }
        for key, expected in immutable_context.items():
            if manifest.get(key) != expected:
                fail(
                    f"Cannot resume with a different {key}. "
                    f"Existing={manifest.get(key)!r}, requested={expected!r}."
                )
        previous_selected = [str(item) for item in manifest.get("selected_methods", [])]
        previous_resolved = [str(item) for item in manifest.get("resolved_methods", [])]
        manifest["selected_methods"] = list(dict.fromkeys([*previous_selected, *selected_methods]))
        manifest["resolved_methods"] = [
            method for method in METHOD_ORDER if method in {*previous_resolved, *resolved_methods}
        ]
    else:
        if run_dir.exists():
            fail(f"Run directory already exists; choose another --run-id: {run_dir}")
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "data" / "geometries").mkdir(parents=True)
        shutil.copy2(config_path, run_dir / "config.yaml")
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "planned",
            "config_path": config["_config_path"],
            "config_snapshot": str(run_dir / "config.yaml"),
            "config_sha256": config_hash,
            "environment": environment_name,
            "versions": current_versions,
            "selected_methods": selected_methods,
            "resolved_methods": resolved_methods,
            "materials": materials,
            "sites": sites,
            "mpi_processes": mpi_processes,
            "omp_num_threads": omp_threads,
            "paths": {
                "run_directory": str(run_dir),
                "geometry_directory": str(run_dir / "data" / "geometries"),
                "log_directory": str(run_dir / "logs"),
            },
            "methods": {},
        }
        write_manifest(manifest_path, manifest)

    runtime_environment = os.environ.copy()
    runtime_environment["QML_ZNO_RESULTS_DIR"] = str(run_dir)
    runtime_environment["QML_ZNO_GEOMETRY_DIR"] = str(run_dir / "data" / "geometries")
    runtime_environment["OMP_NUM_THREADS"] = str(omp_threads)
    runtime_environment["PYTHONUNBUFFERED"] = "1"
    existing_python_path = runtime_environment.get("PYTHONPATH", "")
    runtime_environment["PYTHONPATH"] = os.pathsep.join(
        item for item in [str(PROJECT_ROOT), existing_python_path] if item
    )

    manifest["status"] = "running"
    write_manifest(manifest_path, manifest)

    try:
        for method in resolved_methods:
            previous_status = manifest.get("methods", {}).get(method, {}).get("status")
            if previous_status == "completed":
                print(f"[skip] {method} is already completed in this run.")
                continue
            if previous_status in {"running", "failed"}:
                fail(
                    f"Method '{method}' has status '{previous_status}'. Start a new run ID to preserve its partial outputs."
                )

            method_record = {
                "status": "running",
                "started_at": utc_now(),
                "commands": commands_by_method[method],
            }
            manifest.setdefault("methods", {})[method] = method_record
            write_manifest(manifest_path, manifest)

            log_path = run_dir / "logs" / f"{method}.log"
            try:
                for dependency in DEPENDENCIES[method]:
                    if manifest.get("methods", {}).get(dependency, {}).get("status") != "completed":
                        raise RuntimeError(f"{method} requires completed {dependency} in this run.")
                for command in commands_by_method[method]:
                    run_command(command, log_path, runtime_environment)
                verify_method_outputs(method, run_dir, materials, sites)
            except BaseException as exc:
                method_record["status"] = "failed"
                method_record["failed_at"] = utc_now()
                method_record["error"] = f"{type(exc).__name__}: {exc}"
                write_manifest(manifest_path, manifest)
                raise

            method_record["status"] = "completed"
            method_record["completed_at"] = utc_now()
            write_manifest(manifest_path, manifest)

        manifest["status"] = "completed"
        write_manifest(manifest_path, manifest)
        print(f"[done] Simulation workflow completed: {run_dir}")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
