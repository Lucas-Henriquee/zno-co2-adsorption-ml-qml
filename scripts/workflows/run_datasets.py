from __future__ import annotations

import argparse
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
from scripts.common.md_inputs import find_md_summaries


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "zno_existing_results.yaml"
METHOD_ORDER = ("tddft", "multitech", "qml_features")
MODULES = {
    "tddft": "scripts.datasets.prepare_tddft_qml_dataset",
    "multitech": "scripts.datasets.prepare_multitech_dataset",
    "qml_features": "scripts.datasets.export_qml_dataset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build isolated, traceable datasets from simulation and analysis outputs."
    )
    parser.add_argument(
        "methods",
        nargs="*",
        help="One or more methods, or 'all'. Available: tddft, multitech, qml_features.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", help="Safe directory name for this dataset generation run.")
    parser.add_argument("--source-results", type=Path, help="Override input.results_dir.")
    parser.add_argument("--ranking", type=Path, help="Override the ranked candidate input file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without creating files.",
    )
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


def normalize_methods(raw_methods: list[str]) -> list[str]:
    if not raw_methods:
        fail("Select at least one method or use 'all'.")
    normalized = [name.lower() for name in raw_methods]
    if "all" in normalized:
        if len(normalized) != 1:
            fail("Use 'all' by itself, without additional method names.")
        return list(METHOD_ORDER)
    unknown = [name for name in normalized if name not in METHOD_ORDER]
    if unknown:
        fail(f"Unknown methods: {', '.join(unknown)}")
    return list(dict.fromkeys(normalized))


def resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def resolve_source_path(value: Any, source_results: Path, label: str) -> Path:
    if value is None:
        fail(f"Missing configuration field 'input.{label}'.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = source_results / path
    return path.resolve()


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        fail("Run ID may only contain letters, numbers, periods, underscores, and hyphens.")
    return run_id


def default_run_id(output_root: Path | None = None) -> str:
    from scripts.common.run_names import next_run_name
    return next_run_name(output_root or PROJECT_ROOT / "data/generated", "dataset")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "pyyaml"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def method_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    methods = require_mapping(config, "methods")
    value = methods.get(method)
    if not isinstance(value, dict):
        fail(f"Configuration field 'methods.{method}' must be a mapping.")
    return value


def collect_inputs(
    method: str,
    config: dict[str, Any],
    source_results: Path,
    ranking_override: Path | None,
) -> list[Path]:
    input_config = require_mapping(config, "input")
    settings = method_config(config, method)
    paths: list[Path] = []

    if method == "tddft":
        mode = str(settings.get("mode", "both"))
        if mode not in {"summary", "transitions", "both"}:
            fail("methods.tddft.mode must be summary, transitions, or both.")
        paths.append(resolve_source_path(input_config.get("tddft_summary"), source_results, "tddft_summary"))
        if mode in {"transitions", "both"}:
            directory = resolve_source_path(
                input_config.get("tddft_directory"), source_results, "tddft_directory"
            )
            transition_files = sorted(directory.glob("*/*/transitions.csv"))
            if not transition_files:
                fail(f"No TDDFT transition files found below: {directory}")
            paths.extend(transition_files)

    elif method == "multitech":
        targets = settings.get("targets", ["summary", "md_windows"])
        if not isinstance(targets, list) or not targets:
            fail("methods.multitech.targets must be a non-empty list.")
        if "expanded" in [str(item) for item in targets]:
            fail(
                "The multitech 'expanded' target was removed (cartesian product that leaked "
                "the site label); use summary and md_windows."
            )
        valid_targets = {"summary", "md_windows"}
        unknown = [str(item) for item in targets if str(item) not in valid_targets]
        if unknown:
            fail(f"Unknown multitech targets: {', '.join(unknown)}")
        windows = int(settings.get("md_windows_per_site", 10))
        if windows < 0:
            fail("methods.multitech.md_windows_per_site must be zero (no cap) or positive.")

        if "summary" in targets:
            # DFT is required; TDDFT, CO2RR and MD are joined one to one when present.
            paths.append(source_results / "dft_summary.csv")
            paths.extend(
                source_results / name
                for name in ["tddft_summary.csv", "co2rr_pathway_summary.csv"]
                if (source_results / name).is_file()
            )
            try:
                paths.extend(find_md_summaries(source_results))
            except FileNotFoundError:
                pass
        if "md_windows" in targets:
            paths.extend(find_md_summaries(source_results))

    elif method == "qml_features":
        if ranking_override is not None:
            ranking = resolve_project_path(ranking_override)
        else:
            ranking = resolve_source_path(input_config.get("ranking"), source_results, "ranking")
        paths.append(ranking)

    missing = [path for path in paths if not path.is_file()]
    if missing:
        fail("Missing dataset inputs: " + ", ".join(str(path) for path in missing))
    return list(dict.fromkeys(paths))


def build_command(
    method: str,
    config: dict[str, Any],
    source_results: Path,
    ranking_override: Path | None,
    run_dir: Path,
) -> list[str]:
    settings = method_config(config, method)
    input_config = require_mapping(config, "input")
    command = [sys.executable, "-m", MODULES[method]]

    if method == "tddft":
        mode = str(settings.get("mode", "both"))
        summary = resolve_source_path(input_config.get("tddft_summary"), source_results, "tddft_summary")
        result = [
            *command,
            "--summary", str(summary),
            "--mode", mode,
            "--summary-output", str(run_dir / "tddft" / "tddft_summary_dataset.csv"),
            "--transitions-output", str(run_dir / "tddft" / "tddft_transitions_dataset.csv"),
        ]
        if mode in {"transitions", "both"}:
            tddft_dir = resolve_source_path(
                input_config.get("tddft_directory"), source_results, "tddft_directory"
            )
            result.extend(["--tddft-dir", str(tddft_dir)])
        return result

    if method == "multitech":
        targets = [
            str(item)
            for item in settings.get("targets", ["summary", "md_windows"])
        ]
        result = [
            *command,
            "--results-dir", str(source_results),
            "--output-dir", str(run_dir / "multitech"),
            "--targets", *targets,
            "--md-windows-per-site", str(int(settings.get("md_windows_per_site", 10))),
        ]
        for key, option in [
            ("block_tau_factor", "--block-tau-factor"),
            ("min_block_length_fs", "--min-block-length-fs"),
            ("equilibration_fs", "--equilibration-fs"),
        ]:
            if settings.get(key) is not None:
                result.extend([option, str(float(settings[key]))])
        return result

    if ranking_override is not None:
        ranking = resolve_project_path(ranking_override)
    else:
        ranking = resolve_source_path(input_config.get("ranking"), source_results, "ranking")
    result = [
        *command,
        "--ranking", str(ranking),
        "--output", str(run_dir / "qml_features" / "qml_features.csv"),
    ]
    default_material = settings.get("default_material")
    if default_material:
        result.extend(["--default-material", str(default_material)])
    features = settings.get("features")
    if features:
        result.extend(["--features", *[str(item) for item in features]])
    return result


def input_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(paths)
    ]


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def run_command(command: list[str], log_path: Path) -> None:
    print(f"[command] {shlex.join(command)}", flush=True)
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in [str(PROJECT_ROOT), existing_python_path] if item
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[{utc_now()}] {shlex.join(command)}\n")
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


def main() -> None:
    args = parse_args()
    if args.list_methods:
        print("\n".join(METHOD_ORDER))
        return

    config = load_config(args.config)
    selected_methods = normalize_methods(args.methods)
    input_config = require_mapping(config, "input")
    configured_results = Path(str(input_config.get("results_dir", "results")))
    source_results = resolve_project_path(args.source_results or configured_results)
    if not source_results.is_dir():
        fail(f"Source results directory does not exist: {source_results}")

    output_config = require_mapping(config, "output")
    output_root = resolve_project_path(Path(str(output_config.get("root", "data/generated"))))
    protected_root = (PROJECT_ROOT / "data").resolve()
    if not output_root.is_relative_to(protected_root):
        fail(f"Dataset output must stay below the project data directory: {protected_root}")
    run_id = validate_run_id(args.run_id or default_run_id(output_root))
    run_dir = (output_root / run_id).resolve()

    inputs_by_method = {
        method: collect_inputs(method, config, source_results, args.ranking)
        for method in selected_methods
    }
    commands = {
        method: build_command(method, config, source_results, args.ranking, run_dir)
        for method in selected_methods
    }

    print(f"[plan] config: {config['_config_path']}")
    print(f"[plan] run ID: {run_id}")
    print(f"[plan] source results: {source_results}")
    print(f"[plan] output: {run_dir}")
    print(f"[plan] methods: {', '.join(selected_methods)}")
    for method in selected_methods:
        print(f"[plan:{method}] {shlex.join(commands[method])}")
        print(f"[plan:{method}] input files: {len(inputs_by_method[method])}")

    if args.dry_run:
        print("[dry-run] Validation completed; no directories or datasets were created.")
        return

    if run_dir.exists():
        fail(f"Dataset run directory already exists; choose another --run-id: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    config_path = Path(config["_config_path"])
    shutil.copy2(config_path, run_dir / "config.yaml")

    all_inputs = list(dict.fromkeys(path for paths in inputs_by_method.values() for path in paths))
    execution = require_mapping(config, "execution")
    environment_name = os.environ.get(
        "QML_ZNO_ACTIVE_ENVIRONMENT",
        str(execution.get("environment", "qml-zno-modeling")),
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "config_path": str(config_path),
        "config_snapshot": str(run_dir / "config.yaml"),
        "config_sha256": file_sha256(config_path),
        "environment": environment_name,
        "versions": runtime_versions(),
        "source_results": str(source_results),
        "output_directory": str(run_dir),
        "selected_methods": selected_methods,
        "inputs": input_inventory(all_inputs),
        "methods": {},
    }
    write_manifest(manifest_path, manifest)

    try:
        for method in selected_methods:
            record = {
                "status": "running",
                "started_at": utc_now(),
                "command": commands[method],
                "input_files": [str(path) for path in inputs_by_method[method]],
            }
            manifest["methods"][method] = record
            write_manifest(manifest_path, manifest)
            try:
                run_command(commands[method], run_dir / "logs" / f"{method}.log")
                output_files = sorted(
                    path for path in (run_dir / method).rglob("*") if path.is_file()
                )
                if not output_files:
                    raise RuntimeError(
                        f"Method '{method}' completed without creating a dataset."
                    )
                record["outputs"] = input_inventory(output_files)
            except BaseException as exc:
                record["status"] = "failed"
                record["failed_at"] = utc_now()
                record["error"] = f"{type(exc).__name__}: {exc}"
                write_manifest(manifest_path, manifest)
                raise
            record["status"] = "completed"
            record["completed_at"] = utc_now()
            write_manifest(manifest_path, manifest)

        manifest["status"] = "completed"
        write_manifest(manifest_path, manifest)
        print(f"[done] Dataset generation completed: {run_dir}")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
