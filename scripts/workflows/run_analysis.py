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
from scripts.workflows.active_environment import active_environment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "analysis" / "zno_current.yaml"
METHOD_ORDER = ("screening", "co2_reduction", "science_figures", "model_analysis")
DEPENDENCIES = {
    "screening": (),
    "co2_reduction": (),
    "science_figures": ("co2_reduction",),
    "model_analysis": (),
}
ALIASES = {
    "science": ["screening", "co2_reduction", "science_figures"],
    "models": ["model_analysis"],
    "model-analysis": ["model_analysis"],
}
OUTPUT_SUBDIRECTORIES = {
    "screening": Path("science/screening"),
    "co2_reduction": Path("science/co2_reduction"),
    "science_figures": Path("science/figures"),
    "model_analysis": Path("models"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated scientific, model, table, and figure analyses."
    )
    parser.add_argument(
        "methods",
        nargs="*",
        help="screening, co2_reduction, science_figures, model_analysis, science, models, or all.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scientific-results", type=Path)
    parser.add_argument("--model-run", type=Path)
    parser.add_argument("--run-id", help="Safe directory name for this analysis run.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without creating files.",
    )
    parser.add_argument("--list-methods", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[error] {message}")


def resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def load_config(path: Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
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


def method_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    methods = require_mapping(config, "methods")
    value = methods.get(method)
    if not isinstance(value, dict):
        fail(f"Configuration field 'methods.{method}' must be a mapping.")
    return value


def normalize_methods(raw: list[str]) -> list[str]:
    if not raw:
        fail("Select at least one analysis method or use 'all'.")
    requested: list[str] = []
    for name in raw:
        normalized = name.lower()
        if normalized == "all":
            if len(raw) != 1:
                fail("Use 'all' by itself, without additional method names.")
            return list(METHOD_ORDER)
        if normalized in ALIASES:
            requested.extend(ALIASES[normalized])
        elif normalized in METHOD_ORDER:
            requested.append(normalized)
        else:
            fail(f"Unknown analysis method: {normalized}")
    return list(dict.fromkeys(requested))


def resolve_dependencies(selected: list[str]) -> list[str]:
    required = set(selected)

    def add(method: str) -> None:
        for dependency in DEPENDENCIES[method]:
            if dependency not in required:
                required.add(dependency)
                add(dependency)

    for method in selected:
        add(method)
    return [method for method in METHOD_ORDER if method in required]


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        fail("Run ID may only contain letters, numbers, periods, underscores, and hyphens.")
    return run_id


def default_run_id(output_root: Path | None = None) -> str:
    from scripts.common.run_names import next_run_name
    return next_run_name(output_root or PROJECT_ROOT / "results/analyses", "analysis")


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
    for package in ["ase", "numpy", "pandas", "matplotlib", "pyyaml"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def discover_benchmark_summary(model_run: Path) -> Path:
    matches = sorted(model_run.rglob("benchmark_summary.csv"))
    if not matches:
        fail(f"No benchmark_summary.csv found below: {model_run}")
    if len(matches) > 1:
        fail(f"Multiple benchmark summaries found below {model_run}; select a single model run.")
    return matches[0]


def collect_inputs(
    method: str,
    config: dict[str, Any],
    scientific_results: Path,
    model_run: Path,
) -> list[Path]:
    paths: list[Path] = []
    if method in {"screening", "co2_reduction"}:
        paths.extend(
            [scientific_results / "dft_summary.csv", scientific_results / "tddft_summary.csv"]
        )
    if method == "co2_reduction":
        pathway = scientific_results / "co2rr_pathway_summary.csv"
        if bool(method_config(config, method).get("require_pathway_results", False)):
            paths.append(pathway)
        elif pathway.is_file():
            paths.append(pathway)
    if method == "science_figures":
        transitions = sorted(scientific_results.glob("tddft/*/*/transitions.csv"))
        if not transitions:
            fail(f"No TDDFT transition files found below: {scientific_results / 'tddft'}")
        paths.extend(transitions)
    if method == "model_analysis":
        paths.append(discover_benchmark_summary(model_run))
        confusion = sorted(model_run.rglob("confusion_matrix.csv"))
        if not confusion:
            fail(f"No confusion matrices found below: {model_run}")
        paths.extend(confusion)

    missing = [path for path in paths if not path.is_file()]
    if missing:
        fail("Missing analysis inputs: " + ", ".join(str(path) for path in missing))
    return list(dict.fromkeys(paths))


def build_commands(
    method: str,
    config: dict[str, Any],
    scientific_results: Path,
    model_run: Path,
    run_dir: Path,
) -> list[list[str]]:
    python = sys.executable
    output = run_dir / OUTPUT_SUBDIRECTORIES[method]
    if method == "screening":
        return [[
            python,
            "-m",
            "scripts.analysis.analyze_results",
            "--results-dir",
            str(scientific_results),
            "--output-dir",
            str(output),
        ]]
    if method == "co2_reduction":
        return [[
            python,
            "-m",
            "scripts.analysis.analysis_co2_reduction",
            "--results-dir",
            str(scientific_results),
            "--output-dir",
            str(output),
        ]]
    if method == "science_figures":
        return [[
            python,
            "-m",
            "scripts.visualization.render_publication_figures",
            "--analysis-dir",
            str(run_dir / OUTPUT_SUBDIRECTORIES["co2_reduction"]),
            "--tddft-dir",
            str(scientific_results / "tddft"),
            "--output-dir",
            str(output),
            "--dpi",
            str(int(method_config(config, method).get("dpi", 600))),
        ]]
    benchmark_summary = discover_benchmark_summary(model_run)
    model_tables = output / "tables"
    model_figures = output / "figures"
    return [
        [
            python,
            "-m",
            "scripts.analysis.analyze_model_results",
            "--benchmark-summary",
            str(benchmark_summary),
            "--output-dir",
            str(model_tables),
        ],
        [
            python,
            "-m",
            "scripts.visualization.plot_current_model_results",
            "--comparison",
            str(model_tables / "model_comparison.csv"),
            "--model-run-dir",
            str(model_run),
            "--output-dir",
            str(model_figures),
            "--dpi",
            str(int(method_config(config, method).get("dpi", 300))),
        ],
    ]


def inventory(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
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
    cache_root = log_path.parent / ".cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    environment["XDG_CACHE_HOME"] = str(cache_root)
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
        print("\n".join([*METHOD_ORDER, *ALIASES, "all"]))
        return

    config = load_config(args.config)
    selected = normalize_methods(args.methods)
    resolved = resolve_dependencies(selected)
    input_config = require_mapping(config, "input")
    scientific_results = resolve_project_path(
        args.scientific_results or Path(str(input_config.get("scientific_results", "results")))
    )
    model_run = resolve_project_path(
        args.model_run or Path(str(input_config.get("model_run", "results/models")))
    )
    if any(method != "model_analysis" for method in resolved) and not scientific_results.is_dir():
        fail(f"Scientific results directory does not exist: {scientific_results}")
    if "model_analysis" in resolved and not model_run.is_dir():
        fail(f"Model run directory does not exist: {model_run}")

    output_config = require_mapping(config, "output")
    output_root = resolve_project_path(Path(str(output_config.get("root", "results/analyses"))))
    protected_root = (PROJECT_ROOT / "results").resolve()
    if not output_root.is_relative_to(protected_root):
        fail(f"Analysis output must stay below the project results directory: {protected_root}")
    run_id = validate_run_id(args.run_id or default_run_id(output_root))
    run_dir = (output_root / run_id).resolve()

    inputs_by_method = {
        method: collect_inputs(method, config, scientific_results, model_run)
        for method in resolved
    }
    commands_by_method = {
        method: build_commands(method, config, scientific_results, model_run, run_dir)
        for method in resolved
    }

    print(f"[plan] config: {config['_config_path']}")
    print(f"[plan] run ID: {run_id}")
    print(f"[plan] selected methods: {', '.join(selected)}")
    print(f"[plan] resolved methods: {', '.join(resolved)}")
    print(f"[plan] scientific results: {scientific_results}")
    if "model_analysis" in resolved:
        print(f"[plan] model run: {model_run}")
    print(f"[plan] output: {run_dir}")
    for method in resolved:
        for command in commands_by_method[method]:
            print(f"[plan:{method}] {shlex.join(command)}")
        print(f"[plan:{method}] input files: {len(inputs_by_method[method])}")

    if args.dry_run:
        print("[dry-run] Validation completed; no analysis files were created.")
        return
    if run_dir.exists():
        fail(f"Analysis run directory already exists; choose another --run-id: {run_dir}")

    (run_dir / "logs").mkdir(parents=True)
    config_path = Path(config["_config_path"])
    shutil.copy2(config_path, run_dir / "config.yaml")
    all_inputs = list(dict.fromkeys(path for paths in inputs_by_method.values() for path in paths))
    execution = require_mapping(config, "execution")
    environment_name = active_environment(str(execution.get("environment", "qml-zno-modeling")))
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
        "scientific_results": str(scientific_results),
        "model_run": str(model_run),
        "selected_methods": selected,
        "resolved_methods": resolved,
        "inputs": inventory(all_inputs),
        "methods": {},
    }
    write_manifest(manifest_path, manifest)

    try:
        for method in resolved:
            record: dict[str, Any] = {
                "status": "running",
                "started_at": utc_now(),
                "commands": commands_by_method[method],
                "input_files": [str(path) for path in inputs_by_method[method]],
            }
            manifest["methods"][method] = record
            write_manifest(manifest_path, manifest)
            try:
                for command in commands_by_method[method]:
                    run_command(command, run_dir / "logs" / f"{method}.log")
                method_root = run_dir / OUTPUT_SUBDIRECTORIES[method]
                outputs = sorted(path for path in method_root.rglob("*") if path.is_file())
                if not outputs:
                    raise RuntimeError(f"Method '{method}' produced no files.")
                record["outputs"] = inventory(outputs)
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
        print(f"[done] Analysis workflow completed: {run_dir}")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
