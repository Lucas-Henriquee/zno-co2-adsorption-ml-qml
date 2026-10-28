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

import pandas as pd
import yaml

from scripts.models.benchmarks.run_benchmarks import FEATURE_SETS
from scripts.common.run_names import next_run_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models" / "zno_local.yaml"
VALID_METHODS = ("svm", "qsvm", "local", "ibm-plan", "ibm")
BENCHMARK_MODULE = "scripts.models.benchmarks.run_benchmarks"
EVALUATION_MODULE = "scripts.models.evaluate_grouped"
# holdout: single grouped train/test split per seed (run_benchmarks only).
# grouped_27 (alias grouped): additionally run evaluate_grouped, i.e. every
# one-trajectory-per-class-out split (27 for 3 sites x 3 seeds), LOTO, nested
# grouped CV, trajectory-level permutation test and paired comparisons.
VALID_PROTOCOLS = ("holdout", "grouped_27", "grouped")
EVALUATION_MODELS = ("majority", "nearest_centroid", "logistic", "linear_svm", "svm_rbf", "qsvm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated classical, local quantum, and IBM model comparisons."
    )
    parser.add_argument(
        "method",
        nargs="?",
        help="svm, qsvm, local, all, ibm-plan, or ibm.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, help="Override input.dataset.")
    parser.add_argument("--run-id", help="Safe directory name for this model run.")
    parser.add_argument("--backend", help="Override the configured IBM backend name.")
    parser.add_argument(
        "--submit-ibm",
        action="store_true",
        help="Explicitly authorize IBM hardware submission for the ibm method.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the benchmark command without creating files.",
    )
    parser.add_argument(
        "--protocol",
        choices=VALID_PROTOCOLS,
        help="Override evaluation.protocol (holdout or grouped_27) for svm, qsvm and local.",
    )
    parser.add_argument("--list-methods", action="store_true")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[error] {message}")


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


def resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        fail(f"Configuration field '{key}' must be a mapping.")
    return value


def require_list(mapping: dict[str, Any], key: str, label: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        fail(f"Configuration field '{label}.{key}' must be a non-empty list.")
    return value


def normalize_method(method: str | None) -> tuple[str, list[str], bool, bool]:
    if method is None:
        fail("Select a model method or use --list-methods.")
    normalized = method.lower()
    if normalized == "all":
        normalized = "local"
    if normalized not in VALID_METHODS:
        fail(f"Unknown model method: {normalized}")
    if normalized == "local":
        return normalized, ["svm", "qsvm"], False, False
    if normalized == "ibm-plan":
        return normalized, ["qsvm", "qsvm_ibm"], True, False
    if normalized == "ibm":
        return normalized, ["qsvm", "qsvm_ibm"], False, True
    return normalized, [normalized], False, False


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        fail("Run ID may only contain letters, numbers, periods, underscores, and hyphens.")
    return run_id


def default_run_id(method: str, output_root: Path | None = None) -> str:
    return next_run_name(output_root or PROJECT_ROOT / "results/models", method.replace('-', '_'))


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
    for package in [
        "numpy",
        "pandas",
        "scikit-learn",
        "qiskit",
        "qiskit-machine-learning",
        "qiskit-aer",
        "qiskit-ibm-runtime",
        "pyyaml",
    ]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def strings(values: list[Any]) -> list[str]:
    return [str(value) for value in values]


def numbers(values: list[Any]) -> list[str]:
    return [str(value) for value in values]


def validate_dataset(
    dataset: Path,
    label: str,
    feature_sets: list[str],
    max_samples: int,
    test_size: float,
) -> dict[str, Any]:
    if not dataset.is_file():
        fail(f"Model dataset does not exist: {dataset}")
    unknown = [name for name in feature_sets if name not in FEATURE_SETS]
    if unknown:
        fail(f"Unknown feature sets: {', '.join(unknown)}")
    table = pd.read_csv(dataset)
    required = [label]
    for feature_set in feature_sets:
        required.extend(FEATURE_SETS[feature_set])
    required = list(dict.fromkeys(required))
    missing = [column for column in required if column not in table.columns]
    if missing:
        fail(f"Dataset is missing required columns: {missing}")
    usable = table.dropna(subset=required)
    if usable.empty:
        fail("Dataset has no usable rows after removing missing values.")
    class_counts = usable[label].value_counts().sort_index()
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        fail("Dataset must contain at least two classes with two samples each.")
    if max_samples < 0:
        fail("split.max_samples must be zero or positive.")
    selected_count = len(usable) if max_samples == 0 else min(max_samples, len(usable))
    if selected_count < 2 * len(class_counts):
        fail("split.max_samples is too small for a stratified train/test split.")
    if not 0.0 < test_size < 1.0:
        fail("split.test_size must be between zero and one.")
    return {
        "rows": len(table),
        "usable_rows": len(usable),
        "columns": len(table.columns),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
    }


def build_command(
    config: dict[str, Any],
    models: list[str],
    dataset: Path,
    run_dir: Path,
    plan_only: bool,
    submit_hardware: bool,
    backend_override: str | None,
) -> list[str]:
    input_config = require_mapping(config, "input")
    split = require_mapping(config, "split")
    common = require_mapping(config, "common")
    svm = require_mapping(config, "svm")
    qsvm = require_mapping(config, "qsvm")
    ibm = require_mapping(config, "ibm")
    execution = require_mapping(config, "execution")

    command = [
        sys.executable,
        "-m",
        BENCHMARK_MODULE,
        "--models", *models,
        "--csv", str(dataset),
        "--label", str(input_config.get("label", "label_site_id")),
        "--feature-sets", *strings(require_list(common, "feature_sets", "common")),
        "--ibm-feature-sets", *strings(require_list(ibm, "feature_sets", "ibm")),
        "--seeds", *numbers(require_list(split, "seeds", "split")),
        "--test-size", str(float(split.get("test_size", 0.25))),
        "--max-samples", str(int(split.get("max_samples", 0))),
        "--ibm-max-samples", str(int(ibm.get("max_samples", 12))),
        "--ibm-kernel-diagonal", str(ibm.get("kernel_diagonal", "ideal")),
        "--svm-kernels", *strings(require_list(svm, "kernels", "svm")),
        "--svm-scalers", *strings(require_list(svm, "scalers", "svm")),
        "--svm-gamma", str(svm.get("gamma", "scale")),
        "--svm-degree", str(int(svm.get("degree", 3))),
        "--svm-cv", str(int(svm.get("cv_folds", common.get("cv_folds", 5)))),
        "--qsvm-scalers", *strings(require_list(qsvm, "scalers", "qsvm")),
        "--entanglements", *strings(require_list(qsvm, "entanglements", "qsvm")),
        "--reps", *numbers(require_list(qsvm, "reps", "qsvm")),
        "--qsvm-qubits", *numbers(require_list(qsvm, "qubits", "qsvm")),
        "--qsvm-cv", str(int(qsvm.get("cv_folds", common.get("cv_folds", 5)))),
        "--qsvm-simulators", *strings(require_list(qsvm, "simulators", "qsvm")),
        "--mps-max-bond-dimension", str(int(qsvm.get("mps_max_bond_dimension", 0))),
        "--mps-truncation-threshold", str(float(qsvm.get("mps_truncation_threshold", 0.0))),
        "--ibm-qubits", *numbers(require_list(ibm, "qubits", "ibm")),
        "--shots", *numbers(require_list(ibm, "shots", "ibm")),
        "--optimization-level", str(int(ibm.get("optimization_level", 2))),
        "--transpiler-seed", str(int(ibm.get("transpiler_seed", 42))),
        "--max-circuits-per-job", str(int(ibm.get("max_circuits_per_job", 100))),
        "--ibm-trials", str(int(ibm.get("trials", 1))),
        "--C", str(float(common.get("c_parameter", 1.0))),
        "--class-weight", str(common.get("class_weight", "balanced")),
        "--timeout", str(float(execution.get("timeout_seconds", 300))),
        "--output-root", str(run_dir / "benchmark"),
    ]

    if bool(execution.get("fail_fast", True)):
        command.append("--fail-fast")
    if plan_only:
        command.append("--plan-only")
    if submit_hardware:
        command.append("--submit-hardware")

    backend = backend_override or ibm.get("backend")
    if backend:
        command.extend(["--backend", str(backend)])
    if ibm.get("account"):
        command.extend(["--ibm-account", str(ibm["account"])])
    if ibm.get("instance"):
        command.extend(["--ibm-instance", str(ibm["instance"])])
    if bool(ibm.get("force_ipv4", False)):
        command.append("--force-ipv4")
    return command


def evaluation_protocol(config: dict[str, Any], override: str | None) -> str:
    evaluation = config.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        fail("Configuration field 'evaluation' must be a mapping.")
    protocol = str(override or evaluation.get("protocol", "holdout"))
    if protocol not in VALID_PROTOCOLS:
        fail(f"Unknown evaluation.protocol: {protocol}; use one of {', '.join(VALID_PROTOCOLS)}.")
    return "grouped_27" if protocol == "grouped" else protocol


def evaluation_models(config: dict[str, Any], models: list[str]) -> list[str]:
    """Models of evaluate_grouped for a method: svm drops qsvm, qsvm keeps qsvm and majority."""
    evaluation = config.get("evaluation") or {}
    configured = strings(evaluation.get("models", list(EVALUATION_MODELS)))
    unknown = sorted(set(configured) - set(EVALUATION_MODELS))
    if unknown:
        fail(f"Unknown evaluation.models: {unknown}")
    if models == ["svm"]:
        return [name for name in configured if name != "qsvm"]
    if models == ["qsvm"]:
        return [name for name in configured if name in {"majority", "qsvm"}] or ["qsvm"]
    return configured


def build_evaluation_command(
    config: dict[str, Any],
    models: list[str],
    dataset: Path,
    run_dir: Path,
) -> list[str]:
    """Command for the grouped evaluation protocol (scripts.models.evaluate_grouped)."""
    input_config = require_mapping(config, "input")
    common = require_mapping(config, "common")
    qsvm = require_mapping(config, "qsvm")
    evaluation = config.get("evaluation") or {}
    selected = evaluation_models(config, models)
    feature_sets = strings(require_list(common, "feature_sets", "common"))
    command = [
        sys.executable,
        "-m",
        EVALUATION_MODULE,
        "--csv", str(dataset),
        "--label", str(input_config.get("label", "label_site_id")),
        "--group-column", str(input_config.get("group_column", "group")),
        "--feature-set", feature_sets[0],
        "--models", *selected,
        "--outer", str(evaluation.get("outer", "both")),
        "--preprocessing", str(evaluation.get("preprocessing", "power_minmax")),
        "--tuning", str(evaluation.get("tuning", "nested")),
        "--class-weight", str(common.get("class_weight", "balanced")),
        "--n-permutations", str(int(evaluation.get("n_permutations", 1000))),
        "--n-jobs", str(int(evaluation.get("n_jobs", 1))),
        "--random-state", str(int(evaluation.get("random_state", 42))),
        "--output-dir", str(run_dir / "evaluation"),
        "--run-name", "grouped_eval",
    ]
    if "qsvm" in selected:
        qubits = require_list(qsvm, "qubits", "qsvm")
        reps = require_list(qsvm, "reps", "qsvm")
        entanglements = require_list(qsvm, "entanglements", "qsvm")
        command += [
            "--qsvm-kernel", str(evaluation.get("qsvm_kernel", "statevector")),
            "--qsvm-qubits", str(int(qubits[0])),
            "--qsvm-reps", str(int(reps[0])),
            "--qsvm-entanglement", str(entanglements[0]),
        ]
    permuted = [name for name in strings(evaluation.get("permutation_models", [])) if name in selected]
    if permuted:
        command += ["--permutation-models", *permuted]
    if "compare" in evaluation:
        compare = [pair for pair in strings(evaluation.get("compare") or [])
                   if all(name in selected for name in pair.split(":"))]
        command += ["--compare", *compare]
    for key, flag in (("param_grid", "--param-grid"), ("fixed_params", "--fixed-params")):
        if evaluation.get(key):
            command += [flag, json.dumps(evaluation[key])]
    return command


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def output_inventory(root: Path) -> list[dict[str, Any]]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    )
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]


def run_command(command: list[str], log_path: Path) -> None:
    print(f"[command] {shlex.join(command)}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
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
        print("\n".join([*VALID_METHODS, "all"]))
        return

    mode, models, plan_only, hardware_mode = normalize_method(args.method)
    if args.submit_ibm and not hardware_mode:
        fail("--submit-ibm is accepted only with the ibm method.")
    if hardware_mode and not args.submit_ibm:
        fail("The ibm method requires the explicit --submit-ibm authorization flag.")

    config = load_config(args.config)
    input_config = require_mapping(config, "input")
    dataset = resolve_project_path(args.dataset or Path(str(input_config.get("dataset", ""))))
    label = str(input_config.get("label", "label_site_id"))
    common = require_mapping(config, "common")
    split = require_mapping(config, "split")
    ibm = require_mapping(config, "ibm")
    local_features = strings(require_list(common, "feature_sets", "common"))
    ibm_features = strings(require_list(ibm, "feature_sets", "ibm"))
    feature_sets = list(dict.fromkeys([*local_features, *(ibm_features if "qsvm_ibm" in models else [])]))
    dataset_summary = validate_dataset(
        dataset,
        label,
        feature_sets,
        int(split.get("max_samples", 0)),
        float(split.get("test_size", 0.25)),
    )

    output_config = require_mapping(config, "output")
    output_root = resolve_project_path(Path(str(output_config.get("root", "results/models"))))
    protected_root = (PROJECT_ROOT / "results").resolve()
    if not output_root.is_relative_to(protected_root):
        fail(f"Model output must stay below the project results directory: {protected_root}")
    run_id = validate_run_id(args.run_id or default_run_id(mode, output_root))
    run_dir = (output_root / run_id).resolve()

    backend = args.backend or ibm.get("backend")
    if hardware_mode and not backend:
        fail("A real IBM hardware run requires --backend or ibm.backend in the configuration.")

    command = build_command(
        config,
        models,
        dataset,
        run_dir,
        plan_only,
        hardware_mode,
        args.backend,
    )
    protocol = evaluation_protocol(config, args.protocol)
    # The grouped protocol is local only; IBM methods keep the hardware holdout.
    evaluation_command = (
        build_evaluation_command(config, models, dataset, run_dir)
        if protocol == "grouped_27" and "qsvm_ibm" not in models
        else None
    )

    print(f"[plan] config: {config['_config_path']}")
    print(f"[plan] mode: {mode}")
    print(f"[plan] models: {', '.join(models)}")
    print(f"[plan] dataset: {dataset}")
    print(f"[plan] dataset rows: {dataset_summary['rows']}")
    print(f"[plan] class counts: {dataset_summary['class_counts']}")
    print(f"[plan] output: {run_dir}")
    print(f"[plan] command: {shlex.join(command)}")
    print(f"[plan] protocol: {protocol}")
    if evaluation_command is not None:
        print(f"[plan] evaluation command: {shlex.join(evaluation_command)}")
    if plan_only:
        print("[plan] IBM mode: offline command and workload planning only.")
    if hardware_mode:
        print(f"[warning] IBM hardware submission authorized for backend: {backend}")

    if args.dry_run:
        print("[dry-run] Validation completed; no model or output directory was created.")
        return

    if run_dir.exists():
        fail(f"Model run directory already exists; choose another --run-id: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    config_path = Path(config["_config_path"])
    shutil.copy2(config_path, run_dir / "config.yaml")
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
        "mode": mode,
        "models": models,
        "ibm_submission": hardware_mode,
        "config_path": str(config_path),
        "config_snapshot": str(run_dir / "config.yaml"),
        "config_sha256": file_sha256(config_path),
        "environment": environment_name,
        "versions": runtime_versions(),
        "dataset": {
            "path": str(dataset),
            "bytes": dataset.stat().st_size,
            "sha256": file_sha256(dataset),
            **dataset_summary,
        },
        "command": command,
        "protocol": protocol,
        "evaluation_command": evaluation_command,
        "output_directory": str(run_dir),
    }
    write_manifest(manifest_path, manifest)

    try:
        run_command(command, run_dir / "logs" / "models.log")
        if evaluation_command is not None:
            run_command(evaluation_command, run_dir / "logs" / "evaluation.log")
        outputs = output_inventory(run_dir / "benchmark")
        if evaluation_command is not None:
            outputs += output_inventory(run_dir / "evaluation")
        if not outputs:
            raise RuntimeError("Benchmark completed without producing output files.")
        manifest["outputs"] = outputs
        manifest["status"] = "completed"
        write_manifest(manifest_path, manifest)
        print(f"[done] Model workflow completed: {run_dir}")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
