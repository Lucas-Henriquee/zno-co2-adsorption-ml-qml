from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.common.project_config import RESULTS_DIR
    from scripts.models.grouped_split import (
        DEFAULT_GROUP_COLUMN,
        grouped_train_test_split,
        load_grouped_dataset,
        split_table,
    )
except ModuleNotFoundError:  # executed as a file from outside the project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.common.project_config import RESULTS_DIR
    from scripts.models.grouped_split import (
        DEFAULT_GROUP_COLUMN,
        grouped_train_test_split,
        load_grouped_dataset,
        split_table,
    )


# Only per-window MD descriptors; see classical_svm_baseline.py for why the
# former cartesian-product feature sets were removed (D1).
FEATURE_SETS = {
    "md3": [
        "feature_md_window_c_surface_distance_mean_a", "feature_md_window_c_surface_distance_std_a",
        "feature_md_window_mean_oco_angle_deg",
    ],
    "md_mic5": [
        "feature_md_window_c_min_slab_distance_mean_a", "feature_md_window_o_min_metal_distance_mean_a",
        "feature_md_window_c_surface_distance_mean_a", "feature_md_window_co_bond_avg_mean_a",
        "feature_md_window_mean_oco_angle_deg",
    ],
}

ALL_SCALERS = ["none", "standard", "minmax", "minmax_pi", "power", "power_minmax"]
ALL_SVM_KERNELS = ["linear", "rbf", "poly", "sigmoid"]
ALL_ENTANGLEMENTS = ["linear", "circular", "full"]


def parse_args():
    p = argparse.ArgumentParser(description=("Orchestrate classical SVM, ideal QSVM, and IBM-hardware QSVM benchmarks."))

    p.add_argument("--models", nargs="+", default=["all"], choices=["all", "svm", "qsvm", "qsvm_ibm"])
    p.add_argument("--csv", type=Path, default=RESULTS_DIR / "datasets" / "md_window_unique_dataset.csv")
    p.add_argument("--label", default="label_site_id")
    p.add_argument(
        "--group-column",
        default=DEFAULT_GROUP_COLUMN,
        help="Independent-trajectory id; the shared split and every CV keep each group on one side.",
    )
    p.add_argument("--feature-sets", nargs="+", default=["md3"], choices=sorted(FEATURE_SETS))
    p.add_argument("--ibm-feature-sets", nargs="+", default=["md3"], choices=sorted(FEATURE_SETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--max-samples", type=int, default=12, help="Shared sample limit, spread over as many groups as possible. 0 = full data.")

    p.add_argument("--svm-kernels", nargs="+", default=["rbf"], choices=ALL_SVM_KERNELS)
    p.add_argument("--svm-scalers", nargs="+", default=["power"], choices=ALL_SCALERS)
    p.add_argument("--svm-gamma", default="scale", help="'scale', 'auto', or a positive float.")
    p.add_argument("--svm-degree", type=int, default=3)
    p.add_argument("--svm-cv", type=int, default=5)

    p.add_argument("--qsvm-scalers", nargs="+", default=["power_minmax"], choices=ALL_SCALERS)
    p.add_argument("--entanglements", nargs="+", default=["linear"], choices=ALL_ENTANGLEMENTS)
    p.add_argument("--reps", nargs="+", type=int, default=[1])
    p.add_argument("--qsvm-qubits", nargs="+", type=int, default=[0], help="0 = number of features.")
    p.add_argument("--qsvm-cv", type=int, default=0)
    p.add_argument("--qsvm-simulators", nargs="+", default=["reference"], choices=["reference", "aer_mps"], help="Local QSVM simulation implementation(s).",)
    p.add_argument("--mps-max-bond-dimension", type=int, default=0, help="Passed to aer_mps; 0 = unlimited/exact main benchmark.",)
    p.add_argument("--mps-truncation-threshold", type=float, default=0.0, help="Passed to aer_mps; 0 avoids deliberate MPS truncation.",)

    p.add_argument("--ibm-qubits", nargs="+", type=int, default=[3])
    p.add_argument("--shots", nargs="+", type=int, default=[512])
    p.add_argument("--backend", default=None)
    p.add_argument("--ibm-account", default=None, help="Name of the saved Qiskit Runtime account.",)
    p.add_argument("--ibm-instance", default=None, help=("Optional IBM instance CRN override. Prefer storing the CRN in --ibm-account."),)
    p.add_argument("--force-ipv4", action="store_true", help="Force IBM Runtime network connections to IPv4.",)
    p.add_argument("--optimization-level", type=int, default=2, choices=[0, 1, 2, 3])
    p.add_argument("--transpiler-seed", type=int, default=42)
    p.add_argument("--max-circuits-per-job", type=int, default=100)
    p.add_argument("--ibm-trials", type=int, default=1, help="Repeat each IBM configuration without changing the shared split.")
    p.add_argument(
        "--ibm-embedded-classical",
        action="store_true",
        help="Also run the small classical baseline embedded in each IBM process. Usually redundant when --models includes svm.",
    )
    p.add_argument("--submit-hardware", action="store_true", help="Actually submit IBM Runtime jobs. Without this, IBM runs use --dry-run.")
    p.add_argument("--allow-large-hardware", action="store_true", help="Required to submit hardware with --max-samples 0 or >100.")

    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    p.add_argument("--full-grid", action="store_true", help="Expand to all supported local scalers/kernels/topologies and common qubit settings.")
    p.add_argument("--plan-only", action="store_true", help="Print commands but do not execute any model.")
    p.add_argument("--timeout", type=float, default=300.0, help="Per-run timeout in seconds for SVM/QSVM local runs. 0 disables it.",)
    p.add_argument("--hardware-timeout", type=float, default=0.0, help=("Timeout for a REAL IBM hardware client process. 0 disables it recommended because killing the client does not necessarily cancel an already submitted Runtime job."),)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--output-root", type=Path, default=RESULTS_DIR / "benchmarks")
    return p.parse_args()


def selected_models(raw: list[str]) -> list[str]:
    return ["svm", "qsvm", "qsvm_ibm"] if "all" in raw else list(dict.fromkeys(raw))


def create_split(
    csv_path: Path,
    label: str,
    feature_set: str,
    max_samples: int,
    test_size: float,
    seed: int,
    split_dir: Path,
    group_column: str = DEFAULT_GROUP_COLUMN,
) -> Path:
    """Write one grouped train/test split shared by SVM, QSVM and IBM runs.

    Whole trajectories (groups) go to either train or test. Too few groups per
    class raises InsufficientGroupsError with the number of trajectories needed.
    """
    features = FEATURE_SETS[feature_set]
    df = load_grouped_dataset(csv_path, features, label, group_column)
    train_df, test_df = grouped_train_test_split(df, label, group_column, test_size, max_samples, seed)
    split = split_table(train_df, test_df, label, group_column)
    split_dir.mkdir(parents=True, exist_ok=True)
    size_tag = "all" if max_samples <= 0 else str(max_samples)
    path = split_dir / f"split_{feature_set}_n{size_tag}_test{test_size:g}_seed{seed}.csv"
    split.to_csv(path, index=False)
    return path


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Terminate the whole local process group, with a forced fallback."""
    if proc.poll() is not None:
        return

    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


def run_streaming(
    cmd: list[str],
    plan_only: bool,
    timeout_s: float,
) -> tuple[int, Path | None, str, float]:
    print("\n$ " + command_text(cmd), flush=True)
    if timeout_s > 0:
        print(f"[timeout] limit for this run: {timeout_s:g} s", flush=True)
    else:
        print("[timeout] disabled for this run", flush=True)

    if plan_only:
        return 0, None, "planned", 0.0

    start_time = time.monotonic()
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
        start_new_session=(os.name != "nt"),
    )

    output_queue: queue.Queue[str | None] = queue.Queue()
    result_dir = None

    def _reader():
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    reader_finished = False
    timed_out = False

    while True:
        elapsed = time.monotonic() - start_time

        if timeout_s > 0 and elapsed >= timeout_s and proc.poll() is None:
            timed_out = True
            print(
                f"\n[TIMEOUT] Run exceeded {timeout_s:g} s. "
                "Terminating it and continuing...",
                flush=True,
            )
            _terminate_process_tree(proc)

        try:
            item = output_queue.get(timeout=0.10)
        except queue.Empty:
            item = None if reader_finished else "__QUEUE_EMPTY__"

        if item == "__QUEUE_EMPTY__":
            pass
        elif item is None:
            reader_finished = True
        else:
            print(item, end="", flush=True)
            if item.startswith("RESULT_DIR="):
                result_dir = Path(item.strip().split("=", 1)[1])

        if proc.poll() is not None and reader_finished:
            break

    elapsed = time.monotonic() - start_time
    rc = proc.returncode if proc.returncode is not None else -1

    if timed_out:
        print(f"[status] timeout after {elapsed:.1f} s", flush=True)
        return rc, result_dir, "timeout", elapsed

    status = "ok" if rc == 0 else "failed"
    print(f"[status] {status} after {elapsed:.1f} s", flush=True)
    return rc, result_dir, status, elapsed


def safe_get(d: dict, *keys):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def summarize_result(
    model_kind: str,
    result_dir: Path | None,
    return_code: int,
    execution_status: str,
    elapsed_s: float,
) -> dict:
    row = {
        "requested_model": model_kind,
        "return_code": return_code,
        "status": execution_status,
        "benchmark_elapsed_s": float(elapsed_s),
        "_result_dir": None if result_dir is None else str(result_dir),
    }
    if result_dir is None:
        return row
    metrics_path = result_dir / ("run_metadata.json" if model_kind == "qsvm_ibm" else "metrics.json")
    if not metrics_path.exists():
        row["status"] = "no_metrics" if return_code == 0 else "failed"
        return row
    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        row["metrics_error"] = str(exc); return row

    row.update({
        "model": m.get("model"),
        "feature_set": m.get("feature_set"),
        "features_json": json.dumps(m.get("features"), ensure_ascii=False),
        "n_features": m.get("n_features"),
        "n_qubits": m.get("n_qubits"),
        "scaler": m.get("scaler"),
        "simulator": m.get("simulator"),
        "kernel": m.get("kernel"),
        "entanglement": m.get("entanglement"),
        "reps": m.get("reps"),
        "C": m.get("C"),
        "class_weight": m.get("class_weight"),
        "backend": m.get("backend"),
        "shots": m.get("shots"),
        "trial": m.get("trial"),
        "random_state": m.get("random_state"),
        "test_size": m.get("test_size"),
        "max_samples": m.get("max_samples"),
        "split_sha256": m.get("split_sha256"),
        "feature_map_fingerprint": m.get("feature_map_fingerprint"),
        "encoding": m.get("encoding"),
        "feature_map_depth_pretranspile": (
            m.get("feature_map_depth_pretranspile")
            if m.get("feature_map_depth_pretranspile") is not None
            else m.get("feature_map_depth")
        ),
        "feature_map_size_pretranspile": (
            m.get("feature_map_size_pretranspile")
            if m.get("feature_map_size_pretranspile") is not None
            else m.get("feature_map_size")
        ),
        "optimization_level": m.get("optimization_level"),
        "transpiler_seed": m.get("transpiler_seed"),
        "max_circuits_per_job": m.get("max_circuits_per_job"),
        "samples": m.get("samples"),
        "train_samples": m.get("train_samples"),
        "test_samples": m.get("test_samples"),
        "split_strategy": m.get("split_strategy"),
        "train_groups": safe_get(m, "grouping", "train_groups"),
        "test_groups": safe_get(m, "grouping", "test_groups"),
        "train_balanced_accuracy": safe_get(m, "train_metrics", "balanced_accuracy"),
        "test_accuracy": safe_get(m, "test_metrics", "accuracy"),
        "test_balanced_accuracy": safe_get(m, "test_metrics", "balanced_accuracy"),
        "test_f1_macro": safe_get(m, "test_metrics", "f1_macro"),
        "diagnosis": safe_get(m, "fit_diagnosis", "diagnosis"),
        "train_test_gap": safe_get(m, "fit_diagnosis", "train_test_gap"),
        "cv_balanced_accuracy_mean": m.get("cv_balanced_accuracy_mean"),
        "cv_balanced_accuracy_std": m.get("cv_balanced_accuracy_std"),
    })
    workload = m.get("kernel_workload_estimate")
    if not isinstance(workload, dict):
        workload = m.get("workload") if isinstance(m.get("workload"), dict) else {}
    row["fidelity_circuit_count"] = workload.get("approx_total_fidelity_entries")
    if row.get("fidelity_circuit_count") is not None and row.get("shots") is not None:
        row["requested_circuit_shots"] = int(row["fidelity_circuit_count"]) * int(row["shots"])
    runtime_jobs = m.get("runtime_jobs")
    row["runtime_job_count"] = len(runtime_jobs) if isinstance(runtime_jobs, list) else None
    timing = m.get("timing", {}) if isinstance(m.get("timing"), dict) else {}
    for key, value in timing.items():
        row[f"time_{key}"] = value

    preprocess_values = [
        timing.get("preprocess_train_wall_s"),
        timing.get("preprocess_test_wall_s"),
    ]
    row["preprocessing_time_s"] = (
        None
        if any(value is None for value in preprocess_values)
        else float(sum(preprocess_values))
    )
    row["kernel_time_s"] = timing.get("kernel_total_wall_s")
    row["training_time_s"] = timing.get("model_fit_wall_s")
    row["prediction_time_s"] = timing.get("predict_test_wall_s")
    row["quantum_time_s"] = timing.get("quantum_execution_s")
    row["transpilation_time_s"] = timing.get("transpile_wall_s")
    row["queue_time_s"] = timing.get("queue_wall_s")
    row["job_wait_time_s"] = timing.get("job_wait_wall_s")
    row["end_to_end_test_time_s"] = timing.get("end_to_end_test_wall_s")

    stats = m.get("transpiled_circuit_stats", {})
    if isinstance(stats, dict):
        row["train_transpiled_depth_mean"] = safe_get(stats, "K_train", "depth_mean")
        row["train_transpiled_2q_mean"] = safe_get(stats, "K_train", "two_qubit_gates_mean")
    if str(m.get("model", "")).endswith("DRY_RUN"):
        row["status"] = "dry_run"
    return row


FAIRNESS_FIELDS = [
    "feature_set",
    "features_json",
    "n_features",
    "n_qubits",
    "scaler",
    "encoding",
    "feature_map_fingerprint",
    "entanglement",
    "reps",
    "C",
    "class_weight",
    "random_state",
    "split_sha256",
    "train_samples",
    "test_samples",
]


def annotate_local_ibm_comparisons(records: list[dict]) -> None:
    local_rows = [
        row
        for row in records
        if row.get("requested_model") == "qsvm" and row.get("status") == "ok"
    ]

    for ibm_row in [
        row
        for row in records
        if row.get("requested_model") == "qsvm_ibm"
        and row.get("status") == "ok"
        and row.get("model") == "QSVM_IBM"
    ]:
        candidates = [
            row
            for row in local_rows
            if row.get("feature_set") == ibm_row.get("feature_set")
            and row.get("n_qubits") == ibm_row.get("n_qubits")
            and row.get("random_state") == ibm_row.get("random_state")
        ]
        if not candidates:
            ibm_row["fair_local_ibm_comparison"] = False
            ibm_row["fairness_issues"] = "no local QSVM candidate with the same feature set, qubits and split seed"
            continue

        # Prefer the exact reference implementation when several local
        # simulators were requested.
        candidates.sort(key=lambda row: row.get("simulator") != "reference")
        local_row = candidates[0]
        issues = [
            f"{field}: local={local_row.get(field)!r}, IBM={ibm_row.get(field)!r}"
            for field in FAIRNESS_FIELDS
            if local_row.get(field) != ibm_row.get(field)
        ]
        ibm_row["matched_local_simulator"] = local_row.get("simulator")
        ibm_row["fair_local_ibm_comparison"] = not issues
        ibm_row["fairness_issues"] = "; ".join(issues) if issues else None

        if issues or not ibm_row.get("_result_dir") or not local_row.get("_result_dir"):
            continue

        ideal_dir = Path(local_row["_result_dir"])
        hardware_dir = Path(ibm_row["_result_dir"])
        try:
            ktr_ideal = pd.read_csv(ideal_dir / "K_train.csv").to_numpy(float)
            kte_ideal = pd.read_csv(ideal_dir / "K_test.csv").to_numpy(float)
            ktr_hardware = pd.read_csv(hardware_dir / "K_train_raw.csv").to_numpy(float)
            kte_hardware = pd.read_csv(hardware_dir / "K_test.csv").to_numpy(float)
            if ktr_ideal.shape == ktr_hardware.shape:
                diff = ktr_hardware - ktr_ideal
                ibm_row["kernel_train_mae_vs_ideal"] = float(np.mean(np.abs(diff)))
                ibm_row["kernel_train_rmse_vs_ideal"] = float(np.sqrt(np.mean(diff ** 2)))
            if kte_ideal.shape == kte_hardware.shape:
                diff = kte_hardware - kte_ideal
                ibm_row["kernel_test_mae_vs_ideal"] = float(np.mean(np.abs(diff)))
                ibm_row["kernel_test_rmse_vs_ideal"] = float(np.sqrt(np.mean(diff ** 2)))
        except Exception as exc:
            ibm_row["kernel_comparison_error"] = str(exc)


def _matrix_mae(path_a: Path, path_b: Path) -> float | None:
    try:
        a = pd.read_csv(path_a).to_numpy(float)
        b = pd.read_csv(path_b).to_numpy(float)
        if a.shape != b.shape:
            return None
        return float(np.mean(np.abs(a - b)))
    except Exception:
        return None


def _prediction_disagreement(path_a: Path, path_b: Path) -> float | None:
    try:
        a = pd.read_csv(path_a)[["row_id", "qsvm_hardware_pred"]]
        b = pd.read_csv(path_b)[["row_id", "qsvm_hardware_pred"]]
        merged = a.merge(b, on="row_id", suffixes=("_a", "_b"), validate="one_to_one")
        if merged.empty:
            return None
        return float(
            np.mean(
                merged["qsvm_hardware_pred_a"].astype(str)
                != merged["qsvm_hardware_pred_b"].astype(str)
            )
        )
    except Exception:
        return None


def build_shot_sensitivity(records: list[dict]) -> list[dict]:
    hardware_rows = [
        row
        for row in records
        if row.get("requested_model") == "qsvm_ibm"
        and row.get("status") == "ok"
        and row.get("model") == "QSVM_IBM"
        and row.get("shots") is not None
        and row.get("_result_dir")
    ]
    group_fields = [
        "feature_set",
        "n_qubits",
        "scaler",
        "entanglement",
        "reps",
        "C",
        "class_weight",
        "backend",
        "random_state",
        "split_sha256",
        "trial",
    ]
    groups: dict[tuple, list[dict]] = {}
    for row in hardware_rows:
        key = tuple(row.get(field) for field in group_fields)
        groups.setdefault(key, []).append(row)

    output = []
    for key, rows in groups.items():
        if len({int(row["shots"]) for row in rows}) < 2:
            continue
        reference = max(rows, key=lambda row: int(row["shots"]))
        reference_dir = Path(reference["_result_dir"])
        for row in sorted(rows, key=lambda item: int(item["shots"])):
            run_dir = Path(row["_result_dir"])
            item = dict(zip(group_fields, key))
            item.update({
                "shots": int(row["shots"]),
                "reference_shots": int(reference["shots"]),
                "test_balanced_accuracy": row.get("test_balanced_accuracy"),
                "delta_balanced_accuracy_vs_max_shots": (
                    None
                    if row.get("test_balanced_accuracy") is None
                    or reference.get("test_balanced_accuracy") is None
                    else float(row["test_balanced_accuracy"] - reference["test_balanced_accuracy"])
                ),
                "test_f1_macro": row.get("test_f1_macro"),
                "delta_f1_macro_vs_max_shots": (
                    None
                    if row.get("test_f1_macro") is None
                    or reference.get("test_f1_macro") is None
                    else float(row["test_f1_macro"] - reference["test_f1_macro"])
                ),
                "prediction_disagreement_vs_max_shots": _prediction_disagreement(
                    run_dir / "predictions.csv",
                    reference_dir / "predictions.csv",
                ),
                "kernel_train_mae_vs_max_shots": _matrix_mae(
                    run_dir / "K_train_raw.csv",
                    reference_dir / "K_train_raw.csv",
                ),
                "kernel_test_mae_vs_max_shots": _matrix_mae(
                    run_dir / "K_test.csv",
                    reference_dir / "K_test.csv",
                ),
                "quantum_time_s": row.get("quantum_time_s"),
                "job_wait_time_s": row.get("job_wait_time_s"),
            })
            output.append(item)
    return output


def main():
    args = parse_args()

    # Apply presets before validating cross-model equivalence. The previous
    # order validated the defaults and could reject a valid --full-grid plan.
    if args.full_grid:
        args.feature_sets = sorted(FEATURE_SETS)
        args.svm_kernels = ALL_SVM_KERNELS
        args.svm_scalers = ALL_SCALERS
        args.qsvm_scalers = ALL_SCALERS
        args.entanglements = ALL_ENTANGLEMENTS
        args.reps = [1, 2]
        args.qsvm_qubits = [0, 12, 20]
        args.qsvm_simulators = ["reference"]
        args.ibm_feature_sets = ["md3"]
        args.ibm_qubits = [3, 12, 20]

    models = selected_models(args.models)

    if not 0.0 < args.test_size < 1.0:
        raise SystemExit("--test-size must be between 0 and 1.")
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be >= 0.")
    if args.C <= 0 or args.ibm_trials < 1 or args.max_circuits_per_job < 1:
        raise SystemExit("--C, --ibm-trials and --max-circuits-per-job must be positive.")
    if any(value < 0 for value in args.qsvm_qubits + args.ibm_qubits):
        raise SystemExit("Qubit counts must be >= 0 (0 means one qubit per feature).")
    if any(value <= 0 for value in args.shots):
        raise SystemExit("Every shots value must be positive.")
    if any(value < 1 for value in args.reps):
        raise SystemExit("Every feature-map reps value must be positive.")

    if "qsvm" in models and "qsvm_ibm" in models:
        missing_feature_sets = set(args.ibm_feature_sets) - set(args.feature_sets)

        if missing_feature_sets:
            raise SystemExit(
                "Every IBM feature set must also have a matching local QSVM run. "
                f"Missing locally: {sorted(missing_feature_sets)}"
            )

        for feature_set in args.ibm_feature_sets:
            n_features = len(FEATURE_SETS[feature_set])

            local_qubits = {
                n_features if value == 0 else value
                for value in args.qsvm_qubits
            }
            ibm_qubits = {
                n_features if value == 0 else value
                for value in args.ibm_qubits
            }

            missing_local = ibm_qubits - local_qubits

            if missing_local:
                raise SystemExit(
                    f"Biased comparison for {feature_set}: IBM requests "
                    f"{sorted(missing_local)} qubits without an equivalent "
                    "local QSVM configuration."
                )

    if (
        args.submit_hardware
        and "qsvm_ibm" in models
        and (len(set(args.shots)) > 1 or args.ibm_trials > 1)
        and not args.backend
    ):
        raise SystemExit(
            "A shot-sensitivity/repeated hardware study requires --backend so "
            "every IBM run uses the same QPU."
        )

    if args.submit_hardware and "qsvm_ibm" in models and (args.max_samples <= 0 or args.max_samples > 100) and not args.allow_large_hardware:
        raise SystemExit("Refusing a large IBM submission. Use a small --max-samples or explicitly add --allow-large-hardware.")

    index = 1
    while (args.output_root / f"benchmark_{index:02d}").exists():
        index += 1
    suite_dir = args.output_root / f"benchmark_{index:02d}"
    split_dir = suite_dir / "splits"
    suite_dir.mkdir(parents=True, exist_ok=False)

    script_dir = Path(__file__).resolve().parent.parent
    scripts = {
        "svm": script_dir / "ml" / "classical_svm_baseline.py",
        "qsvm": script_dir / "qml" / "qsvm_baseline.py",
        "qsvm_ibm": script_dir / "qml_ibm" / "qsvm_ibm_hardware.py",
    }
    missing_scripts = [str(path) for key, path in scripts.items() if key in models and not path.exists()]
    if missing_scripts:
        raise SystemExit("Missing benchmark scripts: " + ", ".join(missing_scripts))

    split_cache: dict[tuple[str, int], Path] = {}
    def split_for(feature_set: str, seed: int) -> Path:
        key = (feature_set, seed)
        if key not in split_cache:
            try:
                split_cache[key] = create_split(
                    args.csv, args.label, feature_set, args.max_samples, args.test_size, seed, split_dir, args.group_column
                )
            except ValueError as exc:
                raise SystemExit(f"[error] Cannot build a grouped split for {feature_set}: {exc}") from exc
        return split_cache[key]

    commands: list[tuple[str, list[str]]] = []
    hardware_plan: list[dict] = []
    py = sys.executable

    if "svm" in models:
        for fs, kernel, scaler, seed in itertools.product(args.feature_sets, args.svm_kernels, args.svm_scalers, args.seeds):
            split = split_for(fs, seed)
            run_name = f"svm_{fs}_{kernel}_{scaler}_seed{seed}"
            commands.append(("svm", [
                py, str(scripts["svm"]), "--csv", str(args.csv), "--label", args.label,
                "--group-column", args.group_column,
                "--feature-set", fs, "--kernel", kernel, "--scaler", scaler, "--C", str(args.C),
                "--gamma", str(args.svm_gamma), "--degree", str(args.svm_degree),
                "--class-weight", args.class_weight, "--cv", str(args.svm_cv), "--split-file", str(split),
                "--test-size", str(args.test_size), "--max-samples", str(args.max_samples),
                "--random-state", str(seed), "--output-dir", str(suite_dir / "svm"), "--run-name", run_name,
            ]))

    if "qsvm" in models:
        seen_qsvm = set()
        for fs, scaler, ent, reps, nq, simulator_name, seed in itertools.product(
            args.feature_sets,
            args.qsvm_scalers,
            args.entanglements,
            args.reps,
            args.qsvm_qubits,
            args.qsvm_simulators,
            args.seeds,
        ):
            n_features = len(FEATURE_SETS[fs])
            resolved = n_features if nq == 0 else nq
            if resolved < n_features:
                print(f"[skip] QSVM {fs}: {resolved} qubits < {n_features} features")
                continue
            config_key = (fs, scaler, ent, reps, resolved, simulator_name, seed)
            if config_key in seen_qsvm:
                continue
            seen_qsvm.add(config_key)
            split = split_for(fs, seed)
            run_name = f"qsvm_{fs}_{resolved}q_{ent}_r{reps}_{scaler}_{simulator_name}_seed{seed}"
            commands.append(("qsvm", [
                py,
                str(scripts["qsvm"]),
                "--csv", str(args.csv),
                "--label", args.label,
                "--group-column", args.group_column,
                "--feature-set", fs,
                "--scaler", scaler,
                "--entanglement", ent,
                "--reps", str(reps),
                "--n-qubits", str(nq),
                "--C", str(args.C),
                "--class-weight", args.class_weight,
                "--cv", str(args.qsvm_cv),
                "--simulator", simulator_name,
                "--mps-max-bond-dimension", str(args.mps_max_bond_dimension),
                "--mps-truncation-threshold", str(args.mps_truncation_threshold),
                "--split-file", str(split),
                "--random-state", str(seed),
                "--test-size", str(args.test_size),
                "--max-samples", str(args.max_samples),
                "--output-dir", str(suite_dir / "qsvm"),
                "--run-name", run_name,
            ]))

    if "qsvm_ibm" in models:
        seen_ibm = set()
        for fs, scaler, ent, reps, nq, shots, seed, trial in itertools.product(
            args.ibm_feature_sets,
            args.qsvm_scalers,
            args.entanglements,
            args.reps,
            args.ibm_qubits,
            args.shots,
            args.seeds,
            range(1, args.ibm_trials + 1),
        ):
            n_features = len(FEATURE_SETS[fs])
            resolved = n_features if nq == 0 else nq
            if resolved < n_features:
                print(f"[skip] IBM {fs}: {resolved} qubits < {n_features} features")
                continue
            config_key = (fs, scaler, ent, reps, resolved, shots, seed, trial)
            if config_key in seen_ibm:
                continue
            seen_ibm.add(config_key)
            split = split_for(fs, seed)
            split_table = pd.read_csv(split)
            n_train = int((split_table["split"] == "train").sum())
            n_test = int((split_table["split"] == "test").sum())
            fidelity_circuits = n_train * (n_train - 1) // 2 + n_test * n_train
            hardware_plan.append({
                "feature_set": fs,
                "n_qubits": resolved,
                "shots": shots,
                "trial": trial,
                "fidelity_circuits": fidelity_circuits,
                "requested_circuit_shots": fidelity_circuits * shots,
            })
            run_name = f"ibm_{fs}_{resolved}q_{ent}_r{reps}_{scaler}_{shots}shots_seed{seed}_trial{trial}"
            cmd = [
                py,
                str(scripts["qsvm_ibm"]),
                "--csv", str(args.csv),
                "--label", args.label,
                "--group-column", args.group_column,
                "--feature-set", fs,
                "--scaler", scaler,
                "--entanglement", ent,
                "--reps", str(reps),
                "--n-qubits", str(nq),
                "--shots", str(shots),
                "--C", str(args.C),
                "--class-weight", args.class_weight,
                "--split-file", str(split),
                "--random-state", str(seed),
                "--test-size", str(args.test_size),
                "--max-samples", str(args.max_samples),
                "--optimization-level", str(args.optimization_level),
                "--max-circuits-per-job", str(args.max_circuits_per_job),
                "--transpiler-seed", str(args.transpiler_seed),
                "--trial", str(trial),
                "--output-dir", str(suite_dir / "qsvm_ibm"),
                "--run-name", run_name,
            ]
            if args.backend:
                cmd += ["--backend", args.backend]

            if args.ibm_account:
                cmd += ["--ibm-account", args.ibm_account]

            if args.ibm_instance:
                cmd += ["--ibm-instance", args.ibm_instance]

            if args.force_ipv4:
                cmd.append("--force-ipv4")

            if not args.ibm_embedded_classical:
                cmd.append("--skip-classical-baseline")
            if not args.submit_hardware:
                cmd.append("--dry-run")
            commands.append(("qsvm_ibm", cmd))

    print("\n===== BENCHMARK PLAN =====")
    print(f"Models: {models}")
    print(f"Commands: {len(commands)}")
    print(f"Suite directory: {suite_dir}")
    if "qsvm_ibm" in models:
        print("IBM mode: " + ("REAL HARDWARE SUBMISSION" if args.submit_hardware else "DRY-RUN (safe, no QPU submission)"))
        print(f"IBM configurations: {len(hardware_plan)}")
        print(f"Estimated fidelity circuits: {sum(item['fidelity_circuits'] for item in hardware_plan)}")
        print(f"Requested circuit-shots: {sum(item['requested_circuit_shots'] for item in hardware_plan)}")

    records = []
    for index, (kind, cmd) in enumerate(commands, 1):
        print(f"\n===== RUN {index}/{len(commands)}: {kind} =====")

        if kind == "qsvm_ibm" and args.submit_hardware:
            timeout_s = args.hardware_timeout
        else:
            timeout_s = args.timeout

        rc, result_dir, execution_status, elapsed_s = run_streaming(
            cmd,
            args.plan_only,
            timeout_s,
        )
        records.append(
            summarize_result(
                kind,
                result_dir,
                rc,
                execution_status,
                elapsed_s,
            )
        )

        if execution_status not in {"ok", "planned"} and args.fail_fast:
            break

    annotate_local_ibm_comparisons(records)
    shot_sensitivity = build_shot_sensitivity(records)

    if records:
        export_records = [
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in records
        ]
        summary_df = pd.DataFrame(export_records)
        summary_df.to_csv(
            suite_dir / "benchmark_summary.csv",
            index=False,
        )
        engineering_columns = [
            "requested_model",
            "model",
            "status",
            "feature_set",
            "n_features",
            "n_qubits",
            "simulator",
            "backend",
            "shots",
            "trial",
            "scaler",
            "encoding",
            "entanglement",
            "reps",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1_macro",
            "train_balanced_accuracy",
            "train_test_gap",
            "split_strategy",
            "train_groups",
            "test_groups",
            "preprocessing_time_s",
            "kernel_time_s",
            "training_time_s",
            "prediction_time_s",
            "transpilation_time_s",
            "queue_time_s",
            "quantum_time_s",
            "job_wait_time_s",
            "end_to_end_test_time_s",
            "feature_map_depth_pretranspile",
            "feature_map_size_pretranspile",
            "train_transpiled_depth_mean",
            "train_transpiled_2q_mean",
            "fidelity_circuit_count",
            "requested_circuit_shots",
            "runtime_job_count",
            "fair_local_ibm_comparison",
            "fairness_issues",
            "kernel_train_mae_vs_ideal",
            "kernel_test_mae_vs_ideal",
        ]
        summary_df.reindex(columns=engineering_columns).to_csv(
            suite_dir / "engineering_comparison.csv",
            index=False,
        )
    if shot_sensitivity:
        pd.DataFrame(shot_sensitivity).to_csv(
            suite_dir / "shot_sensitivity.csv",
            index=False,
        )

    print("\n===== BENCHMARK SUITE COMPLETE =====")
    print(f"Commands planned/executed: {len(records)}")
    print(f"Summary will be saved at: {suite_dir / 'benchmark_summary.csv'}")
    print(f"Engineering table will be saved at: {suite_dir / 'engineering_comparison.csv'}")
    if shot_sensitivity:
        print(f"Shot sensitivity will be saved at: {suite_dir / 'shot_sensitivity.csv'}")
    print(f"Suite directory: {suite_dir}")
    print(f"SUITE_DIR={suite_dir}")


if __name__ == "__main__":
    main()
