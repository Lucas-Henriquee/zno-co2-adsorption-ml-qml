from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import uuid
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import SVC

# qiskit and qiskit-ibm-runtime are imported lazily inside the functions that
# build circuits or contact IBM, so data loading and splitting work without them.
try:
    from scripts.common.project_config import RESULTS_DIR
    from scripts.models.grouped_split import (
        DEFAULT_GROUP_COLUMN,
        group_summary,
        grouped_train_test_split,
        load_grouped_dataset,
        split_table,
    )
except ModuleNotFoundError:  # executed as a file from outside the project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.common.project_config import RESULTS_DIR
    from scripts.models.grouped_split import (
        DEFAULT_GROUP_COLUMN,
        group_summary,
        grouped_train_test_split,
        load_grouped_dataset,
        split_table,
    )


# Only per-window MD descriptors; see classical_svm_baseline.py for why the
# former cartesian-product feature sets were removed (D1).
FEATURE_SETS = {
    "md3": [
        "feature_md_window_c_surface_distance_mean_a",
        "feature_md_window_c_surface_distance_std_a",
        "feature_md_window_mean_oco_angle_deg",
    ],
    "md_mic5": [
        "feature_md_window_c_min_slab_distance_mean_a",
        "feature_md_window_o_min_metal_distance_mean_a",
        "feature_md_window_c_surface_distance_mean_a",
        "feature_md_window_co_bond_avg_mean_a",
        "feature_md_window_mean_oco_angle_deg",
    ],
}

SCALERS = ["none", "standard", "minmax", "minmax_pi", "power", "power_minmax"]
ENTANGLEMENTS = ["linear", "circular", "full"]


def parse_args():
    parser = argparse.ArgumentParser(description="General quantum-kernel SVM benchmark on real IBM Quantum hardware.")
    parser.add_argument("--csv", type=Path, default=RESULTS_DIR / "datasets" / "md_window_unique_dataset.csv")
    parser.add_argument("--label", default="label_site_id")
    parser.add_argument(
        "--group-column",
        default=DEFAULT_GROUP_COLUMN,
        help="Column with the independent-trajectory id; train/test never split a group.",
    )
    parser.add_argument("--feature-set", default="md3", choices=sorted(FEATURE_SETS))
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--split-file", type=Path, default=None, help="Optional CSV with row_id and split=train/test; must not share groups.")
    parser.add_argument("--scaler", default="power_minmax", choices=SCALERS)

    parser.add_argument("--n-qubits", type=int, default=0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--entanglement", default="linear", choices=ENTANGLEMENTS)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])

    parser.add_argument("--backend", default=None)
    parser.add_argument("--ibm-account", default=None, help=("Name of a saved Qiskit Runtime account. If omitted, use the default account."),)
    parser.add_argument("--ibm-instance", default=None, help=("Optional IBM instance CRN override. Prefer saving the CRN in --ibm-account."),)
    parser.add_argument("--force-ipv4", action="store_true", help="Force urllib3/IBM Runtime connections to IPv4.",)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--optimization-level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--transpiler-seed", type=int, default=42, help="Fixes stochastic transpiler choices across shot-sensitivity runs.",)
    parser.add_argument("--max-circuits-per-job", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--classical-kernel", default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--classical-gamma", default="scale")
    parser.add_argument("--skip-classical-baseline", action="store_true")

    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "qsvm_hardware")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--trial", type=int, default=1, help="Hardware repetition index")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None

def configure_runtime_network(force_ipv4: bool) -> None:
    if not force_ipv4:
        return

    import socket
    import urllib3.util.connection as urllib3_connection

    urllib3_connection.allowed_gai_family = lambda: socket.AF_INET
    print(
        "[network] IBM Runtime connections forced to IPv4.",
        flush=True,
    )


def create_runtime_service(
    account_name: str | None,
    instance: str | None,
):
    service_kwargs = {}

    if account_name:
        service_kwargs["name"] = account_name
    else:
        service_kwargs["channel"] = "ibm_quantum_platform"

    if instance:
        service_kwargs["instance"] = instance

    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService(**service_kwargs)    


def file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_weight_value(name: str):
    return None if name == "none" else name


def parse_gamma(value: str):
    if value in {"scale", "auto"}:
        return value
    gamma = float(value)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return gamma


def get_features(args) -> list[str]:
    return list(args.features) if args.features is not None else FEATURE_SETS[args.feature_set].copy()


def diagnose_fit(train_score: float, test_score: float,
                 gap_threshold: float = 0.10, low_threshold: float = 0.60) -> dict:
    gap = train_score - test_score
    if train_score < low_threshold and test_score < low_threshold:
        diagnosis = "possible_underfitting"
    elif gap > gap_threshold:
        diagnosis = "possible_overfitting"
    else:
        diagnosis = "no_obvious_train_test_gap"
    return {
        "train_score": float(train_score),
        "test_score": float(test_score),
        "train_test_gap": float(gap),
        "diagnosis": diagnosis,
    }


def metric_bundle(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def load_base_data(args, features: list[str]) -> pd.DataFrame:
    return load_grouped_dataset(args.csv, features, args.label, args.group_column)


def split_data(args, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return grouped_train_test_split(
        df,
        args.label,
        args.group_column,
        args.test_size,
        args.max_samples,
        args.random_state,
        args.split_file,
    )


def fit_transform_scaler(X_train, X_test, scaler_name: str):
    Xtr, Xte = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
    if scaler_name == "none":
        train_start = time.perf_counter(); Xtr_out = Xtr.copy(); train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter(); Xte_out = Xte.copy(); test_wall = time.perf_counter() - test_start
        return Xtr_out, Xte_out, train_wall, test_wall
    if scaler_name == "standard":
        s = StandardScaler()
        train_start = time.perf_counter(); Xtr_out = s.fit_transform(Xtr); train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter(); Xte_out = s.transform(Xte); test_wall = time.perf_counter() - test_start
        return Xtr_out, Xte_out, train_wall, test_wall
    if scaler_name == "minmax":
        s = MinMaxScaler((0.0, 1.0))
        train_start = time.perf_counter(); Xtr_out = s.fit_transform(Xtr); train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter(); Xte_out = s.transform(Xte); test_wall = time.perf_counter() - test_start
        return Xtr_out, Xte_out, train_wall, test_wall
    if scaler_name == "minmax_pi":
        s = MinMaxScaler((0.0, np.pi))
        train_start = time.perf_counter(); Xtr_out = s.fit_transform(Xtr); train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter(); Xte_out = s.transform(Xte); test_wall = time.perf_counter() - test_start
        return Xtr_out, Xte_out, train_wall, test_wall
    if scaler_name == "power":
        s = PowerTransformer(method="yeo-johnson", standardize=True)
        train_start = time.perf_counter(); Xtr_out = s.fit_transform(Xtr); train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter(); Xte_out = s.transform(Xte); test_wall = time.perf_counter() - test_start
        return Xtr_out, Xte_out, train_wall, test_wall
    if scaler_name == "power_minmax":
        power = PowerTransformer(method="yeo-johnson", standardize=True)
        train_start = time.perf_counter()
        trp = power.fit_transform(Xtr)
        mm = MinMaxScaler((0.0, np.pi)); Xtr_out = mm.fit_transform(trp)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        tep = power.transform(Xte); Xte_out = mm.transform(tep)
        test_wall = time.perf_counter() - test_start
        return Xtr_out, Xte_out, train_wall, test_wall
    raise ValueError(f"Unknown scaler: {scaler_name}")


def entanglement_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []
    if entanglement == "linear":
        return [(q, q + 1) for q in range(n_qubits - 1)]
    if entanglement == "circular":
        pairs = [(q, q + 1) for q in range(n_qubits - 1)]
        if n_qubits > 2:
            pairs.append((n_qubits - 1, 0))
        return pairs
    if entanglement == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    raise ValueError(f"Unsupported entanglement: {entanglement}")


def build_multiqubit_feature_map(n_features: int, n_qubits: int, reps: int, entanglement: str) -> QuantumCircuit:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    if n_qubits < n_features:
        raise ValueError("n_qubits must be >= n_features")
    x = ParameterVector("x", n_features)
    qc = QuantumCircuit(n_qubits, name=f"multi_zz_{n_features}f_{n_qubits}q")
    pairs = entanglement_pairs(n_qubits, entanglement)
    for rep in range(reps):
        for q in range(n_qubits):
            k = q % n_features
            qc.h(q)
            qc.p(2.0 * x[k], q)
        for q1, q2 in pairs:
            f1, f2 = q1 % n_features, q2 % n_features
            angle = 2.0 * (np.pi - x[f1]) * (np.pi - x[f2])
            qc.cx(q1, q2)
            qc.p(angle, q2)
            qc.cx(q1, q2)
        if rep != reps - 1:
            qc.barrier()
    if qc.num_parameters != n_features:
        raise RuntimeError("Unexpected number of feature-map parameters.")
    return qc


def build_feature_map(n_features: int, requested_qubits: int, reps: int, entanglement: str):
    from qiskit.circuit.library import zz_feature_map

    n_qubits = n_features if requested_qubits == 0 else requested_qubits
    if n_qubits < n_features:
        raise ValueError(f"n_qubits={n_qubits} cannot be smaller than n_features={n_features}")
    if n_qubits == n_features:
        return zz_feature_map(feature_dimension=n_features, reps=reps, entanglement=entanglement), "standard_zz_feature_map"
    return build_multiqubit_feature_map(n_features, n_qubits, reps, entanglement), "custom_multiqubit_zz_reupload"


def feature_map_fingerprint(circuit: QuantumCircuit) -> str:

    instructions = []
    for item in circuit.data:
        operation = item.operation
        instructions.append({
            "name": operation.name,
            "qubits": [circuit.find_bit(bit).index for bit in item.qubits],
            "clbits": [circuit.find_bit(bit).index for bit in item.clbits],
            "params": [str(value) for value in operation.params],
        })
    payload = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "global_phase": str(circuit.global_phase),
        "instructions": instructions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pick_backend(service, requested_name: str | None, n_qubits: int):
    if requested_name:
        backend = service.backend(requested_name)
        status = backend.status()
        if not status.operational:
            raise RuntimeError(f"Requested backend {requested_name!r} is not operational")
        if backend.num_qubits < n_qubits:
            raise RuntimeError(f"Backend {backend.name} has {backend.num_qubits} qubits; run requests {n_qubits}")
        return backend
    return service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)


def fit_classical_baseline(args, X_train, y_train, X_test, y_test):
    model = SVC(
        kernel=args.classical_kernel,
        C=args.C,
        gamma=parse_gamma(args.classical_gamma),
        class_weight=class_weight_value(args.class_weight),
        random_state=args.random_state,
    )

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_wall = time.perf_counter() - fit_start

    train_predict_start = time.perf_counter()
    pred_train = model.predict(X_train)
    predict_train_wall = time.perf_counter() - train_predict_start

    test_predict_start = time.perf_counter()
    pred_test = model.predict(X_test)
    predict_test_wall = time.perf_counter() - test_predict_start

    elapsed = fit_wall + predict_train_wall + predict_test_wall
    train_metrics, test_metrics = metric_bundle(y_train, pred_train), metric_bundle(y_test, pred_test)
    return {
        "pred_train": pred_train,
        "pred_test": pred_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cpu_wall_time_s": float(elapsed),
        "model_fit_wall_s": float(fit_wall),
        "predict_train_wall_s": float(predict_train_wall),
        "predict_test_wall_s": float(predict_test_wall),
        "fit_diagnosis": diagnose_fit(train_metrics["balanced_accuracy"], test_metrics["balanced_accuracy"]),
    }


def _project_to_psd(kernel_matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (kernel_matrix + kernel_matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, 0.0)
    return (eigvecs @ np.diag(eigvals) @ eigvecs.T).real


def _bind_feature_map(feature_map: QuantumCircuit, values: np.ndarray) -> QuantumCircuit:
    params = list(feature_map.parameters)
    if len(values) != len(params):
        raise ValueError(f"Expected {len(params)} feature values, received {len(values)}")
    bound = feature_map.assign_parameters(dict(zip(params, np.asarray(values, dtype=float))), inplace=False)
    if bound.num_parameters != 0:
        raise RuntimeError(f"Feature map still has free parameters: {list(bound.parameters)}")
    return bound


def _make_bound_fidelity_circuit(feature_map: QuantumCircuit, x: np.ndarray, y: np.ndarray) -> QuantumCircuit:
    ux, uy = _bind_feature_map(feature_map, x), _bind_feature_map(feature_map, y)
    circuit = ux.compose(uy.inverse())
    circuit.global_phase = 0
    circuit.measure_all()
    if circuit.num_parameters != 0:
        raise RuntimeError("Fidelity circuit unexpectedly contains free parameters")
    return circuit


def _extract_all_zero_probability(pub_result, n_measured_qubits: int) -> float:
    counts = pub_result.data.meas.get_counts()
    total = int(sum(counts.values()))
    if total <= 0:
        raise RuntimeError("SamplerV2 returned zero total shots")
    return float(counts.get("0" * n_measured_qubits, 0) / total)


def _circuit_stats(circuit: QuantumCircuit) -> dict:
    two_qubit = sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)
    return {
        "depth": int(circuit.depth() or 0),
        "size": int(circuit.size()),
        "two_qubit_gates": int(two_qubit),
    }


def _summarize_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out = {"n_circuits": len(rows)}
    for key in ["depth", "size", "two_qubit_gates"]:
        values = np.asarray([r[key] for r in rows], dtype=float)
        out[f"{key}_min"] = float(values.min())
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_max"] = float(values.max())
    return out


def _seconds_between(start, end) -> float | None:
    if start is None or end is None:
        return None
    try:
        return max(0.0, float(pd.Timestamp(end).timestamp() - pd.Timestamp(start).timestamp()))
    except Exception:
        return None


def _runtime_job_record(job, submit_wall_s: float, wait_wall_s: float) -> dict:
    record = {
        "job_id": None,
        "status": None,
        "client_submit_wall_s": float(submit_wall_s),
        "client_wait_wall_s": float(wait_wall_s),
        "queue_wall_s": None,
        "service_run_wall_s": None,
        "quantum_seconds": None,
        "metrics": None,
    }
    try:
        record["job_id"] = job.job_id()
    except Exception:
        pass
    try:
        record["status"] = str(job.status())
    except Exception:
        pass

    try:
        metrics = job.metrics()
        record["metrics"] = metrics
    except Exception as exc:
        record["metrics_error"] = str(exc)
        metrics = {}

    if isinstance(metrics, dict):
        usage = metrics.get("usage")
        if isinstance(usage, dict):
            value = usage.get("quantum_seconds")
            try:
                record["quantum_seconds"] = None if value is None else float(value)
            except (TypeError, ValueError):
                pass

        timestamps = metrics.get("timestamps")
        if isinstance(timestamps, dict):
            created = timestamps.get("created") or timestamps.get("created_at")
            running = (
                timestamps.get("running")
                or timestamps.get("running_at")
                or timestamps.get("started")
                or timestamps.get("started_at")
            )
            finished = timestamps.get("finished") or timestamps.get("finished_at")
            record["queue_wall_s"] = _seconds_between(created, running)
            record["service_run_wall_s"] = _seconds_between(running, finished)

    if record["quantum_seconds"] is None:
        try:
            estimation = job.usage_estimation
            estimation = estimation() if callable(estimation) else estimation
            if isinstance(estimation, dict):
                value = estimation.get("quantum_seconds")
                if value is not None:
                    record["quantum_seconds"] = float(value)
        except Exception:
            pass
    return record


def _sum_available(rows: list[dict], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return None if not values else float(sum(float(value) for value in values))


def _run_fidelity_pairs_on_hardware(pairs, feature_map, sampler, pass_manager, shots, max_circuits_per_job, label):
    fidelities, stats, jobs = [], [], []
    phase_timing = {
        "circuit_build_wall_s": 0.0,
        "transpile_wall_s": 0.0,
        "submit_wall_s": 0.0,
        "job_wait_wall_s": 0.0,
    }
    total_pairs = len(pairs)
    for start in range(0, total_pairs, max_circuits_per_job):
        stop = min(start + max_circuits_per_job, total_pairs)
        chunk_pairs = pairs[start:stop]
        print(f"[QPU] {label}: preparing/transpiling pairs {start + 1}-{stop}/{total_pairs} ...")

        build_start = time.perf_counter()
        bound_circuits = [
            _make_bound_fidelity_circuit(feature_map, x, y)
            for x, y in chunk_pairs
        ]
        phase_timing["circuit_build_wall_s"] += time.perf_counter() - build_start

        transpile_start = time.perf_counter()
        isa_circuits = []
        for bound in bound_circuits:
            isa = pass_manager.run(bound)
            if isa.num_parameters != 0:
                raise RuntimeError("ISA circuit still contains free parameters; refusing submission")
            stats.append(_circuit_stats(isa))
            isa_circuits.append(isa)
        phase_timing["transpile_wall_s"] += time.perf_counter() - transpile_start

        print(f"[QPU] {label}: submitting {len(isa_circuits)} parameter-free circuits ...")
        submit_start = time.perf_counter()
        job = sampler.run(isa_circuits, shots=shots)
        submit_wall = time.perf_counter() - submit_start
        phase_timing["submit_wall_s"] += submit_wall
        print(f"[QPU] {label}: job_id={job.job_id()}")

        wait_start = time.perf_counter()
        result = job.result()
        wait_wall = time.perf_counter() - wait_start
        phase_timing["job_wait_wall_s"] += wait_wall
        jobs.append(_runtime_job_record(job, submit_wall, wait_wall))

        if len(result) != len(isa_circuits):
            raise RuntimeError("Sampler returned an unexpected number of PUB results")
        fidelities.extend(_extract_all_zero_probability(pub, feature_map.num_qubits) for pub in result)
    return fidelities, stats, phase_timing, jobs


def evaluate_hardware_kernel_train(X_train, feature_map, sampler, pass_manager, shots, max_circuits_per_job):
    n = len(X_train)
    raw = np.eye(n, dtype=float)
    indices, pairs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            indices.append((i, j)); pairs.append((X_train[i], X_train[j]))
    values, stats, phase_timing, jobs = _run_fidelity_pairs_on_hardware(
        pairs,
        feature_map,
        sampler,
        pass_manager,
        shots,
        max_circuits_per_job,
        "K_train",
    )
    for (i, j), value in zip(indices, values):
        raw[i, j] = value; raw[j, i] = value
    return raw, _project_to_psd(raw), _summarize_stats(stats), phase_timing, jobs


def evaluate_hardware_kernel_cross(X_left, X_right, feature_map, sampler, pass_manager, shots, max_circuits_per_job):
    kernel = np.empty((len(X_left), len(X_right)), dtype=float)
    indices, pairs = [], []
    for i, x in enumerate(X_left):
        for j, y in enumerate(X_right):
            indices.append((i, j)); pairs.append((x, y))
    values, stats, phase_timing, jobs = _run_fidelity_pairs_on_hardware(
        pairs,
        feature_map,
        sampler,
        pass_manager,
        shots,
        max_circuits_per_job,
        "K_test",
    )
    for (i, j), value in zip(indices, values):
        kernel[i, j] = value
    return kernel, _summarize_stats(stats), phase_timing, jobs


def build_hardware_executor(
    backend,
    shots: int,
    optimization_level: int,
    run_tag: str,
    transpiler_seed: int,
):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend, options={"default_shots": shots, "environment": {"job_tags": [run_tag]}})
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=transpiler_seed,
    )
    return sampler, pass_manager


def estimate_kernel_entries(n_train: int, n_test: int) -> dict:
    train_pairs = n_train * (n_train - 1) // 2
    test_pairs = n_test * n_train
    return {
        "train_off_diagonal_pairs": int(train_pairs),
        "test_train_pairs": int(test_pairs),
        "approx_total_fidelity_entries": int(train_pairs + test_pairs),
    }


def make_run_dir(args, feature_set_name: str, n_qubits: int) -> Path:
    if args.run_name:
        name = args.run_name
    else:
        prefix = f"{feature_set_name}_{n_qubits}q_{args.entanglement}_{args.scaler}_seed{args.random_state}"
        index = 1
        while (args.output_dir / f"{prefix}_{index:02d}").exists():
            index += 1
        name = f"{prefix}_{index:02d}"
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, datetime): return obj.isoformat()
    return str(obj)


def main():
    args = parse_args()
    configure_runtime_network(args.force_ipv4)
    if (
        args.C <= 0
        or args.reps < 1
        or args.shots <= 0
        or args.max_circuits_per_job <= 0
        or args.trial < 1
    ):
        raise ValueError("C, reps, shots, max-circuits-per-job and trial must be positive")

    features = get_features(args)
    feature_set_name = "custom" if args.features is not None else args.feature_set
    df = load_base_data(args, features)
    train_df, test_df = split_data(args, df)
    selected_df = pd.concat([train_df, test_df], ignore_index=True)
    (
        X_train_scaled,
        X_test_scaled,
        preprocess_train_wall,
        preprocess_test_wall,
    ) = fit_transform_scaler(train_df[features], test_df[features], args.scaler)

    feature_map, encoding_name = build_feature_map(len(features), args.n_qubits, args.reps, args.entanglement)
    n_qubits = feature_map.num_qubits
    if feature_map.num_parameters != len(features):
        raise RuntimeError("Feature-map parameter count does not match selected data features")

    run_dir = make_run_dir(args, feature_set_name, n_qubits)
    (run_dir / "feature_map.txt").write_text(str(feature_map.draw(output="text", fold=-1)) + "\n", encoding="utf-8")
    split_table(train_df, test_df, args.label, args.group_column).to_csv(run_dir / "split.csv", index=False)

    workload = estimate_kernel_entries(len(train_df), len(test_df))
    print("\nDATA / FEATURE MAP")
    print(f"Feature set: {feature_set_name} ({len(features)} features)")
    print(f"Samples: {len(selected_df)} | train={len(train_df)} | test={len(test_df)}")
    print(
        f"Groups ({args.group_column}): train={train_df[args.group_column].nunique()} "
        f"| test={test_df[args.group_column].nunique()}"
    )
    print(f"Scaler: {args.scaler}")
    print(f"Encoding: {encoding_name} | qubits={n_qubits} | reps={args.reps} | entanglement={args.entanglement}")
    print(f"Feature-map depth pre-transpile: {feature_map.depth()} | size={feature_map.size()}")
    print(f"Approximate fidelity entries: {workload['approx_total_fidelity_entries']}")

    classical = None
    if not args.skip_classical_baseline:
        classical = fit_classical_baseline(args, X_train_scaled, train_df[args.label], X_test_scaled, test_df[args.label])
        print("\nCLASSICAL BASELINE ON SAME SPLIT/PREPROCESSING")
        print(f"Balanced accuracy: {classical['test_metrics']['balanced_accuracy']:.4f}")
        print(f"Macro F1:          {classical['test_metrics']['f1_macro']:.4f}")
        print(f"CPU wall time:     {classical['cpu_wall_time_s']:.6f} s")

    print("[IBM] Initializing Qiskit Runtime service ...", flush=True, )
    service = create_runtime_service(args.ibm_account, args.ibm_instance,)

    print("[IBM] Resolving backend ...", flush=True)
    backend = pick_backend(service, args.backend, n_qubits)
    print("\nIBM BACKEND")
    print(f"Backend: {backend.name} | qubits={backend.num_qubits} | pending_jobs={backend.status().pending_jobs} | operational={backend.status().operational}")

    if args.dry_run:
        dry_meta = {
            "model": "QSVM_IBM_DRY_RUN",
            "feature_set": feature_set_name,
            "features": features,
            "n_features": len(features),
            "n_qubits": n_qubits,
            "encoding": encoding_name,
            "feature_map_fingerprint": feature_map_fingerprint(feature_map),
            "reps": args.reps,
            "entanglement": args.entanglement,
            "scaler": args.scaler,
            "C": args.C,
            "class_weight": args.class_weight,
            "samples": len(selected_df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "test_size": args.test_size,
            "max_samples": args.max_samples,
            "backend": backend.name,
            "shots": args.shots,
            "trial": args.trial,
            "random_state": args.random_state,
            "split_file": None if args.split_file is None else str(args.split_file),
            "split_sha256": file_sha256(args.split_file),
            "split_strategy": "grouped",
            "grouping": group_summary(train_df, test_df, args.label, args.group_column),
            "optimization_level": args.optimization_level,
            "transpiler_seed": args.transpiler_seed,
            "max_circuits_per_job": args.max_circuits_per_job,
            "workload": workload,
            "classical_baseline": classical,
            "timing": {
                "preprocess_train_wall_s": preprocess_train_wall,
                "preprocess_test_wall_s": preprocess_test_wall,
            },
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(dry_meta, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
        print("[DRY RUN] No quantum job submitted.")
        print(f"RESULT_DIR={run_dir}")
        return

    run_tag = f"qsvm-{uuid.uuid4().hex[:12]}"
    sampler, pass_manager = build_hardware_executor(
        backend,
        args.shots,
        args.optimization_level,
        run_tag,
        args.transpiler_seed,
    )
    print(f"Runtime tag: {run_tag} | shots={args.shots}")
    print(f"Transpiler seed: {args.transpiler_seed} | trial={args.trial}")
    print("[workaround] Submitting only fully bound parameter-free compute-uncompute circuits.")

    train_start = time.perf_counter()
    (
        K_train_raw,
        K_train_psd,
        train_circuit_stats,
        train_phase_timing,
        train_jobs,
    ) = evaluate_hardware_kernel_train(
        X_train_scaled, feature_map, sampler, pass_manager, args.shots, args.max_circuits_per_job
    )
    train_wall = time.perf_counter() - train_start

    test_start = time.perf_counter()
    K_test, test_circuit_stats, test_phase_timing, test_jobs = evaluate_hardware_kernel_cross(
        X_test_scaled, X_train_scaled, feature_map, sampler, pass_manager, args.shots, args.max_circuits_per_job
    )
    test_wall = time.perf_counter() - test_start

    qsvc = SVC(kernel="precomputed", C=args.C, class_weight=class_weight_value(args.class_weight))

    svc_fit_start = time.perf_counter()
    qsvc.fit(K_train_psd, train_df[args.label])
    svc_fit_wall = time.perf_counter() - svc_fit_start

    predict_train_start = time.perf_counter()
    pred_train = qsvc.predict(K_train_psd)
    predict_train_wall = time.perf_counter() - predict_train_start

    predict_test_start = time.perf_counter()
    pred_test = qsvc.predict(K_test)
    predict_test_wall = time.perf_counter() - predict_test_start
    svc_total_wall = svc_fit_wall + predict_train_wall + predict_test_wall

    train_metrics = metric_bundle(train_df[args.label], pred_train)
    test_metrics = metric_bundle(test_df[args.label], pred_test)
    diagnosis = diagnose_fit(train_metrics["balanced_accuracy"], test_metrics["balanced_accuracy"])
    report = classification_report(test_df[args.label], pred_test, zero_division=0)

    runtime_jobs = train_jobs + test_jobs
    quantum_seconds = _sum_available(runtime_jobs, "quantum_seconds")
    queue_wall = _sum_available(runtime_jobs, "queue_wall_s")
    service_run_wall = _sum_available(runtime_jobs, "service_run_wall_s")

    circuit_build_wall = (
        train_phase_timing["circuit_build_wall_s"]
        + test_phase_timing["circuit_build_wall_s"]
    )
    transpile_wall = (
        train_phase_timing["transpile_wall_s"]
        + test_phase_timing["transpile_wall_s"]
    )
    submit_wall = (
        train_phase_timing["submit_wall_s"]
        + test_phase_timing["submit_wall_s"]
    )
    job_wait_wall = (
        train_phase_timing["job_wait_wall_s"]
        + test_phase_timing["job_wait_wall_s"]
    )
    labels = sorted(selected_df[args.label].unique().tolist())

    pd.DataFrame(K_train_raw).to_csv(run_dir / "K_train_raw.csv", index=False)
    pd.DataFrame(K_train_psd).to_csv(run_dir / "K_train_psd.csv", index=False)
    pd.DataFrame(K_test).to_csv(run_dir / "K_test.csv", index=False)
    prediction_data = {
        "row_id": test_df["_row_id"].to_numpy(),
        "true_label": test_df[args.label].to_numpy(),
        "qsvm_hardware_pred": pred_test,
    }
    if classical is not None:
        prediction_data["classical_pred"] = classical["pred_test"]
    pd.DataFrame(prediction_data).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(test_df[args.label], pred_test, labels=labels), index=labels, columns=labels).to_csv(run_dir / "qsvm_confusion_matrix.csv")
    (run_dir / "qsvm_classification_report.txt").write_text(report, encoding="utf-8")
    (run_dir / "transpiled_circuit_stats.json").write_text(json.dumps({"K_train": train_circuit_stats, "K_test": test_circuit_stats}, indent=2), encoding="utf-8")

    metadata = {
        "model": "QSVM_IBM",
        "implementation": "manual bound compute-uncompute + SamplerV2 + SVC(precomputed)",
        "input_csv": str(args.csv),
        "feature_set": feature_set_name,
        "features": features,
        "n_features": len(features),
        "n_qubits": n_qubits,
        "n_feature_map_parameters": feature_map.num_parameters,
        "encoding": encoding_name,
        "feature_map_fingerprint": feature_map_fingerprint(feature_map),
        "reps": args.reps,
        "entanglement": args.entanglement,
        "scaler": args.scaler,
        "C": args.C,
        "class_weight": args.class_weight,
        "samples": len(selected_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "trial": args.trial,
        "max_samples": args.max_samples,
        "split_file": None if args.split_file is None else str(args.split_file),
        "split_sha256": file_sha256(args.split_file),
        "split_strategy": "grouped",
        "grouping": group_summary(train_df, test_df, args.label, args.group_column),
        "feature_map_depth_pretranspile": feature_map.depth(),
        "feature_map_size_pretranspile": feature_map.size(),
        "transpiled_circuit_stats": {"K_train": train_circuit_stats, "K_test": test_circuit_stats},
        "backend": backend.name,
        "backend_num_qubits": backend.num_qubits,
        "shots": args.shots,
        "optimization_level": args.optimization_level,
        "transpiler_seed": args.transpiler_seed,
        "max_circuits_per_job": args.max_circuits_per_job,
        "runtime_job_tag": run_tag,
        "kernel_workload_estimate": workload,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "fit_diagnosis": diagnosis,
        "timing": {

            "preprocess_train_wall_s": preprocess_train_wall,
            "preprocess_test_wall_s": preprocess_test_wall,
            "kernel_train_wall_s": train_wall,
            "kernel_test_wall_s": test_wall,
            "kernel_total_wall_s": train_wall + test_wall,
            "model_fit_wall_s": svc_fit_wall,
            "predict_train_wall_s": predict_train_wall,
            "predict_test_wall_s": predict_test_wall,
            "quantum_execution_s": quantum_seconds,
            "circuit_build_wall_s": circuit_build_wall,
            "transpile_wall_s": transpile_wall,
            "submit_wall_s": submit_wall,

            "job_wait_wall_s": job_wait_wall,
            "queue_wall_s": queue_wall,
            "service_run_wall_s": service_run_wall,
            "end_to_end_test_wall_s": (
                preprocess_train_wall
                + preprocess_test_wall
                + train_wall
                + test_wall
                + svc_fit_wall
                + predict_test_wall
            ),

            "precomputed_svc_cpu_wall_s": svc_total_wall,
            "reported_quantum_seconds_sum": quantum_seconds,
        },
        "classical_baseline": classical,
        "runtime_jobs": runtime_jobs,
        "environment": {
            "python": platform.python_version(),
            "qiskit": package_version("qiskit"),
            "qiskit_ibm_runtime": package_version("qiskit-ibm-runtime"),
            "scikit_learn": package_version("scikit-learn"),
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

    print("\nQSVM IBM RESULT")
    print(f"Test accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test macro F1:           {test_metrics['f1_macro']:.4f}")
    print(f"Train balanced accuracy: {train_metrics['balanced_accuracy']:.4f}")
    print(f"Train-test BA gap:       {diagnosis['train_test_gap']:.4f}")
    print(f"Diagnosis:               {diagnosis['diagnosis']}")
    print(f"Kernel wall time:        {train_wall + test_wall:.3f} s")
    print(f"SVC fit time:            {svc_fit_wall:.6f} s")
    print(f"Predict test time:       {predict_test_wall:.6f} s")
    print(f"Transpilation time:      {transpile_wall:.3f} s")
    print(f"Client job wait time:    {job_wait_wall:.3f} s")
    print(
        "IBM quantum seconds:     "
        + ("unavailable" if quantum_seconds is None else f"{quantum_seconds:.6f}")
    )
    print(f"Transpiled train depth mean: {train_circuit_stats.get('depth_mean')}")
    print(f"Transpiled train 2Q gates mean: {train_circuit_stats.get('two_qubit_gates_mean')}")
    print(report)
    print(f"RESULT_DIR={run_dir}")


if __name__ == "__main__":
    main()
