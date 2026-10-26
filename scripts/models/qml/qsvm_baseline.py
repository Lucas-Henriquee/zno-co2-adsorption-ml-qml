from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
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

# qiskit and qiskit-machine-learning are imported lazily inside the functions that
# build or evaluate circuits, so data loading and splitting work without them.
try:
    from scripts.common.project_config import RESULTS_DIR
    from scripts.models.grouped_split import (
        DEFAULT_GROUP_COLUMN,
        group_summary,
        grouped_cv_splitter,
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
        grouped_cv_splitter,
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
SIMULATORS = ["reference", "aer_mps"]


def parse_args():
    parser = argparse.ArgumentParser(description="General ideal/simulated quantum-kernel SVM benchmark.")
    parser.add_argument("--csv", type=Path, default=RESULTS_DIR / "datasets" / "md_window_unique_dataset.csv")
    parser.add_argument("--label", default="label_site_id")
    parser.add_argument(
        "--group-column",
        default=DEFAULT_GROUP_COLUMN,
        help="Column with the independent-trajectory id; train/test and CV never split a group.",
    )
    parser.add_argument("--feature-set", default="md3", choices=sorted(FEATURE_SETS))
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--scaler", default="power_minmax", choices=SCALERS)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--entanglement", default="linear", choices=ENTANGLEMENTS)
    parser.add_argument("--n-qubits", type=int, default=0, help="0 = one qubit per feature; >features enables repeated multi-qubit encoding.")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--max-samples", type=int, default=0, help="0 uses every available row.")
    parser.add_argument("--split-file", type=Path, default=None, help="Optional CSV with row_id and split=train/test; must not share groups.")
    parser.add_argument("--cv", type=int, default=0, help="Optional stratified group CV on the training set; 0 disables it.")
    parser.add_argument(
        "--simulator",
        default="reference",
        choices=SIMULATORS,
        help=(
            "reference = Qiskit FidelityStatevectorKernel; "
            "aer_mps = Qiskit Aer matrix-product-state simulation."
        ),
    )
    parser.add_argument(
        "--mps-max-bond-dimension",
        type=int,
        default=0,
        help=(
            "MPS only. 0 leaves the bond dimension unlimited. "
        ),
    )
    parser.add_argument(
        "--mps-truncation-threshold",
        type=float,
        default=0.0,
        help=(
            "MPS only. 0 disables coefficient truncation as far as Aer permits. "
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "qsvm")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


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


def diagnose_fit(train_score: float, test_score: float, cv_mean: float | None = None,
                 gap_threshold: float = 0.10, low_threshold: float = 0.60) -> dict:
    gap = train_score - test_score
    if train_score < low_threshold and test_score < low_threshold:
        status = "possible_underfitting"
    elif gap > gap_threshold:
        status = "possible_overfitting"
    else:
        status = "no_obvious_train_test_gap"
    return {
        "train_score": float(train_score),
        "test_score": float(test_score),
        "cv_score": None if cv_mean is None else float(cv_mean),
        "train_test_gap": float(gap),
        "diagnosis": status,
    }


def get_features(args) -> list[str]:
    return list(args.features) if args.features is not None else FEATURE_SETS[args.feature_set].copy()


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
    X_train_np = np.asarray(X_train, dtype=float)
    X_test_np = np.asarray(X_test, dtype=float)
    fitted = []
    if scaler_name == "none":
        train_start = time.perf_counter()
        Xtr = X_train_np.copy()
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = X_test_np.copy()
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, fitted, train_wall, test_wall
    if scaler_name == "standard":
        scaler = StandardScaler()
        train_start = time.perf_counter()
        Xtr = scaler.fit_transform(X_train_np)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = scaler.transform(X_test_np)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [scaler], train_wall, test_wall
    if scaler_name == "minmax":
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        train_start = time.perf_counter()
        Xtr = scaler.fit_transform(X_train_np)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = scaler.transform(X_test_np)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [scaler], train_wall, test_wall
    if scaler_name == "minmax_pi":
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        train_start = time.perf_counter()
        Xtr = scaler.fit_transform(X_train_np)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = scaler.transform(X_test_np)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [scaler], train_wall, test_wall
    if scaler_name == "power":
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
        train_start = time.perf_counter()
        Xtr = scaler.fit_transform(X_train_np)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        Xte = scaler.transform(X_test_np)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [scaler], train_wall, test_wall
    if scaler_name == "power_minmax":
        power = PowerTransformer(method="yeo-johnson", standardize=True)
        train_start = time.perf_counter()
        train_p = power.fit_transform(X_train_np)
        mm = MinMaxScaler(feature_range=(0.0, np.pi))
        Xtr = mm.fit_transform(train_p)
        train_wall = time.perf_counter() - train_start
        test_start = time.perf_counter()
        test_p = power.transform(X_test_np)
        Xte = mm.transform(test_p)
        test_wall = time.perf_counter() - test_start
        return Xtr, Xte, [power, mm], train_wall, test_wall
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


def metric_bundle(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _project_to_psd(kernel_matrix: np.ndarray) -> np.ndarray:

    sym = 0.5 * (kernel_matrix + kernel_matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, 0.0)
    return (eigvecs @ np.diag(eigvals) @ eigvecs.T).real


def _bind_feature_map(feature_map: QuantumCircuit, values: np.ndarray) -> QuantumCircuit:

    params = list(feature_map.parameters)
    values = np.asarray(values, dtype=float)

    if len(params) != len(values):
        raise ValueError(
            f"Feature map expects {len(params)} values, received {len(values)}."
        )

    bound = feature_map.assign_parameters(
        dict(zip(params, values)),
        inplace=False,
    )

    if bound.num_parameters != 0:
        raise RuntimeError(
            "Feature-map circuit still has unbound parameters after binding."
        )
    return bound


def _make_mps_fidelity_circuit(
    feature_map: QuantumCircuit,
    x: np.ndarray,
    y: np.ndarray,
) -> QuantumCircuit:
    
    ux = _bind_feature_map(feature_map, x)
    uy = _bind_feature_map(feature_map, y)

    circuit = ux.compose(uy.inverse())
    circuit.global_phase = 0

    try:
        import qiskit_aer 
    except ImportError as exc:
        raise RuntimeError(
            "Simulator 'aer_mps' requires qiskit-aer. "
            "Install it in this environment with: pip install qiskit-aer"
        ) from exc

    circuit.save_amplitudes_squared(
        [0],
        label="fidelity_zero",
    )
    return circuit


def _build_mps_simulator(
    max_bond_dimension: int,
    truncation_threshold: float,
):
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise RuntimeError(
            "Simulator 'aer_mps' requires qiskit-aer. "
            "Install it in this environment with: pip install qiskit-aer"
        ) from exc

    if max_bond_dimension < 0:
        raise ValueError("--mps-max-bond-dimension must be >= 0.")
    if truncation_threshold < 0:
        raise ValueError("--mps-truncation-threshold must be >= 0.")

    options = {
        "method": "matrix_product_state",
        "max_parallel_experiments": 1,
        "matrix_product_state_truncation_threshold": float(truncation_threshold),
    }
    if max_bond_dimension > 0:
        options["matrix_product_state_max_bond_dimension"] = int(max_bond_dimension)

    return AerSimulator(**options)


def _run_mps_pair_batch(
    simulator,
    feature_map: QuantumCircuit,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    if not pairs:
        return []

    circuits = [
        _make_mps_fidelity_circuit(feature_map, x, y)
        for x, y in pairs
    ]

    result = simulator.run(circuits).result()

    values: list[float] = []
    for i in range(len(circuits)):
        saved = result.data(i)["fidelity_zero"]
        arr = np.asarray(saved, dtype=float).reshape(-1)
        if arr.size != 1:
            raise RuntimeError(
                f"Unexpected MPS fidelity result shape for circuit {i}: {arr.shape}"
            )
        values.append(float(arr[0]))
    return values


def _evaluate_mps_kernel_train(
    X_train: np.ndarray,
    feature_map: QuantumCircuit,
    simulator,
) -> np.ndarray:
    n = len(X_train)
    kernel = np.eye(n, dtype=float)

    indices: list[tuple[int, int]] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n):
        for j in range(i + 1, n):
            indices.append((i, j))
            pairs.append((X_train[i], X_train[j]))

    values = _run_mps_pair_batch(simulator, feature_map, pairs)
    for (i, j), value in zip(indices, values):
        kernel[i, j] = value
        kernel[j, i] = value

    return _project_to_psd(kernel)


def _evaluate_mps_kernel_cross(
    X_left: np.ndarray,
    X_right: np.ndarray,
    feature_map: QuantumCircuit,
    simulator,
) -> np.ndarray:
    kernel = np.empty((len(X_left), len(X_right)), dtype=float)

    indices: list[tuple[int, int]] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for i, x in enumerate(X_left):
        for j, y in enumerate(X_right):
            indices.append((i, j))
            pairs.append((x, y))

    values = _run_mps_pair_batch(simulator, feature_map, pairs)
    for (i, j), value in zip(indices, values):
        kernel[i, j] = value

    return kernel


def evaluate_one_split(
    X_train,
    y_train,
    X_test,
    y_test,
    scaler_name: str,
    feature_map: QuantumCircuit,
    C: float,
    class_weight: str,
    simulator_name: str,
    mps_max_bond_dimension: int = 0,
    mps_truncation_threshold: float = 0.0,
):
    Xtr, Xte, _, preprocess_train_time, preprocess_test_time = fit_transform_scaler(
        X_train,
        X_test,
        scaler_name,
    )

    if simulator_name == "reference":
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel

        kernel = FidelityStatevectorKernel(
            feature_map=feature_map,
            shots=None,
            enforce_psd=True,
        )

        ktrain_start = time.perf_counter()
        K_train = kernel.evaluate(x_vec=Xtr)
        ktrain_time = time.perf_counter() - ktrain_start

        ktest_start = time.perf_counter()
        K_test = kernel.evaluate(x_vec=Xte, y_vec=Xtr)
        ktest_time = time.perf_counter() - ktest_start

        kernel_implementation = "FidelityStatevectorKernel"

    elif simulator_name == "aer_mps":
        simulator = _build_mps_simulator(
            max_bond_dimension=mps_max_bond_dimension,
            truncation_threshold=mps_truncation_threshold,
        )

        print(
            "[MPS] AerSimulator | "
            f"max_bond_dimension={'unlimited' if mps_max_bond_dimension == 0 else mps_max_bond_dimension} | "
            f"truncation_threshold={mps_truncation_threshold}"
        )

        ktrain_start = time.perf_counter()
        K_train = _evaluate_mps_kernel_train(Xtr, feature_map, simulator)
        ktrain_time = time.perf_counter() - ktrain_start

        ktest_start = time.perf_counter()
        K_test = _evaluate_mps_kernel_cross(Xte, Xtr, feature_map, simulator)
        ktest_time = time.perf_counter() - ktest_start

        kernel_implementation = "manual_compute_uncompute_aer_mps"

    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")

    model = SVC(
        kernel="precomputed",
        C=C,
        class_weight=class_weight_value(class_weight),
    )

    fit_start = time.perf_counter()
    model.fit(K_train, y_train)
    model_fit_time = time.perf_counter() - fit_start

    predict_train_start = time.perf_counter()
    pred_train = model.predict(K_train)
    predict_train_time = time.perf_counter() - predict_train_start

    predict_test_start = time.perf_counter()
    pred_test = model.predict(K_test)
    predict_test_time = time.perf_counter() - predict_test_start

    return {
        "K_train": K_train,
        "K_test": K_test,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "kernel_train_time_s": ktrain_time,
        "kernel_test_time_s": ktest_time,
        "preprocess_train_time_s": preprocess_train_time,
        "preprocess_test_time_s": preprocess_test_time,
        "model_fit_time_s": model_fit_time,
        "predict_train_time_s": predict_train_time,
        "predict_test_time_s": predict_test_time,
        "svc_total_time_s": model_fit_time + predict_train_time + predict_test_time,
        "kernel_implementation": kernel_implementation,
    }


def run_cv(args, selected_df: pd.DataFrame, features: list[str], n_qubits: int) -> tuple[float | None, float | None, float | None, int]:
    splitter, folds = grouped_cv_splitter(
        selected_df, args.label, args.group_column, int(args.cv), args.random_state
    )
    if splitter is None:
        return None, None, None, 0

    y = selected_df[args.label]
    groups = selected_df[args.group_column].to_numpy()
    scores = []
    start = time.perf_counter()
    # Whole trajectories per fold; scalers are refitted inside evaluate_one_split.
    for train_idx, val_idx in splitter.split(selected_df[features], y, groups):
        tr, va = selected_df.iloc[train_idx], selected_df.iloc[val_idx]
        fmap, _ = build_feature_map(len(features), n_qubits, args.reps, args.entanglement)
        result = evaluate_one_split(
            tr[features],
            tr[args.label],
            va[features],
            va[args.label],
            args.scaler,
            fmap,
            args.C,
            args.class_weight,
            args.simulator,
            args.mps_max_bond_dimension,
            args.mps_truncation_threshold,
        )
        scores.append(balanced_accuracy_score(va[args.label], result["pred_test"]))
    elapsed = time.perf_counter() - start
    return float(np.mean(scores)), float(np.std(scores)), elapsed, folds


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


def main():
    args = parse_args()
    if args.C <= 0 or args.reps < 1:
        raise ValueError("--C and --reps must be positive")

    features = get_features(args)
    feature_set_name = "custom" if args.features is not None else args.feature_set
    df = load_base_data(args, features)
    train_df, test_df = split_data(args, df)
    selected_df = pd.concat([train_df, test_df], ignore_index=True)

    feature_map, encoding_name = build_feature_map(len(features), args.n_qubits, args.reps, args.entanglement)
    n_qubits = feature_map.num_qubits
    result = evaluate_one_split(
        train_df[features],
        train_df[args.label],
        test_df[features],
        test_df[args.label],
        args.scaler,
        feature_map,
        args.C,
        args.class_weight,
        args.simulator,
        args.mps_max_bond_dimension,
        args.mps_truncation_threshold,
    )

    train_metrics = metric_bundle(train_df[args.label], result["pred_train"])
    test_metrics = metric_bundle(test_df[args.label], result["pred_test"])
    cv_mean, cv_std, cv_time, cv_folds = run_cv(args, train_df, features, n_qubits)
    diagnosis = diagnose_fit(train_metrics["balanced_accuracy"], test_metrics["balanced_accuracy"], cv_mean)

    run_dir = make_run_dir(args, feature_set_name, n_qubits)
    labels = sorted(selected_df[args.label].unique().tolist())

    pd.DataFrame(result["K_train"]).to_csv(run_dir / "K_train.csv", index=False)
    pd.DataFrame(result["K_test"]).to_csv(run_dir / "K_test.csv", index=False)
    pd.DataFrame({
        "row_id": test_df["_row_id"].to_numpy(),
        "true_label": test_df[args.label].to_numpy(),
        "pred_label": result["pred_test"],
    }).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(test_df[args.label], result["pred_test"], labels=labels), index=labels, columns=labels).to_csv(run_dir / "confusion_matrix.csv")
    report = classification_report(test_df[args.label], result["pred_test"], zero_division=0)
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (run_dir / "used_features.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    (run_dir / "feature_map.txt").write_text(str(feature_map.draw(output="text", fold=-1)) + "\n", encoding="utf-8")
    split_table(train_df, test_df, args.label, args.group_column).to_csv(run_dir / "split.csv", index=False)

    metrics = {
        "model": "QSVM_IDEAL",
        "implementation": result["kernel_implementation"] + " + SVC(precomputed)",
        "input_csv": str(args.csv),
        "feature_set": feature_set_name,
        "features": features,
        "n_features": len(features),
        "n_qubits": n_qubits,
        "encoding": encoding_name,
        "feature_map_fingerprint": feature_map_fingerprint(feature_map),
        "feature_map": "zz_feature_map" if n_qubits == len(features) else "custom_multiqubit_zz_reupload",
        "reps": args.reps,
        "entanglement": args.entanglement,
        "feature_map_depth": feature_map.depth(),
        "feature_map_size": feature_map.size(),
        "samples": len(selected_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "scaler": args.scaler,
        "simulator": args.simulator,
        "mps_max_bond_dimension": int(args.mps_max_bond_dimension),
        "mps_truncation_threshold": float(args.mps_truncation_threshold),
        "mps_approximate": bool(
            args.simulator == "aer_mps"
            and (args.mps_max_bond_dimension > 0 or args.mps_truncation_threshold > 0)
        ),
        "C": args.C,
        "class_weight": args.class_weight,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_samples": args.max_samples,
        "split_file": None if args.split_file is None else str(args.split_file),
        "split_sha256": file_sha256(args.split_file),
        "split_strategy": "grouped",
        "grouping": group_summary(train_df, test_df, args.label, args.group_column),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_folds": cv_folds,
        "cv_balanced_accuracy_mean": cv_mean,
        "cv_balanced_accuracy_std": cv_std,
        "fit_diagnosis": diagnosis,
        "timing": {
            "preprocess_train_wall_s": result["preprocess_train_time_s"],
            "preprocess_test_wall_s": result["preprocess_test_time_s"],
            "kernel_train_wall_s": result["kernel_train_time_s"],
            "kernel_test_wall_s": result["kernel_test_time_s"],
            "kernel_total_wall_s": result["kernel_train_time_s"] + result["kernel_test_time_s"],
            "model_fit_wall_s": result["model_fit_time_s"],
            "predict_train_wall_s": result["predict_train_time_s"],
            "predict_test_wall_s": result["predict_test_time_s"],
            "quantum_execution_s": None,
            "transpile_wall_s": None,
            "job_wait_wall_s": None,
            "end_to_end_test_wall_s": (
                result["preprocess_train_time_s"]
                + result["preprocess_test_time_s"]
                + result["kernel_train_time_s"]
                + result["kernel_test_time_s"]
                + result["model_fit_time_s"]
                + result["predict_test_time_s"]
            ),
            "cv_wall_s": cv_time,
            "kernel_train_time_s": result["kernel_train_time_s"],
            "kernel_test_time_s": result["kernel_test_time_s"],
            "kernel_total_time_s": result["kernel_train_time_s"] + result["kernel_test_time_s"],
            "precomputed_svc_time_s": result["svc_total_time_s"],
            "total_test_pipeline_s": (
                result["kernel_train_time_s"]
                + result["kernel_test_time_s"]
                + result["svc_total_time_s"]
            ),
            "cv_time_s": cv_time,
        },
        "environment": {
            "python": platform.python_version(),
            "qiskit": package_version("qiskit"),
            "qiskit_machine_learning": package_version("qiskit-machine-learning"),
            "qiskit_aer": package_version("qiskit-aer"),
            "scikit_learn": package_version("scikit-learn"),
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== QSVM =====")
    print(f"Feature set: {feature_set_name} ({len(features)} features)")
    print(f"Qubits: {n_qubits} | encoding={encoding_name} | reps={args.reps} | entanglement={args.entanglement}")
    print(f"Scaler: {args.scaler} | C={args.C} | class_weight={args.class_weight}")
    print(f"Simulator: {args.simulator}")
    print(f"Test accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test macro F1:          {test_metrics['f1_macro']:.4f}")
    print(f"Train balanced accuracy:{train_metrics['balanced_accuracy']:.4f}")
    print(f"Train-test BA gap:      {diagnosis['train_test_gap']:.4f}")
    print(f"Diagnosis:              {diagnosis['diagnosis']}")
    if cv_mean is not None:
        print(f"CV balanced accuracy:   {cv_mean:.4f} +/- {cv_std:.4f} ({cv_folds} folds)")
    print(f"K_train time:           {result['kernel_train_time_s']:.6f} s")
    print(f"K_test time:            {result['kernel_test_time_s']:.6f} s")
    print(f"SVC fit time:           {result['model_fit_time_s']:.6f} s")
    print(f"Predict test time:      {result['predict_test_time_s']:.6f} s")
    print(
        "Preprocess train/test:  "
        f"{result['preprocess_train_time_s']:.6f} / "
        f"{result['preprocess_test_time_s']:.6f} s"
    )
    print(report)
    print(f"RESULT_DIR={run_dir}")


if __name__ == "__main__":
    main()
