"""Site memory on IBM Quantum hardware: SVM, ideal QSVM and hardware QSVM on the same windows.

For a few time windows chosen in advance (``--window-starts-fs``, e.g. an early window
where the starting site is identifiable and a late one where it is not), this script
evaluates three models on exactly the same rows, splits, scaling and C:

* ``svm_rbf``   classical SVM, RBF kernel, gamma = scale;
* ``qsvm_ideal`` SVM with the exact fidelity kernel of Qiskit's zz_feature_map
  (statevector, or the same kernel with numpy: ``--ideal-kernel zz_exact``);
* ``qsvm_ibm``  SVM with that kernel estimated on an IBM backend: compute-uncompute
  circuits U(x) U(y)^dagger, kernel entry = probability of measuring all zeros.

Data, rows (one per trajectory per window), leave-one-seed-out validation and the
within-seed permutation test are those of ``scripts.analysis.site_memory``. The kernels
do not depend on the labels, so the permutation test reuses them (no extra circuits).

Scaling (``--scaling``):

* ``fixed`` (default): every feature (a distance in Angstrom) is mapped from the
  physical range ``--feature-range-ang`` (default 2.0-6.0 A) to [0, pi/2], with values
  outside it clipped and counted. Nothing is fitted to the data, so one kernel over all
  trajectories of a window serves every split: N(N-1)/2 circuits per window (36 for 3
  seeds, 153 for 6).
* ``train``: MinMax to [0, pi/2] fitted on the training seeds of each split, as in
  site_memory.py; one kernel per split: per split n_tr(n_tr-1)/2 + n_te*n_tr circuits.

Upper limit pi/2: P(2x) of the ZZ feature map has period pi in x.

Hardware kernels: the training diagonal is 1 (ideal value of K(x, x)); the training
kernel is projected onto the positive semidefinite cone (eigenvalues clipped at 0); test
entries are used as measured. Raw hardware kernels, job ids, circuit statistics and the
reported quantum time are saved, so the analysis can be redone without the QPU
(``--reuse-hardware DIR``).

Safety: without ``--submit`` (and without ``--reuse-hardware``) the script only writes
the plan (windows, rows, circuits, shots) and never contacts IBM. ``--submit`` needs a
saved Qiskit Runtime account (QiskitRuntimeService.save_account) and runs on the
backend given by ``--backend`` or on the least busy one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import site_memory as sm

MODELS = ("svm_rbf", "qsvm_ideal", "qsvm_ibm")
DEFAULT_RANGE_ANG = (2.0, 6.0)
HARDWARE_DIR = "hardware_kernels"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Site memory with SVM, ideal QSVM and IBM hardware QSVM.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--sites", nargs="+", default=list(sm.DEFAULT_SITES))
    parser.add_argument("--features", default="geometry3", choices=sorted(sm.FEATURE_SETS))
    parser.add_argument("--window-fs", type=float, default=40.0)
    parser.add_argument("--window-starts-fs", nargs="+", type=float, required=True,
                        help="Start times of the windows to run, chosen before looking at hardware results.")
    parser.add_argument("--scaling", default="fixed", choices=["fixed", "train"])
    parser.add_argument("--feature-range-ang", nargs=2, type=float, default=list(DEFAULT_RANGE_ANG),
                        metavar=("LOW", "HIGH"), help="Physical range mapped to [0, pi/2] by --scaling fixed.")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--qsvm-reps", type=int, default=1)
    parser.add_argument("--qsvm-entanglement", default="linear", choices=["linear", "circular", "full"])
    parser.add_argument("--ideal-kernel", default="statevector", choices=["statevector", "zz_exact"])
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    hardware = parser.add_argument_group("IBM hardware")
    hardware.add_argument("--submit", action="store_true",
                          help="Contact IBM and run the circuits. Without it only the plan is written.")
    hardware.add_argument("--reuse-hardware", type=Path, default=None,
                          help="Folder with hardware kernels of an earlier --submit run (no QPU access).")
    hardware.add_argument("--backend", default=None)
    hardware.add_argument("--account", default=None, help="Name of a saved Qiskit Runtime account.")
    hardware.add_argument("--instance", default=None)
    hardware.add_argument("--shots", type=int, default=512)
    hardware.add_argument("--optimization-level", type=int, default=2)
    hardware.add_argument("--transpiler-seed", type=int, default=42)
    hardware.add_argument("--max-circuits-per-job", type=int, default=100)
    hardware.add_argument("--force-ipv4", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- data

def scale_fixed(X: np.ndarray, low: float, high: float) -> tuple[np.ndarray, int]:
    """Map [low, high] to [0, pi/2], clipping outside values; returns (Z, n_clipped)."""
    clipped = int(np.sum((X < low) | (X > high)))
    return sm.FEATURE_RANGE * (np.clip(X, low, high) - low) / (high - low), clipped


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.round(values, 12)).tobytes()).hexdigest()


def select_windows(dataset: pd.DataFrame, starts: list[float]) -> list[int]:
    available = dataset.drop_duplicates("window").set_index("window")["t_start_fs"]
    chosen = []
    for start in starts:
        match = available[np.isclose(available.to_numpy(dtype=float), start)]
        if match.empty:
            raise SystemExit(f"[site-memory-ibm] no window starts at {start:g} fs; available starts: "
                             f"{available.iloc[:5].tolist()} ... {available.iloc[-1]:g} fs (step {available.iloc[1] - available.iloc[0]:g}).")
        chosen.append(int(match.index[0]))
    return chosen


class WindowJob:
    """Rows of one window, their splits and the kernels to compute."""

    def __init__(self, rows: pd.DataFrame, features: list[str], args):
        self.rows = rows.sort_values(["seed", "label"]).reset_index(drop=True)
        self.window = int(self.rows["window"].iloc[0])
        self.t_start, self.t_end = float(self.rows["t_start_fs"].iloc[0]), float(self.rows["t_end_fs"].iloc[0])
        self.X = self.rows[features].to_numpy(dtype=float)
        self.y = self.rows["label"].to_numpy(dtype=int)
        self.seeds = self.rows["seed"].to_numpy(dtype=int)
        self.scaling = args.scaling
        self.clipped = 0
        self.splits = []  # (train mask, test mask, scaled train, scaled test)
        if args.scaling == "fixed":
            self.Z, self.clipped = scale_fixed(self.X, *args.feature_range_ang)
        for seed in sorted(set(self.seeds.tolist())):
            test = self.seeds == seed
            if args.scaling == "fixed":
                train_x, test_x = self.Z[~test], self.Z[test]
            else:
                train_x, test_x = sm.scale_features(self.X[~test], self.X[test])
            self.splits.append((~test, test, train_x, test_x))

    def kernel_requests(self) -> list[dict]:
        """Kernel matrices to estimate: one symmetric matrix (fixed) or one pair per split (train)."""
        if self.scaling == "fixed":
            n = len(self.Z)
            return [{"name": f"w{self.window}_all", "kind": "symmetric", "left": self.Z,
                     "circuits": n * (n - 1) // 2}]
        requests = []
        for index, (_, _, train_x, test_x) in enumerate(self.splits):
            n_tr, n_te = len(train_x), len(test_x)
            requests.append({"name": f"w{self.window}_split{index}_train", "kind": "symmetric", "left": train_x,
                             "circuits": n_tr * (n_tr - 1) // 2})
            requests.append({"name": f"w{self.window}_split{index}_test", "kind": "cross", "left": test_x,
                             "right": train_x, "circuits": n_te * n_tr})
        return requests

    def split_kernels(self, matrices: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(K_train, K_test) of every split from the estimated matrices."""
        out = []
        for index, (train, test, _, _) in enumerate(self.splits):
            if self.scaling == "fixed":
                full = matrices[f"w{self.window}_all"]
                out.append((full[np.ix_(train, train)], full[np.ix_(test, train)]))
            else:
                out.append((matrices[f"w{self.window}_split{index}_train"],
                            matrices[f"w{self.window}_split{index}_test"]))
        return out


# --------------------------------------------------------------------------- kernels

def ideal_matrices(job: WindowJob, kernel) -> dict[str, np.ndarray]:
    matrices = {}
    for request in job.kernel_requests():
        right = request.get("right")
        matrices[request["name"]] = kernel(request["left"]) if right is None else kernel(request["left"], right)
    return matrices


def psd_projection(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, 0.0)) @ vectors.T


def hardware_split_kernels(job: WindowJob, raw: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Training kernels with unit diagonal projected to PSD; test kernels as measured."""
    out = []
    for k_train, k_test in job.split_kernels(raw):
        k_train = np.array(k_train, dtype=float, copy=True)
        np.fill_diagonal(k_train, 1.0)
        out.append((psd_projection(k_train), np.array(k_test, dtype=float)))
    return out


def run_on_hardware(jobs: list[WindowJob], args, output: Path) -> tuple[dict, dict]:
    """Estimate every requested kernel on IBM; returns (raw matrices, hardware record)."""
    from scripts.models.qml.qsvm_baseline import build_feature_map
    from scripts.models.qml_ibm import qsvm_ibm_hardware as hw

    hw.configure_runtime_network(args.force_ipv4)
    service = hw.create_runtime_service(args.account, args.instance)
    n_qubits = jobs[0].X.shape[1]
    backend = hw.pick_backend(service, args.backend, n_qubits)
    tag = f"site-memory-{uuid.uuid4().hex[:8]}"
    sampler, pass_manager = hw.build_hardware_executor(backend, args.shots, args.optimization_level, tag,
                                                       args.transpiler_seed)
    feature_map, map_kind = build_feature_map(n_qubits, 0, args.qsvm_reps, args.qsvm_entanglement)
    raw, record = {}, {"backend": backend.name, "job_tag": tag, "feature_map": map_kind, "kernels": {}}
    folder = output / HARDWARE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        for request in job.kernel_requests():
            if request["kind"] == "symmetric":
                matrix, _, stats, timing, job_records = hw.evaluate_hardware_kernel_train(
                    request["left"], feature_map, sampler, pass_manager, args.shots, args.max_circuits_per_job)
            else:
                matrix, stats, timing, job_records = hw.evaluate_hardware_kernel_cross(
                    request["left"], request["right"], feature_map, sampler, pass_manager, args.shots,
                    args.max_circuits_per_job)
            raw[request["name"]] = np.asarray(matrix, dtype=float)
            np.save(folder / f"{request['name']}.npy", raw[request["name"]])
            record["kernels"][request["name"]] = {
                "circuits": request["circuits"], "left_sha256": array_sha256(request["left"]),
                "right_sha256": array_sha256(request["right"]) if "right" in request else None,
                "circuit_stats": stats, "timing": timing, "jobs": job_records,
                "quantum_seconds": sum(float(r.get("quantum_seconds") or 0.0) for r in job_records),
            }
            (folder / "hardware_record.json").write_text(json.dumps(record, indent=2, default=str) + "\n",
                                                         encoding="utf-8")
    return raw, record


def load_hardware(jobs: list[WindowJob], folder: Path) -> tuple[dict, dict]:
    record_path = folder / "hardware_record.json"
    if not record_path.is_file():
        raise SystemExit(f"[site-memory-ibm] {record_path} not found.")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    raw = {}
    for job in jobs:
        for request in job.kernel_requests():
            name = request["name"]
            saved = record["kernels"].get(name)
            if saved is None or not (folder / f"{name}.npy").is_file():
                raise SystemExit(f"[site-memory-ibm] no saved hardware kernel {name} in {folder}.")
            if saved["left_sha256"] != array_sha256(request["left"]) or (
                    "right" in request and saved["right_sha256"] != array_sha256(request["right"])):
                raise SystemExit(f"[site-memory-ibm] the saved kernel {name} was computed on other data "
                                 "(features, scaling or runs differ).")
            raw[name] = np.load(folder / f"{name}.npy")
    return raw, record


# --------------------------------------------------------------------------- evaluation

def predict(job: WindowJob, y: np.ndarray, model: str, kernels, args) -> np.ndarray:
    from sklearn.svm import SVC

    predicted = np.empty_like(y)
    for index, (train, test, train_x, test_x) in enumerate(job.splits):
        if model == "svm_rbf":
            estimator = SVC(kernel="rbf", C=args.c, gamma="scale", class_weight="balanced",
                            random_state=args.random_state)
            predicted[test] = estimator.fit(train_x, y[train]).predict(test_x)
        else:
            k_train, k_test = kernels[index]
            estimator = SVC(kernel="precomputed", C=args.c, class_weight="balanced")
            predicted[test] = estimator.fit(k_train, y[train]).predict(k_test)
    return predicted


def kernel_deviation(job: WindowJob, ideal: dict, raw: dict) -> dict:
    """Off-diagonal differences between the raw hardware and the ideal kernels."""
    differences = []
    for name, matrix in raw.items():
        if not name.startswith(f"w{job.window}_"):
            continue
        delta = np.asarray(matrix) - np.asarray(ideal[name])
        if delta.shape[0] == delta.shape[1] and name.endswith(("_all", "_train")):
            delta = delta[~np.eye(delta.shape[0], dtype=bool)]
        differences.append(delta.ravel())
    values = np.concatenate(differences) if differences else np.array([])
    return {"kernel_mae": float(np.mean(np.abs(values))), "kernel_rmse": float(np.sqrt(np.mean(values ** 2))),
            "kernel_bias": float(np.mean(values)), "kernel_max_abs": float(np.max(np.abs(values))),
            "kernel_entries": int(values.size)}


def evaluate(job: WindowJob, kernels_by_model: dict, args, rng) -> tuple[list[dict], list[dict]]:
    labelings, exact = sm.within_seed_relabelings(job.y, job.seeds, args.n_permutations, rng)
    rows, predictions = [], []
    for model in MODELS:
        kernels = kernels_by_model.get(model)
        if model != "svm_rbf" and kernels is None:
            continue
        predicted = predict(job, job.y, model, kernels, args)
        observed = sm.balanced_accuracy(job.y, predicted)
        null = np.array([sm.balanced_accuracy(labels, predict(job, labels, model, kernels, args))
                         for labels in labelings])
        rows.append({"window": job.window, "t_start_fs": job.t_start, "t_end_fs": job.t_end, "model": model,
                     "balanced_accuracy": observed, "n_correct": int(np.sum(predicted == job.y)),
                     "n_trajectories": int(len(job.y)),
                     "p_value": float((1 + np.sum(null >= observed - 1e-12)) / (1 + len(null))),
                     "null_mean": float(null.mean()), "n_null": int(len(null)), "exact_null": bool(exact)})
        predictions += [{"window": job.window, "model": model, "seed": int(s), "site": args.sites[int(t)],
                         "predicted_site": args.sites[int(p)]}
                        for s, t, p in zip(job.seeds, job.y, predicted)]
    return rows, predictions


def plan(jobs: list[WindowJob], args) -> dict:
    windows = []
    for job in jobs:
        requests = job.kernel_requests()
        windows.append({"window": job.window, "t_start_fs": job.t_start, "t_end_fs": job.t_end,
                        "trajectories": int(len(job.y)), "seeds": sorted(set(job.seeds.tolist())),
                        "clipped_feature_values": job.clipped,
                        "kernels": [{"name": r["name"], "circuits": r["circuits"]} for r in requests],
                        "circuits": int(sum(r["circuits"] for r in requests))})
    circuits = sum(w["circuits"] for w in windows)
    return {"windows": windows, "total_circuits": int(circuits), "shots": args.shots,
            "total_shots": int(circuits * args.shots), "scaling": args.scaling,
            "feature_range_ang": args.feature_range_ang if args.scaling == "fixed" else None,
            "qubits": int(jobs[0].X.shape[1]), "feature_map": "zz_feature_map",
            "reps": args.qsvm_reps, "entanglement": args.qsvm_entanglement}


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.submit and args.reuse_hardware is not None:
        raise SystemExit("[site-memory-ibm] use either --submit or --reuse-hardware, not both.")
    low, high = args.feature_range_ang
    if not high > low:
        raise SystemExit("[site-memory-ibm] --feature-range-ang needs LOW < HIGH.")
    runs = [run.expanduser().resolve() for run in args.runs]
    settings = sm.common_settings(runs)
    features = sm.FEATURE_SETS[args.features]
    frames, inputs = sm.load_frames(runs, args.material, args.sites, settings["timestep_fs"])
    seeds, dropped = sm.complete_seeds(frames, args.sites)
    if len(seeds) < 2:
        raise SystemExit(f"[site-memory-ibm] leave one seed out needs >= 2 complete seeds, found {seeds}.")
    frames = frames[frames["seed"].isin(seeds)]
    dataset = sm.window_dataset(frames, features, sm.window_edges(frames, args.window_fs, None), args.sites)
    jobs = [WindowJob(dataset[dataset["window"] == index], features, args)
            for index in select_windows(dataset, args.window_starts_fs)]

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    the_plan = plan(jobs, args)
    (output / "site_memory_ibm_plan.json").write_text(json.dumps(the_plan, indent=2) + "\n", encoding="utf-8")
    for window in the_plan["windows"]:
        print(f"[site-memory-ibm] window {window['t_start_fs']:g}-{window['t_end_fs']:g} fs: "
              f"{window['trajectories']} trajectories, {window['circuits']} circuits"
              + (f", {window['clipped_feature_values']} feature values clipped" if window["clipped_feature_values"] else ""))
    print(f"[site-memory-ibm] total {the_plan['total_circuits']} circuits x {args.shots} shots "
          f"(scaling {args.scaling}, {the_plan['qubits']} qubits)")

    if args.ideal_kernel == "zz_exact":
        from scripts.models.qml.zz_kernel import zz_kernel_factory

        ideal_kernel = zz_kernel_factory(args.qsvm_reps, args.qsvm_entanglement)
    else:
        ideal_kernel = sm.make_qsvm_kernel(argparse.Namespace(qsvm_kernel="statevector", qsvm_reps=args.qsvm_reps,
                                                              qsvm_entanglement=args.qsvm_entanglement),
                                           len(features))

    raw, record = None, None
    if args.submit:
        raw, record = run_on_hardware(jobs, args, output)
    elif args.reuse_hardware is not None:
        raw, record = load_hardware(jobs, args.reuse_hardware.expanduser().resolve())
    else:
        print("[site-memory-ibm] plan only (no --submit): IBM was not contacted. The SVM and ideal QSVM "
              "are evaluated below; add --submit to run the hardware kernels.")

    rng = np.random.default_rng(args.random_state)
    window_rows, prediction_rows, deviations = [], [], []
    for job in jobs:
        ideal = ideal_matrices(job, ideal_kernel)
        kernels = {"qsvm_ideal": job.split_kernels(ideal)}
        if raw is not None:
            kernels["qsvm_ibm"] = hardware_split_kernels(job, raw)
            deviations.append({"window": job.window, "t_start_fs": job.t_start, **kernel_deviation(job, ideal, raw)})
        rows, predictions = evaluate(job, kernels, args, rng)
        window_rows += rows
        prediction_rows += predictions
        print(f"[site-memory-ibm] {job.t_start:g}-{job.t_end:g} fs  " + "  ".join(
            f"{r['model']}={r['n_correct']}/{r['n_trajectories']}(p={r['p_value']:.3f})" for r in rows))

    pd.DataFrame(window_rows).to_csv(output / "site_memory_ibm_windows.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "site_memory_ibm_predictions.csv", index=False)
    if deviations:
        pd.DataFrame(deviations).to_csv(output / "site_memory_ibm_kernel_deviation.csv", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [str(run) for run in runs], "inputs": inputs, "md_settings": settings,
        "sites": args.sites, "seeds": seeds, "seeds_left_out": dropped, "features": features,
        "window_fs": args.window_fs, "plan": the_plan, "c": args.c,
        "ideal_kernel": args.ideal_kernel,
        "hardware": None if record is None else {
            "backend": record.get("backend"), "job_tag": record.get("job_tag"),
            "shots": args.shots, "optimization_level": args.optimization_level,
            "transpiler_seed": args.transpiler_seed,
            "quantum_seconds": sum(k.get("quantum_seconds") or 0.0 for k in record["kernels"].values()),
            "reused_from": str(args.reuse_hardware) if args.reuse_hardware else None},
        "hardware_kernel_handling": "training diagonal 1, training kernel projected to PSD, test entries as measured",
        "validation": "leave one seed out; same rows, splits, scaling and C for every model",
    }
    (output / "site_memory_ibm_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n",
                                                         encoding="utf-8")
    print(f"[site-memory-ibm] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
