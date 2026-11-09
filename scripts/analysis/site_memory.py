"""How long can the starting adsorption site be identified from the AIMD? (site memory)

Every trajectory starts from the DFT geometry of its site. This analysis cuts the AIMD
into consecutive time windows counted from the start of the equilibration and, for each
window, asks whether the starting site can be predicted from the CO2 geometry in that
window alone.

Data. ``md/<material>/<site>/md_frame_analysis.csv`` of one or more simulation runs
(written by ``scripts.analysis.md_trajectory_analysis``), equilibration and production
frames, on one time axis t = step x timestep. The frame at the end of the equilibration
is also the first production frame; it is counted once. Runs are combined only when
they share temperature, time step, log interval and equilibration length.

Samples. In each window, one row per trajectory: the mean of its frames in the window.
A trajectory is never split across rows of the same window, and every window is
analysed on its own, so no frame of a test trajectory is used in training.

Features (``--features``): geometry of the CO2 only, never the site assignment columns
(nearest site, lateral distance to a site, lateral displacement), which would leak the
label. ``geometry3`` = C height above the top plane, shortest C-slab distance and
shortest CO2-O-slab distance; ``height`` = the first one.

Validation. Leave one seed out: the test set holds the trajectories of one seed (one
per site) and the training set every other seed. Trajectories with the same seed share
their starting velocities and noise in runs with ``rng_stream: shared`` (zno_ic_final),
so they are always on the same side. Only seeds that have all the selected sites are
used. Preprocessing (MinMax to [0, pi/2], the same for every model) is fitted on the
training rows of each split. The upper limit is pi/2, not pi: the single-qubit phase
P(2x) of the ZZ feature map has period pi in x, so with [0, pi] the smallest and the
largest training value of a feature would be encoded as the same phase. The RBF SVM
(gamma = scale) and the nearest centroid do not depend on this range.
Every model uses the same C (``--c``, default 1, no tuning: with 2-5 training seeds per
split there is nothing to tune on).

Statistic and significance. The pooled balanced accuracy of all held-out predictions of
a window (each trajectory is predicted once). Null distribution: the site labels are
permuted within each seed (each seed keeps one trajectory per site) and the whole
leave-one-seed-out evaluation is rerun; all (n_sites!)^n_seeds - 1 relabelings are
enumerated when there are at most ``--n-permutations`` of them, otherwise
``--n-permutations`` are sampled. p = (1 + #{null >= observed}) / (1 + N). Renaming the
sites consistently in every seed gives the same balanced accuracy, so the smallest
attainable p is n_sites! / (n_sites!)^n_seeds: 1/36 for 3 seeds, 1/7776 for 6 seeds.

Memory time of a model: the end of the last window of the run of consecutive windows,
starting at t = 0, whose p-value is <= alpha. The first windows are trivially
identifiable (every trajectory of a site starts from the same geometry); the result is
when the identification stops.

Outputs in ``--output-dir``: site_memory_windows.csv (one row per window and model),
site_memory_predictions.csv, site_memory_features.csv (the per-window dataset),
site_memory_summary.json and site_memory_curve.png.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FRAME_FILE = "md_frame_analysis.csv"
DEFAULT_SITES = ("top_metal", "top_oxygen", "bridge")
FEATURE_SETS = {
    "geometry3": ["c_height_top_plane_ang", "c_min_slab_distance_ang", "c_o_min_slab_distance_ang"],
    "height": ["c_height_top_plane_ang"],
}
# Defined relative to the site positions or to the starting position: never features.
LEAKING_COLUMNS = ("nearest_site", "nearest_site_lateral_ang", "start_site_lateral_ang",
                   "msd_lateral_ang2", "site_assignment")
MODELS = ("majority", "nearest_centroid", "logistic", "svm_rbf", "qsvm")
DEFAULT_MODELS = MODELS
PLOTTED_MODELS = ("svm_rbf", "qsvm", "nearest_centroid")
MODEL_LABELS = {"majority": "Majority", "nearest_centroid": "Nearest centroid", "logistic": "Logistic",
                "svm_rbf": "SVM (RBF)", "qsvm": "QSVM (ZZ)"}
MD_KEYS = ("temperature_kelvin", "timestep_fs", "log_interval", "equilibration_steps")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Site memory: time-resolved identification of the starting site.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="Simulation run directories (results/simulations/<run-id>) with the same MD settings.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    parser.add_argument("--features", default="geometry3", choices=sorted(FEATURE_SETS))
    parser.add_argument("--window-fs", type=float, default=40.0, help="Window width in fs (default 40 = 2 frames).")
    parser.add_argument("--t-max-fs", type=float, default=None,
                        help="Last time analysed (default: the shortest trajectory).")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=MODELS)
    parser.add_argument("--c", type=float, default=1.0, help="C of every SVM-type model and of the logistic model.")
    parser.add_argument("--qsvm-kernel", default="statevector", choices=["statevector", "zz_exact"],
                        help="statevector: Qiskit zz_feature_map with FidelityStatevectorKernel (needs "
                             "qiskit and qiskit-machine-learning); zz_exact: the same kernel computed with "
                             "numpy (tests and machines without qiskit).")
    parser.add_argument("--qsvm-reps", type=int, default=1)
    parser.add_argument("--qsvm-entanglement", default="linear", choices=["linear", "circular", "full"])
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- data

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md_settings(run: Path) -> dict:
    config_path = run / "config.yaml"
    if not config_path.is_file():
        raise SystemExit(f"[site-memory] {config_path} not found; --runs must be simulation run directories.")
    md = ((yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("methods") or {}).get("md") or {}
    missing = [key for key in MD_KEYS if md.get(key) is None]
    if missing:
        raise SystemExit(f"[site-memory] {config_path}: methods.md lacks {missing}.")
    return {key: float(md[key]) for key in MD_KEYS}


def common_settings(runs: list[Path]) -> dict:
    settings = {str(run): md_settings(run) for run in runs}
    first = next(iter(settings.values()))
    for run, values in settings.items():
        different = [key for key in MD_KEYS if not math.isclose(values[key], first[key])]
        if different:
            raise SystemExit(f"[site-memory] {run} differs from {runs[0]} in {different}; "
                             "only runs with the same MD settings can be combined.")
    return first


def load_frames(runs: list[Path], material: str, sites: list[str], timestep_fs: float) -> tuple[pd.DataFrame, list[dict]]:
    """Frames of every (run, site, seed) on one time axis, with the input inventory."""
    tables, inputs = [], []
    for run in runs:
        for site in sites:
            path = run / "md" / material / site / FRAME_FILE
            if not path.is_file():
                raise SystemExit(
                    f"[site-memory] {path} not found. Write it with: python -m "
                    f"scripts.analysis.md_trajectory_analysis --results-dir {run} --material {material} "
                    f"--output-dir <folder>")
            frame = pd.read_csv(path)
            needed = {"seed", "phase", "step"}
            if not needed <= set(frame.columns):
                raise SystemExit(f"[site-memory] {path} lacks columns {sorted(needed - set(frame.columns))}.")
            frame = frame.assign(run=str(run), site=site, seed=frame["seed"].astype(int))
            # Equilibration first, so the duplicated end-of-equilibration frame keeps that row.
            frame["_phase_order"] = (frame["phase"].astype(str) != "equilibration").astype(int)
            frame = frame.sort_values(["seed", "step", "_phase_order"]).drop_duplicates(["seed", "step"])
            frame["t_fs"] = frame["step"].astype(float) * timestep_fs
            tables.append(frame.drop(columns="_phase_order"))
            inputs.append({"path": str(path), "sha256": sha256(path)})
    frames = pd.concat(tables, ignore_index=True)
    owners = frames.groupby(["site", "seed"])["run"].nunique()
    clashes = owners[owners > 1]
    if len(clashes):
        raise SystemExit(f"[site-memory] the same (site, seed) appears in more than one run: "
                         f"{clashes.index.tolist()}; give the runs distinct seeds.")
    return frames, inputs


def complete_seeds(frames: pd.DataFrame, sites: list[str]) -> tuple[list[int], list[int]]:
    """Seeds that have every site (kept) and the others (dropped)."""
    present = frames.groupby("seed")["site"].apply(set)
    kept = sorted(int(seed) for seed, found in present.items() if set(sites) <= found)
    dropped = sorted(int(seed) for seed in present.index if int(seed) not in kept)
    return kept, dropped


def window_edges(frames: pd.DataFrame, width: float, t_max: float | None) -> list[tuple[float, float]]:
    shortest = float(frames.groupby(["site", "seed"])["t_fs"].max().min())
    end = shortest if t_max is None else min(float(t_max), shortest)
    count = int(math.floor(end / width + 1e-9)) + 1  # the last window may start at `end`
    return [(k * width, (k + 1) * width) for k in range(count) if k * width <= end + 1e-9]


def window_dataset(frames: pd.DataFrame, features: list[str], edges, sites: list[str]) -> pd.DataFrame:
    """One row per (window, trajectory): mean of the trajectory's frames in the window."""
    for column in features:
        if column in LEAKING_COLUMNS:
            raise ValueError(f"{column} is defined from the site positions and cannot be a feature.")
    rows = []
    for index, (lo, hi) in enumerate(edges):
        part = frames[(frames["t_fs"] >= lo - 1e-9) & (frames["t_fs"] < hi - 1e-9)]
        means = part.groupby(["site", "seed"])[features].mean()
        counts = part.groupby(["site", "seed"]).size()
        for (site, seed), values in means.iterrows():
            rows.append({"window": index, "t_start_fs": lo, "t_end_fs": hi, "site": site, "seed": int(seed),
                         "label": sites.index(site), "n_frames": int(counts.loc[(site, seed)]),
                         **{name: float(values[name]) for name in features}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- models

# Upper end of the MinMax range: P(2x) of the ZZ feature map has period pi in x.
FEATURE_RANGE = np.pi / 2


def scale_features(train: np.ndarray, test: np.ndarray, top: float = FEATURE_RANGE) -> tuple[np.ndarray, np.ndarray]:
    """MinMax to [0, top] fitted on the training rows (constant columns map to 0)."""
    low, high = train.min(axis=0), train.max(axis=0)
    span = np.where(high - low > 1e-12, high - low, 1.0)
    return top * (train - low) / span, top * (test - low) / span


def make_model(name: str, c: float, random_state: int):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC

    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "nearest_centroid":
        return NearestCentroid()
    if name == "logistic":
        return LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
    if name == "svm_rbf":
        return SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced", random_state=random_state)
    if name == "qsvm":
        return SVC(kernel="precomputed", C=c, class_weight="balanced")
    raise ValueError(f"Unknown model {name!r}")


class SplitCache:
    """Scaled features and QSVM kernels of each leave-one-seed-out split (label independent)."""

    def __init__(self, X: np.ndarray, seeds: np.ndarray, kernel):
        self.splits = []
        for seed in sorted(set(seeds.tolist())):
            test = seeds == seed
            train_x, test_x = scale_features(X[~test], X[test])
            k_train = kernel(train_x) if kernel is not None else None
            k_test = kernel(test_x, train_x) if kernel is not None else None
            self.splits.append((~test, test, train_x, test_x, k_train, k_test))


def make_qsvm_kernel(args, n_features: int):
    """Fidelity kernel of the ZZ feature map, one qubit per feature (Qiskit or numpy)."""
    if args.qsvm_kernel == "zz_exact":
        from scripts.models.qml.zz_kernel import zz_kernel_factory

        return zz_kernel_factory(args.qsvm_reps, args.qsvm_entanglement)
    try:
        from scripts.models.evaluate_grouped import statevector_kernel_factory

        return statevector_kernel_factory(n_features, 0, args.qsvm_reps, args.qsvm_entanglement)
    except ImportError as exc:
        raise SystemExit(
            f"[site-memory] the statevector QSVM needs qiskit and qiskit-machine-learning ({exc}); "
            "run it in the modeling environment, or pass --qsvm-kernel zz_exact."
        ) from exc


def loso_predictions(cache: SplitCache, y: np.ndarray, model: str, c: float, random_state: int) -> np.ndarray:
    predicted = np.empty_like(y)
    for train, test, train_x, test_x, k_train, k_test in cache.splits:
        estimator = make_model(model, c, random_state)
        if model == "qsvm":
            estimator.fit(k_train, y[train])
            predicted[test] = estimator.predict(k_test)
        else:
            estimator.fit(train_x, y[train])
            predicted[test] = estimator.predict(test_x)
    return predicted


def balanced_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean([np.mean(predicted[y == label] == label) for label in np.unique(y)]))


def within_seed_relabelings(y: np.ndarray, seeds: np.ndarray, n_permutations: int,
                            rng: np.random.Generator) -> tuple[list[np.ndarray], bool]:
    """Labelings with the site labels permuted inside each seed (observed one excluded if exact)."""
    groups = [np.flatnonzero(seeds == seed) for seed in sorted(set(seeds.tolist()))]
    per_seed = [list(itertools.permutations(y[index].tolist())) for index in groups]
    total = math.prod(len(options) for options in per_seed)
    if total - 1 <= n_permutations:
        labelings = []
        for choice in itertools.product(*per_seed):
            labels = y.copy()
            for index, values in zip(groups, choice):
                labels[index] = values
            if not np.array_equal(labels, y):
                labelings.append(labels)
        return labelings, True
    labelings = []
    for _ in range(n_permutations):
        labels = y.copy()
        for index in groups:
            labels[index] = rng.permutation(y[index])
        labelings.append(labels)
    return labelings, False


def analyse_window(rows: pd.DataFrame, features: list[str], models: list[str], kernel, args,
                   rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    rows = rows.sort_values(["seed", "label"]).reset_index(drop=True)
    X = rows[features].to_numpy(dtype=float)
    y = rows["label"].to_numpy(dtype=int)
    seeds = rows["seed"].to_numpy(dtype=int)
    cache = SplitCache(X, seeds, kernel if "qsvm" in models else None)
    labelings, exact = within_seed_relabelings(y, seeds, args.n_permutations, rng)
    results, predictions = [], []
    for model in models:
        predicted = loso_predictions(cache, y, model, args.c, args.random_state)
        observed = balanced_accuracy(y, predicted)
        if model == "majority":  # ignores the features: no permutation test
            null = np.array([])
        else:
            null = np.array([balanced_accuracy(labels, loso_predictions(cache, labels, model, args.c,
                                                                         args.random_state))
                             for labels in labelings])
        p_value = float((1 + np.sum(null >= observed - 1e-12)) / (1 + len(null))) if len(null) else float("nan")
        results.append({"model": model, "balanced_accuracy": observed,
                        "n_correct": int(np.sum(predicted == y)), "n_trajectories": int(len(y)),
                        "p_value": p_value, "null_mean": float(null.mean()) if len(null) else float("nan"),
                        "n_null": int(len(null)), "exact_null": bool(exact)})
        predictions += [{"model": model, "seed": int(s), "site_label": int(t), "predicted_label": int(p)}
                        for s, t, p in zip(seeds, y, predicted)]
    return results, predictions


def memory_time(windows: pd.DataFrame, model: str, alpha: float) -> float:
    """End of the last window of the significant run that starts at t = 0 (0 if the first is not)."""
    part = windows[windows["model"] == model].sort_values("t_start_fs")
    end = 0.0
    for _, row in part.iterrows():
        if not (row["p_value"] <= alpha):
            break
        end = float(row["t_end_fs"])
    return end


# --------------------------------------------------------------------------- outputs

def plot_curve(windows: pd.DataFrame, path: Path, title: str, alpha: float, n_classes: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    for model in PLOTTED_MODELS:
        part = windows[windows["model"] == model].sort_values("t_start_fs")
        if part.empty:
            continue
        centre = 0.5 * (part["t_start_fs"] + part["t_end_fs"])
        line, = ax.plot(centre, part["balanced_accuracy"], marker="o", markersize=3.5, linewidth=1.2,
                        markerfacecolor="white", label=MODEL_LABELS[model])
        significant = part["p_value"] <= alpha
        ax.plot(centre[significant], part["balanced_accuracy"][significant], linestyle="none", marker="o",
                markersize=3.5, color=line.get_color())
    ax.axhline(1.0 / n_classes, color="#777777", linestyle="--", linewidth=0.9, label="Chance")
    ax.set_xlabel("Time from the start of the AIMD (fs)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.window_fs <= 0 or args.c <= 0 or args.n_permutations < 0:
        raise SystemExit("[site-memory] --window-fs and --c must be positive and --n-permutations >= 0.")
    runs = [run.expanduser().resolve() for run in args.runs]
    settings = common_settings(runs)
    features = FEATURE_SETS[args.features]
    frames, inputs = load_frames(runs, args.material, args.sites, settings["timestep_fs"])
    seeds, dropped = complete_seeds(frames, args.sites)
    if dropped:
        print(f"[site-memory] seeds without every site are left out: {dropped}")
    if len(seeds) < 2:
        raise SystemExit(f"[site-memory] leave one seed out needs >= 2 complete seeds, found {seeds}.")
    frames = frames[frames["seed"].isin(seeds)]
    edges = window_edges(frames, args.window_fs, args.t_max_fs)
    dataset = window_dataset(frames, features, edges, args.sites)

    kernel = make_qsvm_kernel(args, len(features)) if "qsvm" in args.models else None
    rng = np.random.default_rng(args.random_state)
    window_rows, prediction_rows = [], []
    for index, (lo, hi) in enumerate(edges):
        rows = dataset[dataset["window"] == index]
        if rows.groupby("seed")["site"].nunique().min() < len(args.sites) or rows["seed"].nunique() != len(seeds):
            print(f"[site-memory] window {lo:g}-{hi:g} fs has no frame for some trajectory; skipped.")
            continue
        results, predictions = analyse_window(rows, features, args.models, kernel, args, rng)
        for item in results:
            window_rows.append({"window": index, "t_start_fs": lo, "t_end_fs": hi, **item})
        for item in predictions:
            prediction_rows.append({"window": index, "t_start_fs": lo, "t_end_fs": hi, **item,
                                    "site": args.sites[item["site_label"]],
                                    "predicted_site": args.sites[item["predicted_label"]]})
        best = {r["model"]: r for r in results}
        shown = " ".join(f"{m}={best[m]['n_correct']}/{best[m]['n_trajectories']}(p={best[m]['p_value']:.3f})"
                         for m in args.models if m != "majority")
        print(f"[site-memory] {lo:6.0f}-{hi:<6.0f} fs  {shown}")

    windows = pd.DataFrame(window_rows)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output / "site_memory_windows.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "site_memory_predictions.csv", index=False)
    dataset.to_csv(output / "site_memory_features.csv", index=False)
    memory = {model: memory_time(windows, model, args.alpha) for model in args.models if model != "majority"}
    temperature = settings["temperature_kelvin"]
    title = (f"{args.material}, {temperature:g} K: {len(seeds)} seeds x {len(args.sites)} sites, "
             f"{args.window_fs:g} fs windows")
    plot_curve(windows, output / "site_memory_curve.png", title, args.alpha, len(args.sites))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [str(run) for run in runs],
        "inputs": inputs,
        "md_settings": settings,
        "material": args.material,
        "sites": args.sites,
        "seeds": seeds,
        "seeds_left_out": dropped,
        "features": features,
        "window_fs": args.window_fs,
        "t_max_fs": float(edges[-1][1]) if edges else None,
        "models": args.models,
        "c": args.c,
        "preprocessing": "MinMax to [0, pi/2] fitted on the training seeds of each split (same for every model)",
        "validation": "leave one seed out; one row per trajectory per window",
        "permutation": "site labels permuted within each seed; whole evaluation rerun",
        "qsvm_kernel": {"name": args.qsvm_kernel, "feature_map": "zz_feature_map", "reps": args.qsvm_reps,
                        "entanglement": args.qsvm_entanglement, "qubits": len(features)}
        if "qsvm" in args.models else None,
        "alpha": args.alpha,
        "chance_level": 1.0 / len(args.sites),
        "memory_time_fs": memory,
        "memory_time_definition": ("end of the last window of the run of consecutive windows from t = 0 "
                                   "with p <= alpha"),
        "note": "t = 0 is the DFT geometry of each site, so the first windows are identifiable by construction.",
    }
    (output / "site_memory_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[site-memory] memory time (fs, p <= {args.alpha}): " +
          ", ".join(f"{model}={value:g}" for model, value in memory.items()))
    print(f"[site-memory] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
