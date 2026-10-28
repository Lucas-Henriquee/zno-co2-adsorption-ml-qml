"""Grouped evaluation protocol shared by every model (SVM, QSVM and baselines).

The MD-window datasets contain several windows (blocks) of each production
trajectory. Windows of one trajectory are pseudo-replicates: the statistical
unit is the trajectory (``group`` = material/site/seed). With 3 sites x 3 seeds
there are only 9 independent units, so a single holdout of 3 trajectories says
very little. This module evaluates every model on the same, exhaustive set of
grouped splits and reports uncertainty and significance at the trajectory level.

Protocol (identical for every model):

1. Outer evaluation (``--outer``):
   * ``one_per_class``: every split that leaves exactly one trajectory of each
     class out; the number of splits is the product of the trajectories per
     class (3 x 3 x 3 = 27 for 3 seeds per site). Train on the rest.
   * ``loto``: leave-one-trajectory-out (one split per trajectory).
   * ``both`` (default) runs both schemes. Every trajectory is predicted 9 times
     in ``one_per_class`` (the splits overlap), so the exact confidence interval
     uses ``loto``, where each trajectory is predicted exactly once.
2. Hyperparameters (``--tuning``): ``fixed`` values, or ``nested`` grouped CV
   (StratifiedGroupKFold with n_splits = smallest number of training
   trajectories per class) inside the training trajectories of each outer split.
   The outer test fold is never used for selection.
3. Preprocessing (``--preprocessing``, default ``power_minmax`` = Yeo-Johnson
   power transform then MinMax to [0, pi]) is the same for every model and is
   fitted on the training rows of each (outer or inner) split only. It is
   unsupervised, so it does not depend on the labels.
4. Metrics per split: balanced accuracy per window and per trajectory
   (majority vote of the trajectory's windows; a tied vote counts as wrong).
   Aggregates: mean and distribution over splits, pooled LOTO correct count with
   an exact Clopper-Pearson interval, and chance level 1 / n_classes.
5. Significance: label permutation at the trajectory level. Labels are permuted
   among trajectories (each keeps a single label, class sizes are kept) and the
   whole protocol is rerun, including tuning. The distinct relabelings, i.e.
   the partitions of the trajectories into classes (renaming classes of equal
   size gives the same partition and the same balanced accuracy; 280 for
   3 x 3), are enumerated when there are at most ``--n-permutations`` + 1 of
   them; otherwise ``--n-permutations`` are sampled.
   p = (1 + #{null >= observed}) / (1 + N).
6. Baselines on the same splits: majority class, nearest centroid, logistic
   regression, linear SVM and RBF SVM. The QSVM uses a pluggable kernel
   (``--qsvm-kernel``); qiskit is imported only for ``statevector``.
   Paired comparisons (``--compare A:B``) report per-split differences, a sign
   test, a paired sign-flip permutation test and an exact McNemar test on LOTO.
   With 9 trajectories these tests have very low power; the report says so.

Outputs: ``per_split.csv``, ``trajectory_predictions.csv``,
``permutation_null.csv``, ``paired_differences.csv``, ``summary.json`` and
``report.md`` in ``<output-dir>/<run-name>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

try:
    from scripts.common.project_config import RESULTS_DIR
    from scripts.common.run_names import next_run_name
    from scripts.models.grouped_split import (
        DEFAULT_GROUP_COLUMN,
        InsufficientGroupsError,
        check_groups_have_single_label,
        insufficient_groups_message,
        load_grouped_dataset,
    )
    from scripts.models.ml.classical_svm_baseline import FEATURE_SETS, SCALERS, make_scaler
except ModuleNotFoundError:  # executed as a file from outside the project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.common.project_config import RESULTS_DIR
    from scripts.common.run_names import next_run_name
    from scripts.models.grouped_split import (
        DEFAULT_GROUP_COLUMN,
        InsufficientGroupsError,
        check_groups_have_single_label,
        insufficient_groups_message,
        load_grouped_dataset,
    )
    from scripts.models.ml.classical_svm_baseline import FEATURE_SETS, SCALERS, make_scaler


OUTER_SCHEMES = ("one_per_class", "loto")
ALL_MODELS = ("majority", "nearest_centroid", "logistic", "linear_svm", "svm_rbf", "qsvm")
BUILTIN_KERNELS = ("statevector", "product_angle")
DEFAULT_FIXED_PARAMS = {
    "logistic": {"C": 1.0},
    "linear_svm": {"C": 1.0},
    "svm_rbf": {"C": 1.0, "gamma": "scale"},
    "qsvm": {"C": 1.0},
}
DEFAULT_PARAM_GRID = {
    "logistic": {"C": [0.1, 1.0, 10.0]},
    "linear_svm": {"C": [0.1, 1.0, 10.0]},
    "svm_rbf": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.1, 1.0]},
    "qsvm": {"C": [0.1, 1.0, 10.0]},
}
# Above this many non-zero paired differences the sign-flip test is sampled.
EXACT_SIGN_FLIP_LIMIT = 20
TIE_VOTE = None


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #

KernelFn = Callable[[np.ndarray, "np.ndarray | None"], np.ndarray]


def product_angle_kernel(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Fidelity kernel of the product state RY(x_1) ... RY(x_d)|0...0>.

    K(x, y) = prod_i cos^2((x_i - y_i) / 2). It is a genuine (unentangled)
    quantum fidelity kernel that is classically computable, so the protocol can
    be run and tested without qiskit. It is NOT the ZZ feature map of the QSVM
    scripts; use ``statevector`` for that.
    """
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]
    return np.prod(np.cos(diff / 2.0) ** 2, axis=2)


def statevector_kernel_factory(n_features: int, n_qubits: int = 0, reps: int = 1,
                               entanglement: str = "linear") -> KernelFn:
    """Exact ZZ-feature-map fidelity kernel, the one used by qsvm_baseline.py.

    qiskit and qiskit-machine-learning are imported here, lazily.
    """
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    from scripts.models.qml.qsvm_baseline import build_feature_map

    feature_map, _ = build_feature_map(n_features, n_qubits, reps, entanglement)
    kernel = FidelityStatevectorKernel(feature_map=feature_map, shots=None, enforce_psd=True)

    def evaluate(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        if B is None:
            return np.asarray(kernel.evaluate(x_vec=A), dtype=float)
        return np.asarray(kernel.evaluate(x_vec=A, y_vec=B), dtype=float)

    return evaluate


def resolve_kernel(name: str, n_features: int, n_qubits: int = 0, reps: int = 1,
                   entanglement: str = "linear") -> KernelFn:
    """Return a kernel function K(A, B=None) from a built-in name or 'module:callable'.

    A plugin callable must accept (A, B=None) on preprocessed features and return
    the kernel matrix (B=None means the symmetric training matrix).
    """
    if name == "statevector":
        return statevector_kernel_factory(n_features, n_qubits, reps, entanglement)
    if name == "product_angle":
        return product_angle_kernel
    if ":" in name:
        module_name, attribute = name.split(":", 1)
        function = getattr(importlib.import_module(module_name), attribute)
        if not callable(function):
            raise ValueError(f"Kernel plugin {name!r} is not callable.")
        return function
    raise ValueError(f"Unknown kernel {name!r}; use {BUILTIN_KERNELS} or 'module:callable'.")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

@dataclass
class ModelSpec:
    """One model of the protocol; ``kernel`` is set for precomputed-kernel SVMs."""

    name: str
    build: Callable[[dict], object]
    grid: list[dict]
    fixed: dict
    kernel: KernelFn | None = None


def _class_weight(name: str):
    return None if name == "none" else name


def _gamma(value):
    if isinstance(value, str) and value not in {"scale", "auto"}:
        return float(value)
    return value


def expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def build_model_specs(
    names: list[str],
    class_weight: str = "balanced",
    kernel: KernelFn | None = None,
    param_grid: dict | None = None,
    fixed_params: dict | None = None,
    random_state: int = 42,
) -> list[ModelSpec]:
    grids = {**DEFAULT_PARAM_GRID, **(param_grid or {})}
    fixed = {name: {**DEFAULT_FIXED_PARAMS.get(name, {}), **(fixed_params or {}).get(name, {})}
             for name in ALL_MODELS}
    weight = _class_weight(class_weight)
    builders = {
        # Ignores the features; predicts the most frequent training class.
        "majority": lambda p: DummyClassifier(strategy="most_frequent"),
        "nearest_centroid": lambda p: NearestCentroid(),
        "logistic": lambda p: LogisticRegression(C=float(p["C"]), class_weight=weight, max_iter=5000),
        "linear_svm": lambda p: SVC(kernel="linear", C=float(p["C"]), class_weight=weight,
                                    max_iter=100_000, random_state=random_state),
        "svm_rbf": lambda p: SVC(kernel="rbf", C=float(p["C"]), gamma=_gamma(p["gamma"]),
                                 class_weight=weight, max_iter=100_000, random_state=random_state),
        "qsvm": lambda p: SVC(kernel="precomputed", C=float(p["C"]), class_weight=weight),
    }
    specs = []
    for name in names:
        if name not in builders:
            raise ValueError(f"Unknown model {name!r}; choose from {ALL_MODELS}.")
        if name == "qsvm" and kernel is None:
            raise ValueError("Model 'qsvm' needs a kernel function (see --qsvm-kernel).")
        specs.append(ModelSpec(
            name=name,
            build=builders[name],
            grid=expand_grid(grids.get(name, {})),
            fixed=fixed.get(name, {}),
            kernel=kernel if name == "qsvm" else None,
        ))
    return specs


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def groups_by_class(group_labels: pd.Series) -> dict:
    """Map class -> sorted list of groups, from a Series indexed by group."""
    return {cls: sorted(group_labels.index[group_labels == cls].tolist())
            for cls in sorted(group_labels.unique().tolist())}


def outer_splits(group_labels: pd.Series, scheme: str) -> list[tuple[str, ...]]:
    """Return the test groups of every outer split.

    ``one_per_class``: one trajectory of each class out, all combinations
    (product of the trajectories per class). ``loto``: one trajectory out.
    Training keeps every other trajectory; at least one per class must remain.
    """
    per_class = groups_by_class(group_labels)
    counts = pd.Series({cls: len(groups) for cls, groups in per_class.items()})
    if scheme == "one_per_class":
        if counts.min() < 2:
            raise InsufficientGroupsError(insufficient_groups_message(counts, 2, "outer evaluation"))
        return [tuple(combo) for combo in itertools.product(*per_class.values())]
    if scheme == "loto":
        if counts.min() < 2:
            raise InsufficientGroupsError(insufficient_groups_message(counts, 2, "outer evaluation"))
        return [(name,) for name in sorted(group_labels.index.tolist())]
    raise ValueError(f"Unknown outer scheme {scheme!r}; use {OUTER_SCHEMES}.")


def count_distinct_relabelings(group_labels: pd.Series) -> int:
    """Number of distinct assignments of the class labels to the groups."""
    counts = group_labels.value_counts().to_numpy()
    return math.factorial(int(counts.sum())) // math.prod(math.factorial(int(c)) for c in counts)


def count_distinct_partitions(group_labels: pd.Series) -> int:
    """Distinct partitions of the groups into classes, ignoring class names.

    Relabelings that only swap the names of classes with the same number of
    groups give the same partition, and class-symmetric metrics such as balanced
    accuracy do not distinguish them. For 3 classes x 3 groups: 1680 / 3! = 280.
    """
    sizes = pd.Series(group_labels.value_counts().to_numpy()).value_counts()
    return count_distinct_relabelings(group_labels) // math.prod(math.factorial(int(m)) for m in sizes)


def _multiset_permutations(items: list):
    items = sorted(items)

    def recurse(prefix: list, remaining: list):
        if not remaining:
            yield tuple(prefix)
            return
        previous = object()
        for index, item in enumerate(remaining):
            if item == previous:
                continue
            previous = item
            yield from recurse(prefix + [item], remaining[:index] + remaining[index + 1:])

    yield from recurse([], items)


def _partition_key(groups: list, labels: tuple) -> frozenset:
    blocks: dict = {}
    for name, value in zip(groups, labels):
        blocks.setdefault(value, set()).add(name)
    return frozenset(frozenset(block) for block in blocks.values())


# Above this many labelings the exact enumeration is not attempted.
MAX_ENUMERATED_LABELINGS = 1_000_000


def group_relabelings(group_labels: pd.Series, n_permutations: int,
                      rng: np.random.Generator) -> tuple[list[pd.Series], bool]:
    """Null relabelings of the groups, keeping class sizes and one label per group.

    Returns (relabelings, exact). When the distinct partitions of the groups
    into classes, other than the observed one, number at most ``n_permutations``,
    one labeling per partition is enumerated (exact test, observed partition
    excluded). Otherwise ``n_permutations`` random permutations of the group
    labels are drawn (Monte Carlo; the observed labeling may reappear).
    """
    if n_permutations <= 0:
        return [], False
    names = group_labels.index.tolist()
    observed = tuple(group_labels.tolist())
    n_partitions = count_distinct_partitions(group_labels)
    if (n_partitions - 1 <= n_permutations
            and count_distinct_relabelings(group_labels) <= MAX_ENUMERATED_LABELINGS):
        seen = {_partition_key(names, observed)}
        labelings = []
        for candidate in _multiset_permutations(list(observed)):
            key = _partition_key(names, candidate)
            if key not in seen:
                seen.add(key)
                labelings.append(candidate)
        exact = True
    else:
        labelings = [tuple(rng.permutation(np.asarray(observed, dtype=object)).tolist())
                     for _ in range(n_permutations)]
        exact = False
    return [pd.Series(list(p), index=group_labels.index, name=group_labels.name) for p in labelings], exact


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for k successes out of n."""
    if n <= 0:
        return float("nan"), float("nan")
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def permutation_p_value(observed: float, null: np.ndarray | list[float], tol: float = 1e-12) -> float:
    """p = (1 + #{null >= observed}) / (1 + N)."""
    null = np.asarray(null, dtype=float)
    return float((1 + int(np.sum(null >= observed - tol))) / (1 + len(null)))


def sign_test(differences: np.ndarray) -> dict:
    nonzero = differences[differences != 0]
    positive = int(np.sum(nonzero > 0))
    n = int(len(nonzero))
    p = 1.0 if n == 0 else float(stats.binomtest(positive, n, 0.5, alternative="two-sided").pvalue)
    return {"n_nonzero": n, "n_positive": positive, "n_negative": n - positive, "p_two_sided": p}


def paired_sign_flip_test(differences: np.ndarray, rng: np.random.Generator,
                          n_samples: int = 10_000) -> dict:
    """Two-sided paired permutation (sign-flip) test of the mean difference."""
    nonzero = differences[differences != 0]
    observed = float(np.mean(differences)) if len(differences) else 0.0
    n = len(nonzero)
    if n == 0:
        return {"observed_mean_difference": observed, "p_two_sided": 1.0, "exact": True, "n_flips": 1}
    total = float(np.sum(nonzero))
    if n <= EXACT_SIGN_FLIP_LIMIT:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        exact = True
    else:
        signs = rng.choice([1.0, -1.0], size=(n_samples, n))
        exact = False
    null = np.abs(signs @ nonzero)
    count = int(np.sum(null >= abs(total) - 1e-12))
    p = count / len(signs) if exact else (1 + count) / (1 + len(signs))
    return {"observed_mean_difference": observed, "p_two_sided": float(min(1.0, p)),
            "exact": exact, "n_flips": int(len(signs))}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(stats.binomtest(a_only, n, 0.5, alternative="two-sided").pvalue)
    return {"a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only, "p_two_sided": p,
            "min_achievable_p_with_n_units": float(min(1.0, 2 * 0.5 ** len(correct_a)))}


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over the classes present in y_true (as sklearn's balanced_accuracy_score).

    Implemented directly because the permutation test evaluates it many
    thousands of times and sklearn's input validation dominates the run time.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = [float(np.mean(y_pred[y_true == cls] == cls)) for cls in np.unique(y_true)]
    return float(np.mean(recalls))


def trajectory_votes(groups: np.ndarray, y_pred: np.ndarray) -> dict:
    """Majority vote of the window predictions of each group; ties give TIE_VOTE."""
    votes = {}
    for name in sorted(set(groups.tolist())):
        values, counts = np.unique(y_pred[groups == name], return_counts=True)
        winners = values[counts == counts.max()]
        votes[name] = winners[0].item() if len(winners) == 1 else TIE_VOTE
    return votes


def trajectory_balanced_accuracy(true_by_group: dict, vote_by_group: dict) -> tuple[float, int, int]:
    """Mean over classes of the fraction of trajectories classified correctly."""
    per_class: dict = {}
    for name, vote in vote_by_group.items():
        truth = true_by_group[name]
        per_class.setdefault(truth, []).append(vote is not TIE_VOTE and vote == truth)
    correct = sum(sum(values) for values in per_class.values())
    recall = [float(np.mean(values)) for values in per_class.values()]
    return float(np.mean(recall)), int(correct), int(len(vote_by_group))


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #

@dataclass
class ProtocolSettings:
    features: list[str]
    label: str = "label_site_id"
    group: str = DEFAULT_GROUP_COLUMN
    outer: str = "both"
    preprocessing: str = "power_minmax"
    tuning: str = "nested"
    n_permutations: int = 1000
    permutation_models: list[str] | None = None
    permutation_scheme: str = "one_per_class"
    compare: list[tuple[str, str]] = field(default_factory=list)
    alpha: float = 0.05
    random_state: int = 42
    n_jobs: int = 1

    def schemes(self) -> list[str]:
        return list(OUTER_SCHEMES) if self.outer == "both" else [self.outer]


def make_preprocessor(name: str):
    """Fresh, unfitted preprocessing step; 'none' returns None."""
    scaler = make_scaler(name)
    return None if scaler == "passthrough" else scaler


class _Cache:
    """Caches preprocessed features and kernel matrices by (train rows, eval rows).

    Preprocessing is unsupervised and kernels do not depend on labels, so the
    cache stays valid across hyperparameters and label permutations.
    """

    def __init__(self, X: np.ndarray, preprocessing: str):
        self.X = X
        self.preprocessing = preprocessing
        self.features: dict = {}
        self.kernels: dict = {}

    def transformed(self, train_idx: np.ndarray, eval_idx: np.ndarray):
        key = (train_idx.tobytes(), eval_idx.tobytes())
        if key not in self.features:
            scaler = make_preprocessor(self.preprocessing)
            X_train, X_eval = self.X[train_idx], self.X[eval_idx]
            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
                X_eval = scaler.transform(X_eval)
            self.features[key] = (np.asarray(X_train, dtype=float), np.asarray(X_eval, dtype=float))
        return self.features[key]

    def kernel(self, spec: ModelSpec, train_idx: np.ndarray, eval_idx: np.ndarray):
        key = (spec.name, train_idx.tobytes(), eval_idx.tobytes())
        if key not in self.kernels:
            X_train, X_eval = self.transformed(train_idx, eval_idx)
            self.kernels[key] = (np.asarray(spec.kernel(X_train, None), dtype=float),
                                 np.asarray(spec.kernel(X_eval, X_train), dtype=float))
        return self.kernels[key]


def fit_predict(spec: ModelSpec, params: dict, cache: _Cache, y: np.ndarray,
                train_idx: np.ndarray, eval_idx: np.ndarray) -> np.ndarray:
    model = spec.build(params)
    if spec.kernel is not None:
        K_train, K_eval = cache.kernel(spec, train_idx, eval_idx)
        model.fit(K_train, y[train_idx])
        return model.predict(K_eval)
    X_train, X_eval = cache.transformed(train_idx, eval_idx)
    model.fit(X_train, y[train_idx])
    return model.predict(X_eval)


def inner_folds(y: np.ndarray, groups: np.ndarray, train_idx: np.ndarray,
                random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped stratified folds over the training trajectories only."""
    train_groups = pd.Series(groups[train_idx]).groupby(y[train_idx]).nunique()
    n_splits = int(train_groups.min())
    if n_splits < 2:
        raise InsufficientGroupsError(
            insufficient_groups_message(train_groups, 2, "nested CV inside the training trajectories")
            + " Nested tuning needs at least 3 trajectories per site in the dataset (one held "
            "out by the outer split, two for the inner folds). Use --tuning fixed, or run more "
            "seeds per site."
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for tr, va in splitter.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx]):
        folds.append((np.sort(train_idx[tr]), np.sort(train_idx[va])))
    return folds


def select_params(spec: ModelSpec, settings: ProtocolSettings, cache: _Cache, y: np.ndarray,
                  groups: np.ndarray, train_idx: np.ndarray) -> tuple[dict, float | None]:
    """Choose hyperparameters without looking at the outer test fold.

    Nested tuning maximises the mean inner window-level balanced accuracy; ties
    go to the earlier grid entry (grids are listed from most to least
    regularised C).
    """
    if settings.tuning == "fixed" or len(spec.grid) <= 1:
        return (spec.fixed if settings.tuning == "fixed" or not spec.grid else spec.grid[0]), None
    folds = inner_folds(y, groups, train_idx, settings.random_state)
    best_params, best_score = None, -np.inf
    for params in spec.grid:
        scores = [balanced_accuracy(y[va], fit_predict(spec, params, cache, y, tr, va))
                  for tr, va in folds]
        score = float(np.mean(scores))
        if score > best_score + 1e-12:
            best_params, best_score = params, score
    return best_params, best_score


def run_protocol(
    df: pd.DataFrame,
    specs: list[ModelSpec],
    settings: ProtocolSettings,
    cache: _Cache | None = None,
    schemes: list[str] | None = None,
    collect_trajectories: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model on every outer split; returns (per_split, trajectories).

    ``df`` must be reset to a 0..n-1 index (rows are addressed by position).
    """
    check_groups_have_single_label(df, settings.label, settings.group)
    X = df[settings.features].to_numpy(dtype=float)
    y = df[settings.label].to_numpy()
    groups = df[settings.group].astype(str).to_numpy()
    cache = cache or _Cache(X, settings.preprocessing)
    group_labels = df.groupby(settings.group, sort=True)[settings.label].first()
    group_labels.index = group_labels.index.astype(str)

    split_rows, trajectory_rows = [], []
    for scheme in schemes or settings.schemes():
        for split_id, test_groups in enumerate(outer_splits(group_labels, scheme)):
            is_test = np.isin(groups, test_groups)
            train_idx, test_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
            for spec in specs:
                params, inner_score = select_params(spec, settings, cache, y, groups, train_idx)
                y_pred = fit_predict(spec, params, cache, y, train_idx, test_idx)
                votes = trajectory_votes(groups[test_idx], y_pred)
                truth = {g: group_labels[g] for g in votes}
                traj_ba, n_correct, n_traj = trajectory_balanced_accuracy(truth, votes)
                # LOTO test folds hold one class; balanced accuracy is then its recall.
                window_ba = balanced_accuracy(y[test_idx], y_pred)
                split_rows.append({
                    "scheme": scheme,
                    "split_id": split_id,
                    "model": spec.name,
                    "test_groups": "|".join(test_groups),
                    "n_train_rows": int(len(train_idx)),
                    "n_test_rows": int(len(test_idx)),
                    "n_train_groups": int(len(np.unique(groups[train_idx]))),
                    "n_test_groups": int(n_traj),
                    "window_accuracy": float(np.mean(y_pred == y[test_idx])),
                    "window_balanced_accuracy": window_ba,
                    "trajectory_balanced_accuracy": traj_ba,
                    "trajectories_correct": n_correct,
                    "params": json.dumps(params, sort_keys=True),
                    "inner_cv_balanced_accuracy": inner_score,
                })
                for name in (votes if collect_trajectories else ()):
                    mask = groups[test_idx] == name
                    trajectory_rows.append({
                        "scheme": scheme,
                        "split_id": split_id,
                        "model": spec.name,
                        "group": name,
                        "true_label": truth[name],
                        "vote": votes[name],
                        "correct": bool(votes[name] is not TIE_VOTE and votes[name] == truth[name]),
                        "n_windows": int(mask.sum()),
                        "window_accuracy": float(np.mean(y_pred[mask] == y[test_idx][mask])),
                    })
    return pd.DataFrame(split_rows), pd.DataFrame(trajectory_rows)


def protocol_statistic(per_split: pd.DataFrame, model: str, scheme: str) -> float:
    """Test statistic: mean trajectory-level balanced accuracy over the splits."""
    rows = per_split[(per_split["model"] == model) & (per_split["scheme"] == scheme)]
    return float(rows["trajectory_balanced_accuracy"].mean())


def _null_statistics(df: pd.DataFrame, specs: list[ModelSpec], settings: ProtocolSettings,
                     indexed_labelings: list[tuple[int, pd.Series]], cache: _Cache | None) -> list[dict]:
    if cache is None:
        cache = _Cache(df[settings.features].to_numpy(dtype=float), settings.preprocessing)
    groups = df[settings.group].astype(str)
    rows = []
    for index, labeling in indexed_labelings:
        permuted = df.copy()
        permuted[settings.label] = groups.map(labeling).to_numpy()
        per_split, _ = run_protocol(permuted, specs, settings, cache=cache,
                                    schemes=[settings.permutation_scheme], collect_trajectories=False)
        for spec in specs:
            rows.append({"model": spec.name, "permutation": index,
                         "statistic": protocol_statistic(per_split, spec.name, settings.permutation_scheme)})
    return rows


def permutation_test(
    df: pd.DataFrame,
    specs: list[ModelSpec],
    settings: ProtocolSettings,
    observed: dict[str, float],
    cache: _Cache,
) -> tuple[dict, pd.DataFrame]:
    """Rerun the full protocol (tuning included) on trajectory-level relabelings."""
    models = settings.permutation_models
    selected = [spec for spec in specs if models is None or spec.name in models]
    if settings.n_permutations <= 0 or not selected:
        return {}, pd.DataFrame(columns=["model", "permutation", "statistic"])
    rng = np.random.default_rng(settings.random_state)
    group_labels = df.groupby(settings.group, sort=True)[settings.label].first()
    group_labels.index = group_labels.index.astype(str)
    relabelings, exact = group_relabelings(group_labels, settings.n_permutations, rng)
    indexed = list(enumerate(relabelings))
    if settings.n_jobs == 1 or len(indexed) < 2:
        rows = _null_statistics(df, selected, settings, indexed, cache)
    else:
        from joblib import Parallel, delayed, effective_n_jobs

        # One chunk per worker, each seeded with the parent's cache, so the
        # (label-independent) preprocessing and kernels are reused within a worker.
        n_chunks = min(len(indexed), effective_n_jobs(settings.n_jobs))
        chunks = [indexed[i::n_chunks] for i in range(n_chunks)]
        parts = Parallel(n_jobs=n_chunks)(
            delayed(_null_statistics)(df, selected, settings, chunk, cache) for chunk in chunks)
        rows = sorted((row for part in parts for row in part), key=lambda r: (r["permutation"], r["model"]))
    null = pd.DataFrame(rows)
    results = {}
    for spec in selected:
        values = null.loc[null["model"] == spec.name, "statistic"].to_numpy()
        results[spec.name] = {
            "scheme": settings.permutation_scheme,
            "statistic": "mean trajectory-level balanced accuracy over outer splits",
            "observed": observed[spec.name],
            "n_permutations": int(len(values)),
            "exact_enumeration": bool(exact),
            "distinct_relabelings": int(count_distinct_relabelings(group_labels)),
            "distinct_partitions": int(count_distinct_partitions(group_labels)),
            "unit_enumerated": ("partition of trajectories into classes (class renamings "
                                "of equal-size classes are equivalent)" if exact else None),
            "null_mean": float(np.mean(values)),
            "null_quantile_95": float(np.quantile(values, 0.95)),
            "p_value": permutation_p_value(observed[spec.name], values),
            "min_achievable_p": float(1.0 / (1 + len(values))),
        }
    return results, null


def summarize_models(per_split: pd.DataFrame, trajectories: pd.DataFrame,
                     n_classes: int, alpha: float) -> dict:
    out = {}
    for model, rows in per_split.groupby("model", sort=False):
        entry = {}
        for scheme, sub in rows.groupby("scheme", sort=False):
            traj = sub["trajectory_balanced_accuracy"]
            entry[scheme] = {
                "n_splits": int(len(sub)),
                "window_balanced_accuracy_mean": float(sub["window_balanced_accuracy"].mean()),
                "window_balanced_accuracy_std": float(sub["window_balanced_accuracy"].std(ddof=0)),
                "trajectory_balanced_accuracy_mean": float(traj.mean()),
                "trajectory_balanced_accuracy_std": float(traj.std(ddof=0)),
                "trajectory_balanced_accuracy_min": float(traj.min()),
                "trajectory_balanced_accuracy_median": float(traj.median()),
                "trajectory_balanced_accuracy_max": float(traj.max()),
                "trajectories_correct_distribution": {
                    str(int(k)): int(v)
                    for k, v in sub["trajectories_correct"].value_counts().sort_index().items()
                },
                "selected_params_counts": {
                    str(k): int(v) for k, v in sub["params"].value_counts().items()
                },
            }
        loto = trajectories[(trajectories["model"] == model) & (trajectories["scheme"] == "loto")]
        if len(loto):
            k, n = int(loto["correct"].sum()), int(len(loto))
            lower, upper = clopper_pearson(k, n, alpha)
            entry["loto_pooled"] = {
                "trajectories_correct": k,
                "trajectories": n,
                "accuracy": k / n,
                "clopper_pearson_level": 1 - alpha,
                "clopper_pearson_lower": lower,
                "clopper_pearson_upper": upper,
                "chance_level": 1.0 / n_classes,
                "interval_excludes_chance": bool(lower > 1.0 / n_classes),
            }
        out[model] = entry
    return out


def paired_comparisons(per_split: pd.DataFrame, trajectories: pd.DataFrame,
                       pairs: list[tuple[str, str]], rng: np.random.Generator,
                       n_units: int) -> tuple[dict, pd.DataFrame]:
    results, difference_rows = {}, []
    models = set(per_split["model"])
    for a, b in pairs:
        if a not in models or b not in models:
            continue
        key = f"{a}_vs_{b}"
        entry = {"model_a": a, "model_b": b}
        for scheme, sub in per_split.groupby("scheme", sort=False):
            left = sub[sub["model"] == a].set_index("split_id")
            right = sub[sub["model"] == b].set_index("split_id")
            scheme_entry = {}
            for metric in ("trajectory_balanced_accuracy", "window_balanced_accuracy"):
                diff = (left[metric] - right[metric]).to_numpy(dtype=float)
                scheme_entry[metric] = {
                    "mean_difference": float(np.mean(diff)),
                    "n_a_better": int(np.sum(diff > 0)),
                    "n_b_better": int(np.sum(diff < 0)),
                    "n_ties": int(np.sum(diff == 0)),
                    "sign_test": sign_test(diff),
                    "paired_permutation": paired_sign_flip_test(diff, rng),
                }
                for split_id, value in zip(left.index, diff):
                    difference_rows.append({"comparison": key, "scheme": scheme, "split_id": int(split_id),
                                            "metric": metric, "difference_a_minus_b": float(value)})
            entry[scheme] = scheme_entry
        loto = trajectories[trajectories["scheme"] == "loto"]
        if len(loto):
            ca = loto[loto["model"] == a].set_index("group")["correct"]
            cb = loto[loto["model"] == b].set_index("group")["correct"].reindex(ca.index)
            entry["loto_mcnemar_exact"] = mcnemar_exact(ca.to_numpy(bool), cb.to_numpy(bool))
        entry["power_note"] = (
            f"Low power: only {n_units} independent trajectories. The outer splits share "
            "trajectories, so per-split differences are not independent and the sign and "
            "paired permutation tests over splits are optimistic (anti-conservative). The "
            "McNemar test on LOTO uses each trajectory once but cannot reach small p-values "
            "unless almost every trajectory is discordant. A non-significant result is not "
            "evidence that the models are equivalent."
        )
        results[key] = entry
    return results, pd.DataFrame(difference_rows)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_report(summary: dict) -> str:
    lines = ["# Grouped evaluation report", ""]
    data = summary["data"]
    lines += [
        f"- Dataset: `{data['csv']}` (sha256 `{data['sha256']}`)",
        f"- Rows (windows): {data['n_rows']}; trajectories: {data['n_groups']}; "
        f"trajectories per class: {data['groups_per_class']}",
        f"- Features: {', '.join(summary['protocol']['features'])}",
        f"- Preprocessing: `{summary['protocol']['preprocessing']}` fitted on training rows only; "
        f"tuning: `{summary['protocol']['tuning']}`",
        f"- Chance level (balanced accuracy): {_fmt(summary['chance_level'])}",
        "",
        "The statistical unit is the trajectory. Windows of one trajectory are correlated "
        "pseudo-replicates; window-level scores are reported for completeness only.",
        "",
        "## Results per model",
        "",
        "| model | scheme | splits | window BA mean | trajectory BA mean (min-max) | "
        "LOTO correct | 95% CI (Clopper-Pearson) | permutation p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, entry in summary["models"].items():
        perm = summary["permutation"].get(model, {})
        loto = entry.get("loto_pooled", {})
        for scheme in OUTER_SCHEMES:
            if scheme not in entry:
                continue
            e = entry[scheme]
            ci = (f"[{_fmt(loto.get('clopper_pearson_lower'))}, {_fmt(loto.get('clopper_pearson_upper'))}]"
                  if scheme == "loto" and loto else "")
            correct = (f"{loto['trajectories_correct']}/{loto['trajectories']}"
                       if scheme == "loto" and loto else "")
            p = _fmt(perm.get("p_value"), 4) if perm and perm.get("scheme") == scheme else ""
            lines.append(
                f"| {model} | {scheme} | {e['n_splits']} | {_fmt(e['window_balanced_accuracy_mean'])} | "
                f"{_fmt(e['trajectory_balanced_accuracy_mean'])} "
                f"({_fmt(e['trajectory_balanced_accuracy_min'])}-{_fmt(e['trajectory_balanced_accuracy_max'])}) | "
                f"{correct} | {ci} | {p} |"
            )
    lines += ["", "## Permutation test (trajectory-level labels)", ""]
    if summary["permutation"]:
        for model, perm in summary["permutation"].items():
            lines.append(
                f"- {model}: observed {_fmt(perm['observed'])}, null mean {_fmt(perm['null_mean'])}, "
                f"p = {_fmt(perm['p_value'], 4)} over {perm['n_permutations']} "
                f"{'enumerated' if perm['exact_enumeration'] else 'sampled'} relabelings "
                f"({perm['distinct_partitions']} distinct partitions of the trajectories into "
                f"classes exist; smallest possible p = "
                f"{_fmt(perm['min_achievable_p'], 4)})."
            )
    else:
        lines.append("- Not run (--n-permutations 0).")
    lines += ["", "## Paired comparisons", ""]
    if summary["comparisons"]:
        for key, entry in summary["comparisons"].items():
            lines.append(f"### {entry['model_a']} vs {entry['model_b']}")
            lines.append("")
            for scheme in OUTER_SCHEMES:
                if scheme not in entry:
                    continue
                t = entry[scheme]["trajectory_balanced_accuracy"]
                lines.append(
                    f"- {scheme}, trajectory BA: mean difference {_fmt(t['mean_difference'])}; "
                    f"A better in {t['n_a_better']}, B better in {t['n_b_better']}, ties {t['n_ties']}; "
                    f"sign test p = {_fmt(t['sign_test']['p_two_sided'], 4)}; "
                    f"paired permutation p = {_fmt(t['paired_permutation']['p_two_sided'], 4)}."
                )
            if "loto_mcnemar_exact" in entry:
                m = entry["loto_mcnemar_exact"]
                lines.append(
                    f"- LOTO exact McNemar: A-only correct {m['a_correct_b_wrong']}, B-only correct "
                    f"{m['a_wrong_b_correct']}, p = {_fmt(m['p_two_sided'], 4)} (smallest possible "
                    f"with this many trajectories: {_fmt(m['min_achievable_p_with_n_units'], 4)})."
                )
            lines += [f"- {entry['power_note']}", ""]
    else:
        lines.append("- None requested (or a model of the pair was not run).")
    lines += [
        "## Caveats",
        "",
        "- The one-per-class splits overlap (each trajectory is tested in several splits); "
        "their spread describes split-to-split variability, not independent replicates.",
        "- The confidence interval pools LOTO, where each trajectory is predicted once. It "
        "still treats trajectories as exchangeable and ignores tuning variability.",
        "- A tied majority vote counts as a wrong trajectory prediction.",
        "- In LOTO the held-out class has one trajectory fewer in training, so the majority "
        "baseline is systematically wrong there (0 correct); use the one-per-class splits "
        "(balanced training) to compare against chance.",
        "",
    ]
    return "\n".join(lines)


def evaluate(
    df: pd.DataFrame,
    specs: list[ModelSpec],
    settings: ProtocolSettings,
    csv_path: Path | None = None,
) -> dict:
    """Run the protocol, the permutation test and the paired comparisons.

    Returns a dict with DataFrames ``per_split``, ``trajectories``, ``null``,
    ``differences`` and the JSON-serialisable ``summary``.
    """
    df = df.reset_index(drop=True)
    X = df[settings.features].to_numpy(dtype=float)
    cache = _Cache(X, settings.preprocessing)
    start = time.perf_counter()
    per_split, trajectories = run_protocol(df, specs, settings, cache=cache)
    observed = {spec.name: protocol_statistic(per_split, spec.name, settings.permutation_scheme)
                for spec in specs}
    permutation, null = permutation_test(df, specs, settings, observed, cache)
    group_labels = df.groupby(settings.group)[settings.label].first()
    classes = sorted(group_labels.unique().tolist())
    rng = np.random.default_rng(settings.random_state + 1)
    comparisons, differences = paired_comparisons(per_split, trajectories, settings.compare, rng,
                                                  int(len(group_labels)))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {
            "csv": None if csv_path is None else str(csv_path),
            "sha256": None if csv_path is None else file_sha256(Path(csv_path)),
            "n_rows": int(len(df)),
            "n_groups": int(len(group_labels)),
            "classes": [str(c) for c in classes],
            "groups_per_class": {str(k): int(v) for k, v in group_labels.value_counts().sort_index().items()},
            "rows_per_class": {str(k): int(v) for k, v in df[settings.label].value_counts().sort_index().items()},
        },
        "protocol": {
            "features": list(settings.features),
            "label": settings.label,
            "group_column": settings.group,
            "outer_schemes": settings.schemes(),
            "n_outer_splits": {scheme: int(len(outer_splits(group_labels.rename(index=str), scheme)))
                               for scheme in settings.schemes()},
            "preprocessing": settings.preprocessing,
            "preprocessing_fitted_on": "training rows of each outer/inner split only",
            "tuning": settings.tuning,
            "inner_cv": ("StratifiedGroupKFold, n_splits = min training trajectories per class"
                         if settings.tuning == "nested" else None),
            "inner_selection_metric": "window-level balanced accuracy" if settings.tuning == "nested" else None,
            "models": {spec.name: {"grid": spec.grid if settings.tuning == "nested" else None,
                                   "fixed": spec.fixed if settings.tuning == "fixed" else None}
                       for spec in specs},
            "trajectory_vote": "majority of window predictions; ties count as wrong",
            "n_permutations": settings.n_permutations,
            "permutation_models": settings.permutation_models,
            "permutation_scheme": settings.permutation_scheme,
            "permutation_unit": "trajectory (labels permuted among groups, class sizes kept)",
            "p_value_formula": "(1 + #{null >= observed}) / (1 + N)",
            "compare": [f"{a}:{b}" for a, b in settings.compare],
            "alpha": settings.alpha,
            "random_state": settings.random_state,
        },
        "chance_level": 1.0 / len(classes),
        "models": summarize_models(per_split, trajectories, len(classes), settings.alpha),
        "permutation": permutation,
        "comparisons": comparisons,
        "wall_time_s": time.perf_counter() - start,
        "environment": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit_learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
            "qiskit": package_version("qiskit"),
            "qiskit_machine_learning": package_version("qiskit-machine-learning"),
        },
    }
    return {"per_split": per_split, "trajectories": trajectories, "null": null,
            "differences": differences, "summary": summary}


def write_outputs(result: dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    result["per_split"].to_csv(run_dir / "per_split.csv", index=False)
    result["trajectories"].to_csv(run_dir / "trajectory_predictions.csv", index=False)
    result["null"].to_csv(run_dir / "permutation_null.csv", index=False)
    result["differences"].to_csv(run_dir / "paired_differences.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(markdown_report(result["summary"]), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_pair(text: str) -> tuple[str, str]:
    parts = text.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"Expected MODEL_A:MODEL_B, got {text!r}")
    return parts[0], parts[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grouped (trajectory-level) evaluation protocol for SVM, QSVM and baselines.")
    parser.add_argument("--csv", type=Path, default=RESULTS_DIR / "datasets" / "md_window_unique_dataset.csv")
    parser.add_argument("--label", default="label_site_id")
    parser.add_argument("--group-column", default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--feature-set", default="md3", choices=sorted(FEATURE_SETS))
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    parser.add_argument("--outer", default="both", choices=[*OUTER_SCHEMES, "both"],
                        help="one_per_class = all one-trajectory-per-class-out splits; loto = "
                             "leave-one-trajectory-out (needed for the Clopper-Pearson interval).")
    parser.add_argument("--preprocessing", default="power_minmax", choices=SCALERS,
                        help="Same preprocessing for every model, fitted on training rows only.")
    parser.add_argument("--tuning", default="nested", choices=["fixed", "nested"])
    parser.add_argument("--param-grid", type=json.loads, default=None,
                        help='JSON overriding grids, e.g. \'{"svm_rbf": {"C": [1, 10], "gamma": ["scale"]}}\'.')
    parser.add_argument("--fixed-params", type=json.loads, default=None,
                        help='JSON overriding fixed values, e.g. \'{"qsvm": {"C": 10}}\'.')
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--qsvm-kernel", default="statevector",
                        help="statevector (qiskit ZZ feature map, as qsvm_baseline.py), product_angle "
                             "(classically computable test kernel) or module:callable.")
    parser.add_argument("--qsvm-qubits", type=int, default=0)
    parser.add_argument("--qsvm-reps", type=int, default=1)
    parser.add_argument("--qsvm-entanglement", default="linear", choices=["linear", "circular", "full"])
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Trajectory-level label permutations; all distinct relabelings are "
                             "enumerated when there are at most this many. 0 disables the test.")
    parser.add_argument("--permutation-models", nargs="+", default=None,
                        help="Models rerun under permutation (default: svm_rbf and qsvm when run, "
                             "otherwise every model except majority).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers (joblib) for the permutation test; -1 = all cores.")
    parser.add_argument("--permutation-scheme", default="one_per_class", choices=OUTER_SCHEMES)
    parser.add_argument("--compare", nargs="*", type=parse_pair, default=None,
                        help="Paired comparisons MODEL_A:MODEL_B (default: qsvm:svm_rbf when both run).")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "grouped_evaluation")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    features = list(args.features) if args.features else FEATURE_SETS[args.feature_set].copy()
    models = list(dict.fromkeys(args.models))
    if args.permutation_scheme == "loto" and args.outer == "one_per_class":
        raise SystemExit("--permutation-scheme loto needs --outer loto or both.")
    if args.permutation_scheme == "one_per_class" and args.outer == "loto":
        args.permutation_scheme = "loto"
    default_permuted = [m for m in ("svm_rbf", "qsvm") if m in models] or [m for m in models if m != "majority"]
    permutation_models = args.permutation_models if args.permutation_models is not None else default_permuted
    unknown = sorted(set(permutation_models) - set(models))
    if unknown:
        raise SystemExit(f"--permutation-models not in --models: {unknown}")
    compare = (args.compare if args.compare is not None
               else ([("qsvm", "svm_rbf")] if {"qsvm", "svm_rbf"} <= set(models) else []))
    kernel = (resolve_kernel(args.qsvm_kernel, len(features), args.qsvm_qubits, args.qsvm_reps,
                             args.qsvm_entanglement) if "qsvm" in models else None)
    specs = build_model_specs(models, args.class_weight, kernel, args.param_grid, args.fixed_params,
                              args.random_state)
    settings = ProtocolSettings(
        features=features, label=args.label, group=args.group_column, outer=args.outer,
        preprocessing=args.preprocessing, tuning=args.tuning, n_permutations=args.n_permutations,
        permutation_models=permutation_models, permutation_scheme=args.permutation_scheme,
        compare=compare, alpha=args.alpha, random_state=args.random_state, n_jobs=args.n_jobs,
    )
    df = load_grouped_dataset(args.csv, features, args.label, args.group_column)
    try:
        result = evaluate(df.drop(columns=["_row_id"]), specs, settings, csv_path=args.csv)
    except InsufficientGroupsError as exc:
        raise SystemExit(f"[error] {exc}") from exc
    result["summary"]["protocol"]["qsvm_kernel"] = args.qsvm_kernel if "qsvm" in models else None
    result["summary"]["protocol"]["qsvm_feature_map"] = (
        {"qubits": args.qsvm_qubits, "reps": args.qsvm_reps, "entanglement": args.qsvm_entanglement}
        if "qsvm" in models and args.qsvm_kernel == "statevector" else None)
    run_name = args.run_name or next_run_name(args.output_dir, "grouped_eval")
    run_dir = args.output_dir / run_name
    write_outputs(result, run_dir)
    print(markdown_report(result["summary"]))
    print(f"RESULT_DIR={run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
