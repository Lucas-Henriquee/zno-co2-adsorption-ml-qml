"""Grouped train/test splitting shared by the SVM, QSVM and IBM QSVM scripts.

Rows of the MD-window datasets are blocks of the same trajectory and are
therefore correlated. Every row carries a ``group`` id (material/site/seed); a
group is one independent trajectory. All splits below keep each group entirely
on one side (train or test, or one CV fold), so windows of the same trajectory
never appear in both. When there are too few groups for such a split, the
functions raise ``InsufficientGroupsError`` instead of falling back to a
row-level split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_GROUP_COLUMN = "group"
# Minimum independent trajectories per class for a holdout split: one for
# training and one for testing.
MIN_GROUPS_PER_CLASS_HOLDOUT = 2


class InsufficientGroupsError(ValueError):
    """Raised when there are not enough independent groups for a grouped split."""


def _groups_per_class(df: pd.DataFrame, label: str, group: str) -> pd.Series:
    return df.groupby(label)[group].nunique().sort_index()


def _format_counts(counts: pd.Series) -> str:
    return ", ".join(f"{cls}: {int(value)}" for cls, value in counts.items())


def insufficient_groups_message(counts: pd.Series, required: int, purpose: str) -> str:
    return (
        f"Grouped {purpose} needs at least {required} independent trajectories (groups) "
        f"per class, but found {{{_format_counts(counts)}}}. Windows of one MD trajectory "
        "are correlated and must stay on the same side of the split, so rows cannot be "
        f"split instead. Run at least {required} independent MD trajectories (seeds) per "
        "site, e.g. md/<material>/<site>/seed_<n>/, and regenerate the dataset."
    )


def require_group_column(df: pd.DataFrame, group: str) -> None:
    if group not in df.columns:
        raise ValueError(
            f"Dataset has no '{group}' column. Regenerate it with "
            "scripts.datasets.prepare_multitech_dataset, which writes one group id per "
            "independent MD trajectory (material/site/seed). Row-level splits of "
            "correlated MD windows leak information between train and test."
        )
    if df[group].isna().any():
        raise ValueError(f"Column '{group}' contains missing values.")


def check_groups_have_single_label(df: pd.DataFrame, label: str, group: str) -> None:
    labels_per_group = df.groupby(group)[label].nunique()
    mixed = labels_per_group[labels_per_group > 1]
    if len(mixed):
        raise ValueError(
            f"Groups with more than one '{label}' value: {mixed.index[:10].tolist()}. "
            "Each trajectory must keep its starting-site label."
        )


def load_grouped_dataset(csv_path: Path, features: list[str], label: str, group: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["_row_id"] = np.arange(len(df), dtype=int)
    required = list(features) + [label]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    require_group_column(df, group)
    df = df.dropna(subset=required).copy()
    if df.empty or df[label].nunique() < 2:
        raise ValueError("Dataset must contain at least two classes after dropna.")
    df[group] = df[group].astype(str)
    check_groups_have_single_label(df, label, group)
    return df


def group_balanced_subsample(
    df: pd.DataFrame,
    label: str,
    group: str,
    max_samples: int,
    random_state: int,
) -> pd.DataFrame:
    """Limit the number of rows while spreading them over as many groups as possible.

    Within each class, groups are shuffled and rows are taken round-robin across
    them, so a small ``max_samples`` does not collapse a class to one trajectory.
    """
    if max_samples <= 0 or len(df) <= max_samples:
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    classes = sorted(df[label].drop_duplicates().tolist())
    n_classes = len(classes)
    if max_samples < 2 * n_classes:
        raise ValueError(f"--max-samples must be at least {2 * n_classes} for {n_classes} classes.")

    rng = np.random.default_rng(random_state)
    base, remainder = divmod(max_samples, n_classes)
    class_order = list(classes)
    rng.shuffle(class_order)

    chunks = []
    for i, cls in enumerate(class_order):
        target = base + (1 if i < remainder else 0)
        subset = df[df[label] == cls].sample(frac=1.0, random_state=random_state + i)
        group_names = subset[group].drop_duplicates().tolist()
        rng.shuffle(group_names)
        group_rank = {name: rank for rank, name in enumerate(group_names)}
        ordered = subset.assign(
            _within=subset.groupby(group).cumcount(),
            _group_rank=subset[group].map(group_rank),
        ).sort_values(["_within", "_group_rank"], kind="mergesort")
        chunks.append(ordered.head(target).drop(columns=["_within", "_group_rank"]))

    sampled = pd.concat(chunks, axis=0)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def grouped_holdout_split(
    df: pd.DataFrame,
    label: str,
    group: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified group holdout: whole groups go to test, at least one per class."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    counts = _groups_per_class(df, label, group)
    if len(counts) < 2:
        raise ValueError("Need at least two classes for a train/test split.")
    if counts.min() < MIN_GROUPS_PER_CLASS_HOLDOUT:
        raise InsufficientGroupsError(
            insufficient_groups_message(counts, MIN_GROUPS_PER_CLASS_HOLDOUT, "train/test split")
        )

    rng = np.random.default_rng(random_state)
    test_groups: list[str] = []
    for cls in counts.index:
        names = sorted(df.loc[df[label] == cls, group].drop_duplicates().tolist())
        rng.shuffle(names)
        n_test = int(round(test_size * len(names)))
        n_test = min(max(n_test, 1), len(names) - 1)
        test_groups.extend(names[:n_test])

    is_test = df[group].isin(test_groups)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


def assert_no_group_leak(train_df: pd.DataFrame, test_df: pd.DataFrame, group: str) -> None:
    shared = sorted(set(train_df[group]) & set(test_df[group]))
    if shared:
        raise ValueError(
            f"Groups present in both train and test: {shared[:10]}. Windows of the same "
            "trajectory must not be split between train and test; regenerate the split file."
        )


def read_split_file(
    split_path: Path,
    df: pd.DataFrame,
    label: str,
    group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = pd.read_csv(split_path)
    if not {"row_id", "split"}.issubset(split.columns):
        raise ValueError("split-file must contain columns: row_id, split")
    if split["row_id"].duplicated().any():
        raise ValueError("split-file contains duplicated row_id values")
    invalid_splits = sorted(set(split["split"]) - {"train", "test"})
    if invalid_splits:
        raise ValueError(f"split-file contains invalid split values: {invalid_splits}")
    train_ids = split.loc[split["split"] == "train", "row_id"].astype(int).tolist()
    test_ids = split.loc[split["split"] == "test", "row_id"].astype(int).tolist()
    if not train_ids or not test_ids:
        raise ValueError("split-file must contain at least one train and one test row")
    indexed = df.set_index("_row_id", drop=False)
    missing_ids = sorted((set(train_ids) | set(test_ids)) - set(indexed.index))
    if missing_ids:
        raise ValueError(f"Split contains row_ids unavailable after dropna: {missing_ids[:10]}")
    if "true_label" in split.columns:
        expected = split.set_index("row_id")["true_label"]
        actual = indexed.loc[expected.index, label]
        mismatched = expected.index[expected.astype(str) != actual.astype(str)]
        if len(mismatched):
            raise ValueError(
                "split-file true_label does not match the dataset for row_ids: "
                f"{mismatched[:10].tolist()}"
            )
    train_df, test_df = indexed.loc[train_ids].copy(), indexed.loc[test_ids].copy()
    assert_no_group_leak(train_df, test_df, group)
    return train_df, test_df


def grouped_train_test_split(
    df: pd.DataFrame,
    label: str,
    group: str,
    test_size: float,
    max_samples: int,
    random_state: int,
    split_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_file is not None:
        return read_split_file(split_file, df, label, group)
    sampled = group_balanced_subsample(df, label, group, max_samples, random_state)
    train_df, test_df = grouped_holdout_split(sampled, label, group, test_size, random_state)
    assert_no_group_leak(train_df, test_df, group)
    return train_df, test_df


def grouped_cv_splitter(
    train_df: pd.DataFrame,
    label: str,
    group: str,
    requested_folds: int,
    random_state: int,
) -> tuple[StratifiedGroupKFold | None, int]:
    """Return a StratifiedGroupKFold over the training groups, or (None, 0) when disabled.

    The number of folds is reduced to the smallest number of training groups per
    class. If that is below two, cross-validation is impossible without splitting a
    trajectory, and an InsufficientGroupsError explains how many are needed.
    """
    if requested_folds <= 0:
        return None, 0
    counts = _groups_per_class(train_df, label, group)
    folds = min(int(requested_folds), int(counts.min()))
    if folds < 2:
        raise InsufficientGroupsError(
            insufficient_groups_message(counts, 2, "cross-validation inside the training set")
            + " Training groups are those left after the test holdout, so the dataset needs "
            "at least 3 trajectories per site for holdout plus 2-fold CV; or disable CV with --cv 0."
        )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return splitter, folds


def split_table(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str, group: str) -> pd.DataFrame:
    return pd.concat([
        pd.DataFrame({
            "row_id": train_df["_row_id"].astype(int).to_numpy(),
            "split": "train",
            "true_label": train_df[label].to_numpy(),
            "group": train_df[group].to_numpy(),
        }),
        pd.DataFrame({
            "row_id": test_df["_row_id"].astype(int).to_numpy(),
            "split": "test",
            "true_label": test_df[label].to_numpy(),
            "group": test_df[group].to_numpy(),
        }),
    ], ignore_index=True)


def group_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str, group: str) -> dict:
    return {
        "group_column": group,
        "train_groups": int(train_df[group].nunique()),
        "test_groups": int(test_df[group].nunique()),
        "train_groups_per_class": {
            str(k): int(v) for k, v in _groups_per_class(train_df, label, group).items()
        },
        "test_groups_per_class": {
            str(k): int(v) for k, v in _groups_per_class(test_df, label, group).items()
        },
    }
