# Model configuration

| File | Use |
| --- | --- |
| `zno_local.yaml` | Default local SVM/QSVM comparison on `md_window_unique_dataset.csv`: seed 42, grouped 30% test split, every window (`split.max_samples: 0`), `md3` features, `power_minmax`, QSVM with one qubit per feature, protocol `grouped_27` |
| `zno_ibm_plan.yaml` | IBM plan/hardware: `ibm.max_samples` = 12 rows spread over as many trajectories as possible, 25% test split, protocol `holdout`; a matched local QSVM runs on the same 12 rows, the other local runs use every row |

Commands: [docs/models.md](../../docs/models.md). Hardware results are a feasibility
demonstration, not an accuracy estimate.

## Grouped validation

Every dataset row needs a `group` column (one MD trajectory, `material/site/seed_<n>`).
The SVM, QSVM, IBM and benchmark scripts (`--group-column`, default `group`) keep each
group entirely in train or test; CV uses `StratifiedGroupKFold` on the training groups;
scalers are fitted on training rows only. Split files carry the `group` column and are
checked against the dataset.

- Train/test split: at least 2 trajectories per site.
- CV (`cv_folds` > 0): at least 3 trajectories per site; folds are reduced to the
  training trajectories per site. `cv_folds: 0` disables it.
- With fewer trajectories the run stops instead of splitting windows of one trajectory.

Feature sets: `md3` (C height mean and std, O-C-O angle) and `md_mic5` (minimum-image
C-slab and O-metal distances, C height, C-O bond, O-C-O angle). `all12`, `static6`,
`dynamic6`, `dynamic4` and `tddft3` came from the removed cartesian-product dataset and
were dropped.

## Evaluation protocol (`evaluation.protocol`)

The statistical unit is the trajectory. With 3 sites × 3 seeds there are 9 units, so a
single holdout of 3 trajectories is uninformative.

- `holdout`: single grouped train/test split per seed (`run_benchmarks`, timings,
  engineering tables). Always used by `ibm` and `ibm-plan`.
- `grouped_27`: additionally runs `scripts.models.evaluate_grouped` for `svm`, `qsvm`
  and `local`:
  - every split leaving one trajectory per site out (27 for 3 × 3; in general the
    product of trajectories per site), plus leave-one-trajectory-out (LOTO);
  - the same preprocessing for every model (`power_minmax`: Yeo-Johnson, then MinMax to
    [0, π]), fitted on training rows only;
  - hyperparameters by nested `StratifiedGroupKFold` inside the training trajectories
    (`tuning: nested`, needs >= 3 trajectories per site) or fixed (`tuning: fixed`); the
    outer test fold is never used for selection;
  - balanced accuracy per window and per trajectory (majority vote); exact
    Clopper-Pearson interval on the LOTO correct count (9/9 gives [0.664, 1]); chance
    level 1/3;
  - trajectory-level label permutation test rerunning the whole protocol; the 280
    partitions of 9 trajectories into 3 classes are enumerated exactly,
    p = (1 + #{null >= observed}) / (1 + N); `n_jobs: -1` parallelizes it;
  - baselines (majority, nearest centroid, logistic regression, linear SVM, RBF SVM) on
    the same splits, and a paired QSVM vs SVM comparison reported with its power.

Outputs: `<run>/evaluation/grouped_eval/` (`per_split.csv`,
`trajectory_predictions.csv`, `permutation_null.csv`, `paired_differences.csv`,
`summary.json`, `report.md`). Override without editing YAML: `--protocol holdout` or
`--protocol grouped_27`. Direct call:

```bash
python -m scripts.models.evaluate_grouped \
  --csv data/generated/<run-id>/multitech/md_window_unique_dataset.csv \
  --feature-set md3 --n-jobs -1 --output-dir results/models/grouped_evaluation
```

`--qsvm-kernel`: `statevector` (default, needs qiskit-machine-learning),
`product_angle` (classically computable test kernel) or `module:callable` (any kernel
K(A, B=None)).

## Hyperparameters

- SVM and QSVM share the preprocessing (`power_minmax`) and the CV folds
  (`cv_folds: 5`).
- The holdout scripts report grouped CV but keep C and gamma fixed.
- `run_benchmarks --full-grid` reports every configuration and selects none by test
  score; `--select-by-cv` adds `cv_selection.csv` (choice by training CV only).

## Label shuffling

`--shuffle-label` of the SVM and QSVM scripts permutes labels among the training
trajectories only (whole groups, class sizes kept); test labels stay intact. It is a
single sanity draw that should score near chance; the significance test is the
permutation test of `evaluate_grouped`.

## IBM kernel diagonal (`ibm.kernel_diagonal`)

On hardware K(x, x) < 1.

| Value | Effect |
| --- | --- |
| `ideal` (default) | Training diagonal fixed to 1; test entries are hardware estimates |
| `measured` | Training diagonal measured |
| `measured_normalized` | Train and test diagonals measured; both kernels normalized by sqrt(K(x,x) K(y,y)) |

`measured` and `measured_normalized` add circuits. IBM credentials are never stored in
these files.
