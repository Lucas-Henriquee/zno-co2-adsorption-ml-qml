# Model Configuration

- `zno_local.yaml` is the default local SVM/QSVM comparison on the grouped
  MD-window dataset (`md_window_unique_dataset.csv`) and a shared grouped split.
- `zno_ibm_plan.yaml` limits the IBM hardware runs to `ibm.max_samples` (12)
  samples, spread over as many trajectories as possible, so the circuit workload
  can be reviewed before submission. A matched local QSVM is run on the same 12
  rows for the local/IBM kernel comparison; the other local runs use every row
  (`split.max_samples: 0`). Hardware results are a feasibility demonstration.

## Evaluation protocol (`evaluation.protocol`)

The statistical unit is the MD trajectory (`group`), not the window. With
3 sites x 3 seeds there are only 9 independent units, so a single holdout of
3 trajectories is uninformative.

- `holdout`: only the single grouped train/test split per seed
  (`run_benchmarks`, timings and engineering tables).
- `grouped_27` (default in `zno_local.yaml`): additionally runs
  `scripts.models.evaluate_grouped` for the `svm`, `qsvm` and `local` methods:
  - every split that leaves one trajectory per site out (27 for 3 x 3; in
    general the product of trajectories per site) and leave-one-trajectory-out;
  - the same preprocessing for every model (`power_minmax` = Yeo-Johnson then
    MinMax to [0, pi]), fitted on training rows only;
  - hyperparameters by nested `StratifiedGroupKFold` inside the training
    trajectories (`tuning: nested`) or fixed (`tuning: fixed`); the outer test
    fold is never used for selection;
  - balanced accuracy per window and per trajectory (majority vote), exact
    Clopper-Pearson interval on the LOTO correct count (9/9 gives [0.664, 1]),
    chance level 1/3;
  - a label permutation test at the trajectory level that reruns the whole
    protocol; the 280 partitions of 9 trajectories into 3 classes are
    enumerated exactly, p = (1 + #{null >= observed}) / (1 + N);
  - baselines (majority, nearest centroid, logistic regression, linear SVM,
    RBF SVM) on the same splits and a paired QSVM vs SVM comparison, reported
    with its (low) power.
  Outputs go to `<run>/evaluation/grouped_eval/` (`per_split.csv`,
  `trajectory_predictions.csv`, `permutation_null.csv`, `paired_differences.csv`,
  `summary.json`, `report.md`). Nested tuning needs at least 3 trajectories per
  site. `n_jobs: -1` parallelises the permutation test.

Override the protocol without editing YAML with `--protocol holdout` or
`--protocol grouped_27`. The protocol can also be run directly:

```bash
python -m scripts.models.evaluate_grouped \
  --csv data/generated/<run>/multitech/md_window_unique_dataset.csv \
  --feature-set md3 --n-jobs -1 --output-dir results/models/grouped_evaluation
```

`--qsvm-kernel statevector` (default) needs qiskit-machine-learning;
`product_angle` is a classically computable fidelity kernel for testing, and
`module:callable` plugs in any kernel K(A, B=None).

## Hyperparameters and grids

SVM and QSVM share the preprocessing (`power_minmax`) and the CV folds
(`cv_folds: 5`, reduced to the training trajectories per site). The holdout
scripts report grouped CV but keep C and gamma fixed. `run_benchmarks
--full-grid` reports every configuration and selects none by test score; add
`--select-by-cv` to write `cv_selection.csv` (choice by training-set CV only),
or compare candidates with `evaluate_grouped`.

## Label shuffling

`--shuffle-label` of the SVM and QSVM scripts permutes labels among the
training trajectories only (whole groups, class sizes kept) and leaves the test
labels intact. It is a single sanity draw that should score near chance; the
significance test is the permutation test of `evaluate_grouped`.

## IBM kernel diagonal

On hardware K(x, x) < 1. By default (`ibm.kernel_diagonal: ideal`) the training
diagonal is fixed to 1 while test entries are noisy hardware estimates.
`measured` measures the training diagonal; `measured_normalized` measures train
and test diagonals and normalises both kernels by sqrt(K(x,x) K(y,y)).

## Grouped validation

Every dataset row must carry a `group` column (one independent MD trajectory,
`material/site/seed_<n>`). The benchmark (`--group-column`, default `group`)
and the SVM, QSVM and IBM scripts keep each group entirely in train or in test,
and cross-validation uses `StratifiedGroupKFold` on the training groups.
Preprocessing (scalers) is fitted on the training rows only. Split files must
carry the `group` column written with them; it is checked against the dataset.

- A train/test split needs at least 2 trajectories per site.
- Cross-validation (`cv_folds` > 0) needs at least 2 training trajectories per
  site, i.e. at least 3 trajectories per site; the number of folds is reduced
  to the training trajectories per site. Set `cv_folds: 0` to disable it.
- With fewer trajectories the run stops with a message instead of splitting
  windows of one trajectory between train and test.

The former feature sets `all12`, `static6`, `dynamic6`, `dynamic4` and `tddft3`
came from the removed cartesian-product dataset and were dropped. Available
sets: `md3` (C height mean and std, O-C-O angle) and `md_mic5` (minimum-image C-
slab and O-metal distances, C height, C-O bond, O-C-O angle).

IBM credentials are never stored in these files. A real hardware execution
requires the `ibm` method, the explicit `--submit-ibm` flag, and a backend name.
An `ibm-plan` only calculates and records the command and expected workload; it
does not contact IBM Quantum.

The dataset can be replaced without editing YAML:

```bash
./run_models.sh local \
  --dataset data/generated/another_run/multitech/md_window_unique_dataset.csv \
  --run-id another_local_comparison
```
