# Model Configuration

- `zno_local.yaml` is the default local SVM/QSVM comparison on the grouped
  MD-window dataset (`md_window_unique_dataset.csv`) and a shared grouped split.
- `zno_ibm_plan.yaml` limits the matched local/IBM study to 12 samples, spread
  over as many trajectories as possible, so its circuit workload can be reviewed
  before submission.

## Grouped validation

Every dataset row must carry a `group` column (one independent MD trajectory,
`material/site/seed_<n>`). The benchmark (`--group-column`, default `group`)
and the SVM, QSVM and IBM scripts keep each group entirely in train or in test,
and cross-validation uses `StratifiedGroupKFold` on the training groups.
Preprocessing (scalers) is fitted on the training rows only.

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
