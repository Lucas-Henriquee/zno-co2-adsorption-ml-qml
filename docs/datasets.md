# Datasets

`run_datasets.sh` builds datasets from one simulation run into
`data/generated/<run-id>/`. Configuration: `configs/datasets/zno_existing_results.yaml`;
inputs, outputs and block selection: [configs/datasets/README.md](../configs/datasets/README.md).

```bash
RUN=results/simulations/<run-id>
./run_datasets.sh tddft multitech --source-results $RUN --dry-run
./run_datasets.sh tddft multitech --source-results $RUN --run-id zno_descriptors_01
./run_datasets.sh qml_features --source-results $RUN \
  --ranking results/analyses/<analysis-run>/science/screening/ranked_candidates.csv \
  --run-id zno_qml_features_01
./run_datasets.sh all --source-results $RUN \
  --ranking results/analyses/<analysis-run>/science/screening/ranked_candidates.csv
```

| Option | Effect |
| --- | --- |
| `--source-results DIR` | Simulation run (default `input.results_dir`, `results/simulations/zno_ic_final`) |
| `--ranking FILE` | Ranked candidate table for `qml_features` (default `<run>/analysis/ranked_candidates.csv`) |
| `--config PATH` | Other YAML |
| `--run-id NAME` | Run directory name; default next `dataset_NN`; never reused |
| `--dry-run` | Validate inputs and print the commands |
| `--list-methods` | Print the methods |

| Method | Output |
| --- | --- |
| `tddft` | Summary- and transition-level TDDFT reference tables (isolated CO₂, no features) |
| `multitech` | DFT summary (with CO₂RR, MD and TDDFT reference columns), MD-window and MD-trajectory datasets |
| `qml_features` | Ranked DFT table normalized for QML; needs the ranking written by `./run_analysis.sh science` |

- `all` includes `qml_features`, so it fails without a ranking file.
- MD input: `md/<material>/<site>/md_production_frames.csv` (all seeds,
  `phase == production` rows only).
- The MD-window dataset has one row per block of production frames and a `group`
  column (`material/site/seed_<n>`). Windows of one trajectory are correlated; the model
  configurations split and validate by trajectory.
- The manifest records the SHA-256 of every input, the commands, package versions and
  method status. Simulation outputs are read only.

Column definitions: [datasets_reference.md](datasets_reference.md) (out of date; the
reference is `scripts/datasets/prepare_multitech_dataset.py`).
