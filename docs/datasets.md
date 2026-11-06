# Datasets

`run_datasets.sh` builds datasets from one simulation run into
`data/generated/<run-id>/`. Configuration: `configs/datasets/zno_existing_results.yaml`;
inputs, outputs and block selection: [configs/datasets/README.md](../configs/datasets/README.md).

```bash
RUN=results/simulations/<run-id>
./run_datasets.sh tddft multitech --source-results $RUN --dry-run
./run_datasets.sh tddft multitech --source-results $RUN --run-id zno_descriptors_01
./run_datasets.sh all --source-results $RUN --run-id zno_descriptors_02
```

| Option | Effect |
| --- | --- |
| `--source-results DIR` | Simulation run (default `input.results_dir`, `results/simulations/zno_ic_final`) |
| `--config PATH` | Other YAML |
| `--run-id NAME` | Run directory name; default next `dataset_NN`; never reused |
| `--dry-run` | Validate inputs and print the commands |
| `--list-methods` | Print the methods |

| Method | Output |
| --- | --- |
| `tddft` | Summary- and transition-level TDDFT reference tables (isolated CO₂, no features) |
| `multitech` | DFT summary (with CO₂RR, MD and TDDFT reference columns), MD-window and MD-trajectory datasets |

- `all` = `tddft multitech`. The former `qml_features` method (one row per site from the
  screening ranking: 3 rows, one per class, which cannot be split into training and test
  data) was retired and is rejected; the script is in `scripts/legacy/datasets/`.
- MD input: `md/<material>/<site>/md_production_frames.csv` (all seeds,
  `phase == production` rows only).
- The MD-window dataset has one row per block of production frames and a `group`
  column (`material/site/seed_<n>`). Windows of one trajectory are correlated; the model
  configurations split and validate by trajectory.
- Label: the site where the trajectory started (`label_site`). The physisorbed CO₂ moves
  away from it during the run (Table 2: 13–34 % of the production frames adsorbed at the
  starting site), so the models test whether the descriptors still identify the starting
  site, not where the molecule is. The per-window site diagnostics
  (`window_start_site_fraction`, `window_nearest_site`, `window_desorbed_fraction`) are
  kept for discussion; no window is filtered by them.
- Run `zno_ic_final`: 76 production frames per trajectory (20 fs apart), largest
  statistical inefficiency g = 27.7 frames, so the common block is 56 frames (1120 fs)
  and every trajectory gives one block (frames 0–1100 fs). The MD-window dataset then
  has 9 rows, one per trajectory, and drops the last 20 frames; `md_trajectory_dataset.csv`
  has the same 9 trajectories with all 76 frames.
- The summary's CO₂RR columns are the run summary (variant a), with
  `co2rr_pathway_validity`; the limiting potential is blank when the validity is not
  `ok` (top_oxygen: COOH*/CO* desorbed). The paper uses variant c (Table 2).
- The manifest records the SHA-256 of every input, the commands, package versions and
  method status. Simulation outputs are read only.

Column definitions: [datasets_reference.md](datasets_reference.md).
