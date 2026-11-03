# Combined workflow

`run_all.sh` chains the stage launchers for one run ID: science analysis, datasets,
local models, model analysis, optionally preceded by new simulations. It runs in
`QML_ZNO_MODEL_ENV`; each stage launcher uses its own environment variable
([environments/README.md](../environments/README.md)).

```bash
./run_all.sh --run-id local_reproduction_01 \
  --source-results results/simulations/<run-id> \
  --dataset-config configs/datasets/zno_existing_results.yaml \
  --model-config configs/models/zno_local.yaml \
  --analysis-config configs/analysis/zno_current.yaml \
  --model-dataset multitech/md_window_unique_dataset.csv \
  --dry-run
```

Replace `--dry-run` (the default) with `--execute` to run the plan. The dry run does not
validate inputs that later stages generate.

| Stage | Command | Output |
| --- | --- | --- |
| Science analysis | `run_analysis.sh science` | `results/analyses/<ID>_science/` |
| Datasets | `run_datasets.sh all`, ranking from the science analysis | `data/generated/<ID>/` |
| Local SVM and QSVM | `run_models.sh local` on `--model-dataset` | `results/models/<ID>/` |
| Model analysis | `run_analysis.sh models` | `results/analyses/<ID>_models/` |

- `--run-id` and the dataset, model and analysis configs are required. The plan is
  refused when any stage output of that run ID already exists.
- `--model-dataset` is relative to `data/generated/<ID>/`.
- The workflow stops at the first failed stage, or when a selected config file changes
  during execution. Commands, configuration hashes, exit
  codes and logs go to `results/workflows/<ID>/`; stage manifests record input and
  output hashes.
- Output roots are taken from the selected YAML files.
- IBM submission is never included ([models.md](models.md)).

## New simulations

Replace `--source-results results/simulations/<run-id>` with:

```text
--simulate --simulation-config configs/simulations/zno_production.yaml
```

This first runs `run_simulations.sh all` into `results/simulations/<ID>/`
([simulations.md](simulations.md)).

## Reproducing a specific result

Use the configuration, split and metadata of that result; `zno_local.yaml` is one
profile, not every manuscript experiment ([configs/models/README.md](../configs/models/README.md)).
Archive the source commit, configurations, environment versions and inputs with a
published result. `requirements.txt` is not a lockfile.

Rebuilding tables and figures from saved outputs repeats neither the simulations nor
the hardware measurements. A new simulation campaign needs its own convergence checks;
a new hardware run differs by sampling noise and device calibration.
