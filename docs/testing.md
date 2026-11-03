# Manual testing

Checks from the repository root, in increasing cost. Every executed step needs a new
run ID; existing run directories are never overwritten. Environment selection:
[environments/README.md](../environments/README.md).

## 1. Environments and unit tests

```bash
./scripts/environment/check_environment.sh simulation --mpi
./scripts/environment/check_environment.sh modeling
python -m pytest tests -q          # inside an environment with pytest
```

The checks must end without `[error]`; the tests must pass. The MPI check initializes
GPAW in dry-run mode only.

## 2. Validate commands and inputs

```bash
RUN=results/simulations/<run-id>
DATASET=data/generated/<dataset-run-id>/multitech/md_window_unique_dataset.csv
./run_simulations.sh all --dry-run
./run_analysis.sh science --scientific-results $RUN --dry-run
./run_datasets.sh tddft multitech --source-results $RUN --dry-run
./run_models.sh local --dataset $DATASET --dry-run
./run_models.sh ibm-plan --config configs/models/zno_ibm_plan.yaml --dataset $DATASET --dry-run
```

Each command must end with `[dry-run] Validation completed`. The downstream commands
need a finished simulation run and, for the models, a dataset run. `qml_features` (part
of `run_datasets.sh all`) also needs `--ranking` from an executed science analysis.
Nothing is simulated, trained or submitted.

## 3. Local workflow on a saved simulation run

```bash
./run_all.sh --run-id manual_local_01 \
  --source-results $RUN \
  --dataset-config configs/datasets/zno_existing_results.yaml \
  --model-config configs/models/zno_local.yaml \
  --analysis-config configs/analysis/zno_current.yaml \
  --model-dataset multitech/md_window_unique_dataset.csv \
  --dry-run                        # then the same command with --execute
```

Produces science tables, datasets, SVM/QSVM results and model figures.
`results/workflows/manual_local_01/run_manifest.json` must report `completed`; on
`failed`, read the stage logs in the same directory.

## 4. IBM plan

```bash
./run_models.sh ibm-plan --config configs/models/zno_ibm_plan.yaml \
  --dataset data/generated/manual_local_01/multitech/md_window_unique_dataset.csv \
  --run-id manual_ibm_plan_01
```

Needs no account and does not validate hardware access. Real submission:
[models.md](models.md).

## 5. Simulations

Runs every method at the production settings; MD and NEB can run for days.

```bash
./run_simulations.sh all --run-id manual_simulation_01
```

Check `results/simulations/manual_simulation_01/run_manifest.json` and the method logs.
A completed process is not proof of SCF or optimizer convergence; review the
convergence messages, MD outputs and NEB paths. To test downstream processing on this
run, repeat section 3 with `--source-results results/simulations/manual_simulation_01`
and a new run ID.

These steps test the configured workflow, not the reproduction of every published
comparison ([reproduction.md](reproduction.md)).
