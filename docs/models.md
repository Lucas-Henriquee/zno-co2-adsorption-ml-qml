# Models

`run_models.sh` runs classical SVM, local QSVM, IBM planning and IBM submission on one
dataset into `results/models/<run-id>/`. Configuration and evaluation protocol:
[configs/models/README.md](../configs/models/README.md).

```bash
DATASET=data/generated/<run-id>/multitech/md_window_unique_dataset.csv
./run_models.sh local --dataset $DATASET --dry-run
./run_models.sh local --dataset $DATASET --run-id zno_md3_local_01
./run_models.sh svm --dataset $DATASET --run-id zno_md3_svm_01
./run_models.sh qsvm --dataset $DATASET --run-id zno_md3_qsvm_01
```

| Method | Run |
| --- | --- |
| `svm`, `qsvm` | One model |
| `local` (alias `all`) | Matched SVM and local QSVM on the same rows and split |
| `ibm-plan` | Workload and command of an IBM run; no account, network or submission |
| `ibm` | Real hardware submission |

| Option | Effect |
| --- | --- |
| `--dataset FILE` | Override `input.dataset` (default `data/generated/zno_ic_final_datasets/multitech/md_window_unique_dataset.csv`) |
| `--config PATH` | Other YAML (default `configs/models/zno_local.yaml`) |
| `--protocol holdout\|grouped_27` | Override `evaluation.protocol` for `svm`, `qsvm`, `local` |
| `--backend NAME` | IBM backend |
| `--submit-ibm` | Required for `ibm`; refused with any other method |
| `--run-id NAME` | Run directory name; default next `<method>_NN` |
| `--dry-run` | Validate the dataset and print the commands |

With `evaluation.protocol: grouped_27` (default in `zno_local.yaml`) `svm`, `qsvm` and
`local` also run the grouped trajectory-level evaluation.

## IBM

```bash
./run_models.sh ibm-plan --config configs/models/zno_ibm_plan.yaml \
  --dataset $DATASET --run-id zno_ibm_plan_01
./run_models.sh ibm --config configs/models/zno_ibm_plan.yaml \
  --dataset $DATASET --backend BACKEND_NAME --submit-ibm --run-id zno_ibm_hw_01
```

- `ibm` needs `--submit-ibm` and a backend (`--backend` or `ibm.backend`).
- No tokens, API keys or instance credentials in YAML; use a saved Qiskit Runtime
  account.
- IBM execution is never part of `run_all.sh`.

## Scope

The reference MD setup gives 3 trajectories per site, 9 independent units. Results
measure within-campaign site classification; do not report them as generalization to
unseen materials or trajectories, or as quantum advantage.

Each run records its dataset hash, configuration, commands, versions and output hashes.
