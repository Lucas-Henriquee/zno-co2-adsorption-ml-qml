# ZnO–CO₂ adsorption: simulations, ML and QML

CO₂ adsorption at the top-metal, top-oxygen and bridge sites of a 2×1 ZnO(10-10) slab,
computed with GPAW (DFT relaxations, LrTDDFT of the isolated CO₂, CO₂RR free-energy
pathway, Langevin AIMD, NEB). Classical SVM and quantum-kernel SVM models classify the
starting site from AIMD-window descriptors; a small subsample can be run on IBM Quantum
hardware.

## Repository

| Path | Content |
| --- | --- |
| `run_*.sh` | Root launchers (simulations, datasets, models, analysis, combined workflow) |
| `scripts/workflows/` | Workflow modules behind the launchers |
| `configs/` | YAML settings per stage |
| `environments/` | Conda environment files and setup guide |
| `notebooks/` | Simulation and analysis notebooks |
| `data/`, `results/` | Inputs, generated datasets, run directories |
| `docs/` | Stage guides ([index](docs/README.md)) |
| `results/_invalid_ic_v1/`, `scripts/legacy/` | First campaign, invalid, kept for provenance |

## Install

See [environments/README.md](environments/README.md).

## Run

The Python workflow is the primary interface: the root launchers, or
`python -m scripts.workflows.<stage>` with the same arguments inside an environment.
The notebooks are a secondary, equivalent route:
`notebooks/zno-co2-lucas.ipynb` runs the simulations and
`notebooks/zno-co2-analysis.ipynb` the analyses of a run
([notebooks/README.md](notebooks/README.md)).

Each run writes to its own run directory, named by `--run-id` (default: next free
`<prefix>_NN`), and keeps a `config.yaml` snapshot and a `run_manifest.json`.

```bash
# Simulations (settings: configs/simulations/zno_production.yaml)
./run_simulations.sh all --run-id zno_run_01 --dry-run
./run_simulations.sh all --run-id zno_run_01
./run_simulations.sh all --run-id zno_run_01 --resume

# Status dashboard (read-only), inside the simulation environment
python -m scripts.workflows.run_status zno_run_01

# Paper figures and tables of a finished (or running) run
./run_analysis.sh paper --scientific-results results/simulations/zno_run_01 \
  --run-id zno_run_01_paper
```

`./run_<stage>.sh --help` prints the launcher usage without conda.

`--resume` skips completed methods and refuses failed or interrupted ones; see
[docs/simulations.md](docs/simulations.md) for resume rules and all options.
Datasets, models and the combined workflow:
[datasets](docs/datasets.md), [models](docs/models.md),
[reproduction](docs/reproduction.md).

### Selecting an existing environment

The launchers run in `qml-zno-simulation` / `qml-zno-modeling` by default. Any other
environment (name, or prefix path containing `/`) is selected with:

| Variable | Launchers |
| --- | --- |
| `QML_ZNO_SIM_ENV` | `run_simulations.sh`, `notebooks/run_zno_notebook.sh` |
| `QML_ZNO_ANALYSIS_ENV` | `run_analysis.sh` (falls back to `QML_ZNO_MODEL_ENV`), analysis notebook runner (falls back to `QML_ZNO_SIM_ENV`) |
| `QML_ZNO_MODEL_ENV` | `run_datasets.sh`, `run_models.sh`, `run_all.sh` |
| `QML_ZNO_CONDA` | Package manager: conda, mamba or micromamba (default: first found in `PATH`) |

Example, lab PC with a single environment `gpaw-dft`:

```bash
QML_ZNO_SIM_ENV=gpaw-dft ./run_simulations.sh all --run-id zno_run_01 --dry-run
QML_ZNO_ANALYSIS_ENV=gpaw-dft ./run_analysis.sh paper \
  --scientific-results results/simulations/zno_run_01 --dry-run
```

## Tests

```bash
python -m pytest tests -q
```

Run inside any project environment, or a venv with `requirements-dev.txt`. GPAW and
Qiskit tests skip or use stubs. The tests check code and workflows, not scientific
convergence ([docs/testing.md](docs/testing.md)).

## Documentation

- [Environments](environments/README.md)
- [Simulations](docs/simulations.md)
- [Datasets](docs/datasets.md)
- [Models and IBM](docs/models.md)
- [Analysis](docs/analysis.md), [figures](docs/figures.md)
- [Combined workflow](docs/reproduction.md)
- [Notebooks](notebooks/README.md)
- [Manual testing](docs/testing.md)
- [Repository layout](docs/repository_layout.md)
- [Invalid first campaign](results/_invalid_ic_v1/README.md)
