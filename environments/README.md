# Environments

Two conda-forge environments keep the GPAW/MPI stack apart from the Qiskit stack.

| Profile | File | Default name | Default for |
| --- | --- | --- | --- |
| `simulation` | `simulation.yaml` | `qml-zno-simulation` | `run_simulations.sh`, `notebooks/run_zno_notebook.sh` |
| `modeling` | `modeling.yaml` | `qml-zno-modeling` | `run_datasets.sh`, `run_models.sh`, `run_analysis.sh`, `run_all.sh` |

Both files also contain the analysis, notebook and test packages (ASE, pandas,
matplotlib, PyYAML, JupyterLab, nbconvert, pytest, scikit-learn). Run manifests record
the package versions actually used.

| Package | Simulation | Modeling | pip |
| --- | --- | --- | --- |
| Python | 3.10 | 3.10 | >= 3.10 |
| GPAW (OpenMPI build) | 25.7 | - | - |
| ASE | 3.28 | 3.28 | >= 3.23, < 3.30 |
| numpy / scipy | 2.2 / 1.15 | 2.1 / 1.15 | `requirements.txt` |
| pandas / matplotlib / PyYAML | 2.3 / 3.10 / 6.0 | 2.2 / 3.10 / 6.0 | `requirements.txt` |
| scikit-learn | 1.6 | 1.6.1 | `requirements-modeling.txt` |
| qiskit / aer / machine-learning / ibm-runtime | - | 1.4.6 / 0.17.2 / 0.8.4 / 0.43.1 | `requirements-modeling.txt` |

## Prerequisites

- Linux (x86-64 or aarch64) or macOS. Windows: WSL2, Linux steps.
- bash and one of conda, mamba or micromamba
  ([Miniforge](https://github.com/conda-forge/miniforge) provides conda and mamba).
- macOS on Apple Silicon: GPAW 25.7 has no `osx-arm64` build. `setup_environment.sh`
  builds the simulation environment from `osx-64` packages, which need Rosetta 2
  (`softwareupdate --install-rosetta`). The modeling environment is native.

## Create

```bash
./scripts/environment/setup_environment.sh simulation
./scripts/environment/setup_environment.sh modeling
./scripts/environment/setup_environment.sh all        # both, default names and files
```

The script creates the environment, then runs `check_environment.sh` on it. An existing
environment is left unchanged and only checked.

| Option | Effect |
| --- | --- |
| `--env NAME\|PATH` (alias `--name`) | Other name, or prefix path (contains `/`); one profile only |
| `--file FILE` | Other environment file; one profile only |
| `--manager CMD` | conda, mamba or micromamba (default: `QML_ZNO_CONDA`, else first found in `PATH`) |
| `--update` | Apply the file to an existing environment (never prunes) |
| `--dry-run` | Solve and print the plan, install nothing; not combinable with `--update` |

Manual equivalent: `conda env create -n NAME -f environments/simulation.yaml`
(Apple Silicon: prefix with `CONDA_SUBDIR=osx-64`).

## Select an environment

Check an existing environment under its own name, then point the launchers at it:

```bash
./scripts/environment/check_environment.sh simulation --env gpaw-dft --mpi
./scripts/environment/check_environment.sh analysis --env gpaw-dft
export QML_ZNO_SIM_ENV=gpaw-dft
export QML_ZNO_ANALYSIS_ENV=gpaw-dft
```

| Variable | Read by | Default |
| --- | --- | --- |
| `QML_ZNO_SIM_ENV` | `run_simulations.sh`, `notebooks/run_zno_notebook.sh`, `check_environment.sh simulation` | `qml-zno-simulation` |
| `QML_ZNO_ANALYSIS_ENV` | `run_analysis.sh`, `check_environment.sh analysis`; analysis notebook in `notebooks/run_zno_notebook.sh` | `QML_ZNO_MODEL_ENV` (notebook runner: `QML_ZNO_SIM_ENV`) |
| `QML_ZNO_MODEL_ENV` | `run_datasets.sh`, `run_models.sh`, `run_all.sh`, `check_environment.sh modeling` | `qml-zno-modeling` |
| `QML_ZNO_CONDA` | all launchers, `notebooks/run_zno_notebook.sh`, `setup_environment.sh`, `check_environment.sh` | first of conda, mamba, micromamba in `PATH` |

- Values are environment names, or prefix paths when they contain `/`.
- Before starting, each launcher checks that the environment imports its modules
  (also for `--dry-run`):

  | Launcher | Imports checked |
  | --- | --- |
  | `run_simulations.sh` | ase, gpaw, numpy, yaml |
  | `run_analysis.sh` | ase, matplotlib, numpy, pandas, yaml |
  | `run_datasets.sh` | numpy, pandas, yaml |
  | `run_models.sh` | numpy, pandas, qiskit, qiskit_machine_learning, sklearn, yaml |
  | `run_all.sh` | yaml (each stage launcher checks its own environment) |

- The launcher exports the environment name (last path component for a prefix) as
  `QML_ZNO_ACTIVE_ENVIRONMENT`; the run manifest records it. Without a launcher, the
  workflows record the name of the interpreter's `envs/<name>` directory, else
  `CONDA_DEFAULT_ENV`, else `execution.environment` of the YAML.
- A missing package: `conda install -n NAME -c conda-forge PACKAGE` (the check names it).

## pip only (no GPAW)

Analyses, figures, datasets, models and tests run without GPAW:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt            # analyses, figures, datasets, run_status
python -m pip install -r requirements-modeling.txt   # + scikit-learn, Qiskit
python -m pip install -r requirements-dev.txt        # + scikit-learn, pytest
./scripts/environment/check_environment.sh analysis --python .venv/bin/python
```

The launchers need a package manager. In a venv, call the modules from the repository
root with the launcher arguments, e.g. `python -m scripts.workflows.run_analysis --help`.
`run_all` calls the stage launchers, so it needs a package manager also when started
with `python -m`. GPAW with MPI comes only from `simulation.yaml`.

## MPI

- `simulation.yaml` ships OpenMPI 5. Use its `mpiexec`
  (`conda run -n NAME which mpiexec`); do not mix it with a system MPI.
- Simulations run `mpiexec <extra_args> -n <ranks> gpaw python -m ...` with
  `OMP_NUM_THREADS` = `execution.omp_num_threads` (1). For manual runs:
  `export OMP_NUM_THREADS=1`.
- Ranks: `execution.mpi.processes` = 6 (one per k-point of the production cell).
  With fewer cores: `--mpi-processes N`, N <= physical cores; 1, 2, 3 or 6 divide the
  6 k-points evenly.
- `execution.mpi.extra_args` (`--mca btl self,vader`) is OpenMPI syntax.

## Verify

```bash
./scripts/environment/check_environment.sh simulation --mpi   # add --env NAME
./scripts/environment/check_environment.sh modeling
./scripts/environment/check_environment.sh analysis
```

| Option | Effect |
| --- | --- |
| `--env NAME\|PATH` | Environment to check (default: the variable of the profile, see above) |
| `--python PATH` | Check an interpreter directly, without a package manager |
| `--manager CMD` | conda, mamba or micromamba |
| `--mpi` | Two-process MPI check (simulation only) |

The simulation check imports the packages, probes the GPAW/ASE APIs the scripts use,
prints `gpaw info`, looks up the PAW datasets for H, C, O, Zn, Ti and Ce, runs a
one-process GPAW dry run and, with `--mpi`, a two-process run that requires
`world.size == 2`. No calculation is started.

IBM credentials are never stored in the environment files or the repository; save them
with Qiskit IBM Runtime's credential storage.

## Common errors

| Message | Fix |
| --- | --- |
| `None of conda, mamba or micromamba was found in PATH` | Install Miniforge, `conda init bash`, open a new shell |
| `Environment 'NAME' was not found` / `is not usable` | `conda env list`; pass `--env NAME` or set the `QML_ZNO_*_ENV` variable |
| `Environment 'NAME' lacks: ...` | Install the listed modules, or select another environment |
| Solver finds no `gpaw 25.7` on macOS arm64 | Use `setup_environment.sh`, or prefix with `CONDA_SUBDIR=osx-64` |
| `world.size=1: GPAW was built without MPI` | `conda install -n NAME -c conda-forge "gpaw=25.7.*=*mpi_openmpi_omp*"` |
| PAW datasets not found | `conda install -n NAME -c conda-forge gpaw-data`, or `gpaw install-data DIR` and set `GPAW_SETUP_PATH` |
| `There are not enough slots available` | Fewer ranks: `--mpi-processes N` |
| `mpirun has detected an attempt to run as root` | Containers only: `export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` |
| `ModuleNotFoundError: pytest` or `sklearn` in tests | `conda install -n NAME -c conda-forge pytest scikit-learn` |
| `API missing: ...` in the check | ASE or GPAW version outside the table above |
