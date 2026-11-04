# Notebooks

Secondary route to the Python workflow. Each cell calls the same modules with
`sys.executable`, so a notebook and the command line produce the same files. Numerical
settings come from the YAML files, not from the notebooks:
`configs/simulations/zno_production.yaml` (copied into each run as `config.yaml`) and
`configs/analysis/zno_current.yaml`.

| Notebook | Runs | Configuration cell |
| --- | --- | --- |
| `zno-co2-lucas.ipynb` | Simulations through `scripts.workflows.run_simulations` (geometry, DFT, CO₂RR, TDDFT, MD, NEB) and a summary of the raw outputs; it runs exactly what `./run_simulations.sh` runs | `RUN_ID`, sites, `MPI_PROCESSES` (6), `RETRY_FAILED`, `RUN_*` flags |
| `zno-co2-analysis.ipynb` | Analyses of a finished or running simulation run: status dashboard, then `run_analysis` for `co2_reduction` (with the CO₂RR corrections), `neb`, `md_trajectory` and `paper`, exactly as `./run_analysis.sh` | `RUN_ID`, `OUTPUT_ID`, `ANALYSIS_CONFIG` (`configs/analysis/zno_current.yaml`), `RUN_*` flags |

- Every `RUN_*` flag is `False` in the saved notebooks: nothing runs until one is set.
- Start them from `notebooks/` or the repository root.
- Simulation notebook: when `results/simulations/<RUN_ID>/run_manifest.json` exists,
  each method runs with `--resume` and the run's `config.yaml`; `RETRY_FAILED = True`
  adds `--retry-failed`. Resume rules: [docs/simulations.md](../docs/simulations.md).
  The environment recorded in the run is `QML_ZNO_SIM_ENV` if set, else the kernel's
  environment; a prefix path is recorded by its last component, as by the launchers.
- Analysis notebook: each section runs `python -m scripts.workflows.run_analysis <method>
  --config configs/analysis/zno_current.yaml --scientific-results <run> --run-id
  <OUTPUT_ID>_<section>` into a new `results/analyses/<OUTPUT_ID>_<section>/` (settings
  from the YAML, not from the notebook; an existing ID is refused, so change `OUTPUT_ID`
  to rerun). The simulation run is read only, except `md_frame_analysis.csv` written by
  the MD analysis. Paper items: [docs/figures.md](../docs/figures.md).

## Environment

The simulation notebook needs GPAW; the analysis notebook needs ASE, pandas, matplotlib
and PyYAML. `./setup_lucas_env.sh` (from this directory) creates or checks the
simulation environment; it accepts the options of `setup_environment.sh`, e.g.
`--env NAME`. Details: [environments/README.md](../environments/README.md).

## Execute

Interactive, in the selected environment:

```bash
cd notebooks
jupyter lab zno-co2-analysis.ipynb
```

Headless:

```bash
./run_zno_notebook.sh zno-co2-lucas.ipynb zno-co2-lucas.executed.ipynb
./run_zno_notebook.sh zno-co2-analysis.ipynb zno-co2-analysis.executed.ipynb
QML_ZNO_ANALYSIS_ENV=gpaw-dft ./run_zno_notebook.sh zno-co2-analysis.ipynb run2.executed.ipynb
```

| Item | Behavior |
| --- | --- |
| Arguments | `<input> [<output>]`, both relative to `notebooks/`; default input `zno-co2-lucas.ipynb`, default output `<input>.executed.ipynb` |
| Existing output | Refused; choose a new output name |
| Environment | Simulation notebook: `QML_ZNO_SIM_ENV` (default `qml-zno-simulation`). Analysis notebook: `QML_ZNO_ANALYSIS_ENV`, else `QML_ZNO_SIM_ENV`. Manager from `QML_ZNO_CONDA` |
| Checks | `check_environment.sh analysis` for a notebook whose name contains `analysis`, else `check_environment.sh simulation`; then nbconvert, nbclient and ipykernel must import |
| `NB_TIMEOUT` | Per-cell timeout in seconds; default `-1` (none), so a long MPI cell is not killed |
| `OMP_NUM_THREADS` | Default 1 |
| Flags | The notebook runs with the `RUN_*` flags saved in it |

For long MD or NEB runs the launcher inside `screen` or `tmux` is safer; the notebook
continues the same run afterwards. Progress:
`python -m scripts.workflows.run_status <RUN_ID>`.
