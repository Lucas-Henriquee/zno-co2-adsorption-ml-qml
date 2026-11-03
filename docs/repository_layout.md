# Repository layout

```text
QML_ZnO/
├── run_simulations.sh, run_datasets.sh, run_models.sh, run_analysis.sh, run_all.sh
├── requirements*.txt          # pip subsets (no GPAW)
├── configs/                   # simulations/, datasets/, models/, analysis/
├── environments/              # conda environment files
├── scripts/
│   ├── workflows/             # modules behind the launchers, run_status
│   ├── environment/           # setup_environment.sh, check_environment.sh
│   ├── simulations/           # dft (incl. build_geometries), tddft, co2rr, md, neb
│   ├── datasets/
│   ├── models/                # ml, qml, qml_ibm, benchmarks, evaluate_grouped
│   ├── analysis/
│   ├── visualization/
│   ├── common/
│   └── legacy/                # first-campaign scripts, provenance only
├── data/
│   ├── geometries/ZnO/        # starting structures
│   └── generated/<run-id>/    # dataset runs
├── results/
│   ├── simulations/<run-id>/
│   ├── models/<run-id>/
│   ├── analyses/<run-id>/
│   ├── workflows/<run-id>/    # run_all.sh manifests and logs
│   └── _invalid_ic_v1/        # first campaign, invalid
├── notebooks/                 # zno-co2-lucas.ipynb, zno-co2-analysis.ipynb, launchers
├── docs/
└── tests/
```

- Run Python modules from the repository root, e.g.
  `python -m scripts.simulations.md.run_md --help`.
- Without `--run-id` the stage launchers number runs (`simulation_01`, `dataset_01`,
  `analysis_01`, `<method>_01` for models). Run directories are ignored by Git; their
  READMEs document the layout.
- `scripts/legacy/workflows/run_pipeline.py` and the other scripts in `scripts/legacy/`
  use the flat first-campaign paths ([scripts/legacy/README.md](../scripts/legacy/README.md)).
  `scripts/visualization/benchmarks/` also reads first-campaign paths
  ([figures.md](figures.md)).
