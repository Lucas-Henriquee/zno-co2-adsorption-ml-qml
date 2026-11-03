# Simulation runs

One directory per run of `./run_simulations.sh` (or
`python -m scripts.workflows.run_simulations`, also used by
`notebooks/zno-co2-lucas.ipynb`). Progress:
`python -m scripts.workflows.run_status <run-id>`.

```text
<run-id>/
├── config.yaml             # configuration snapshot
├── run_manifest.json       # commands, hash, environment, versions, status, timestamps
├── logs/                   # one log per method
├── data/geometries/        # run-specific starting structures
├── dft/
├── tddft/
├── pathways/               # CO2RR
├── md/<material>/<site>/seed_<n>/
├── neb/
└── failed_attempts/        # only after --resume --retry-failed
```

Method directories exist only for the selected methods. Run directories are ignored by
Git (GPAW outputs are large). Do not combine files of runs with different configuration
hashes.
