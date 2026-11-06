# Dataset runs

One directory per run of `./run_datasets.sh` (or `python -m scripts.workflows.run_datasets`).

```text
<run-id>/
├── config.yaml
├── run_manifest.json     # inputs with SHA-256, commands, versions, method status
├── logs/
├── tddft/
└── multitech/
```

Method directories exist only for the selected methods. Run directories are ignored by
Git.
