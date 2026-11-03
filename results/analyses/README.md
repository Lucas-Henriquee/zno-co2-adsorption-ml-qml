# Analysis runs

One directory per run of `./run_analysis.sh`.

```text
<run-id>/
├── config.yaml
├── run_manifest.json
├── logs/
├── science/
│   ├── screening/
│   ├── co2_reduction/
│   └── figures/
└── models/
    ├── tables/
    └── figures/
```

Method directories exist only for the selected methods. Run directories are ignored by
Git. Analyses made before the 2026-10 audit read the invalid first campaign; keep them,
if at all, in `results/_local_archive_pre_audit/`.
