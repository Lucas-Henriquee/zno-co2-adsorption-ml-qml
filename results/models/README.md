# Model runs

One directory per run of `./run_models.sh`.

```text
<run-id>/
├── config.yaml
├── run_manifest.json
├── logs/                    # models.log, evaluation.log
├── benchmark/benchmark_NN/
│   ├── splits/
│   ├── svm/
│   ├── qsvm/
│   ├── qsvm_ibm/
│   ├── benchmark_summary.csv
│   └── engineering_comparison.csv
└── evaluation/grouped_eval/ # protocol grouped_27 only
```

Model directories exist only for the selected methods. Run directories are ignored by
Git.
