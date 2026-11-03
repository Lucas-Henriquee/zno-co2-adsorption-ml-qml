# Analysis

`run_analysis.sh` builds tables, reports and figures from one simulation run and/or one
model run into `results/analyses/<run-id>/`. Configuration:
[configs/analysis/README.md](../configs/analysis/README.md).

```bash
RUN=results/simulations/<run-id>
python -m scripts.analysis.co2rr_corrections --results-dir $RUN   # first, for CO2RR variant c
./run_analysis.sh science --scientific-results $RUN --dry-run
./run_analysis.sh science --scientific-results $RUN --run-id zno_science_01
./run_analysis.sh models --model-run results/models/<run-id> --run-id zno_models_analysis_01
./run_analysis.sh all --scientific-results $RUN --model-run results/models/<run-id>
```

| Method | Writes | Needs |
| --- | --- | --- |
| `screening` | `science/screening/` (incl. `ranked_candidates.csv`) | `dft_summary.csv`, `tddft_summary.csv` |
| `co2_reduction` | `science/co2_reduction/` | the same plus `co2rr_pathway_summary.csv` |
| `science_figures` | `science/figures/` | `co2_reduction` (added automatically), `tddft/*/*/transitions.csv` |
| `model_analysis` | `models/tables/`, `models/figures/` | exactly one `benchmark_summary.csv`, and `confusion_matrix.csv` files, below the model run |

Aliases: `science` = the first three, `models` (or `model-analysis`) =
`model_analysis`, `all` = every method.

| Option | Effect |
| --- | --- |
| `--scientific-results DIR` | Simulation run (default `input.scientific_results`) |
| `--model-run DIR` | Model run (default `input.model_run`) |
| `--config PATH` | Other YAML |
| `--run-id NAME` | Run directory name; default next `analysis_NN`; never reused |
| `--dry-run` | Validate inputs and print the commands |

- CO₂RR values come from variant c of `co2rr_corrected_summary.csv` when it exists in
  the run, else from `co2rr_pathway_summary.csv`.
- Each run holds its `config.yaml`, `run_manifest.json` (input and output hashes) and
  `logs/`. Simulation and model runs are read only.
- `notebooks/zno-co2-analysis.ipynb` runs the same producers.

Model figures are descriptive. The single-split, few-trajectory comparison is not
evidence of quantum advantage or of generalization to unseen materials or trajectories.
