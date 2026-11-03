# Figures

Figure producers for one simulation run and one model run. None of them trains models,
runs DFT or submits IBM jobs. Write outputs below `results/analyses/<run-id>/` and take
manuscript figures only from there.

## Simulation run

Run inside an environment with the analysis packages, from the repository root:

```bash
RUN=results/simulations/<run-id>
OUT=results/analyses/<run-id>_figures
export QML_ZNO_RESULTS_DIR=$PWD/$RUN
```

| Content | Command | Notes |
| --- | --- | --- |
| Adsorption-energy heatmap, C-O activation, E_ads vs first bright TDDFT excitation, U_L ranking, TDDFT spectra (fig01-fig05) | `./run_analysis.sh science --scientific-results $RUN` | Only `ok` DFT rows; sites ranked by U_L of variant c |
| CO₂RR corrections (variants a-c) | `python -m scripts.analysis.co2rr_corrections --results-dir $RUN` | Writes `co2rr_corrected_summary.csv` into the run |
| CO₂RR free-energy profile | `python -m scripts.visualization.plot_co2rr_pathway --summary $RUN/co2rr_pathway_summary.csv --variant c --output-dir $OUT/co2rr` | Sites without a valid pathway (COOH*/CO* desorbed or not formed) are greyed; `--unbound hide` drops them |
| NEB image and barrier tables | `python -m scripts.analysis.neb_energy_table --base $RUN/neb/ZnO` | Barriers recomputed from `neb_final_images.traj` |
| NEB profiles and barrier comparison | `python -m scripts.visualization.plot_neb --base $RUN/neb/ZnO --smooth-band` | No barrier when below the resolution or an endpoint is not a minimum along the band |
| MD site occupancy and desorption | `python -m scripts.analysis.md_trajectory_analysis --results-dir $RUN --material ZnO` | Also writes the desorption-threshold sensitivity table |
| Adsorption structures | `python -m scripts.visualization.render_adsorption_structures --input-dir $RUN/dft/ZnO --summary $RUN/dft_summary.csv --output-dir $OUT/structures` | Relaxed DFT structures with starting sites marked |

Outputs written into the run directory:

- `neb_energy_table`, `plot_neb`: `neb/<material>/comparison_plots/` (no output option).
- `md_trajectory_analysis`: `md/<material>/<site>/md_frame_analysis.csv`; the summary
  `md_trajectory_analysis_summary.csv`, `md_desorption_sensitivity.csv` and
  `md/<material>/md_nearest_site_occupancy.png` go to `--output-dir` (default: the run).
- `co2rr_corrections`: `co2rr_corrected_summary.csv`, `co2rr_corrections_note.md`
  (`--output-dir`, default: the run; the analyses read it there).

Report U_L differences between sites together with their uncertainty.

## Model run

```bash
./run_analysis.sh models --model-run results/models/<run-id> --run-id <run-id>_figures
```

`plot_current_model_results.py` renders the local model comparison and confusion
matrices.

## First-campaign producers

`scripts/visualization/benchmarks/` and `scripts/legacy/visualization/plot_ml_results.py`
read fixed `results/benchmarks/` paths of the first campaign. Those benchmarks used
leaked datasets and now lie in `results/_invalid_ic_v1/`, so the scripts fail; do not
use them for new figures. The saved figures are in
`results/_invalid_ic_v1/manuscript_figures/`. Other legacy plotting scripts:
[scripts/legacy/README.md](../scripts/legacy/README.md).
