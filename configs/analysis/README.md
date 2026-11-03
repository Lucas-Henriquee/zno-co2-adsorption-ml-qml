# Analysis configuration

`zno_current.yaml` connects one simulation run and one model run; output goes to a new
directory below `results/analyses/`. Commands: [docs/analysis.md](../../docs/analysis.md).

| Key | Value |
| --- | --- |
| `input.scientific_results` | `results/simulations/zno_ic_final` (`--scientific-results`) |
| `input.model_run` | `results/models/zno_md_grouped_local` (`--model-run`); no valid model run exists yet |
| `methods.co2_reduction.require_pathway_results` | `true` |
| `methods.science_figures.dpi` / `methods.model_analysis.dpi` | 600 / 300 |

- Screening ranks successful DFT rows (`dft_status == ok`, carbonate excluded) by
  adsorption energy. No composite score.
- Band gaps are PBE Kohn-Sham gaps (`*_band_gap_gamma_ev`); TDDFT values describe the
  isolated CO₂. Both are reported for information only, not compared with
  photocatalysis targets.
- Science and model analyses are independent: a missing model run does not block
  `science`.
