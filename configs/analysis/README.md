# Analysis configuration

`zno_current.yaml` is the only analysis configuration and the default of every method
and alias (`science`, `paper`, `models`, `all`). It connects one simulation run (and,
for `models`, one model run); output goes to a new directory below `results/analyses/`.
Commands and aliases: [docs/analysis.md](../../docs/analysis.md).

| Key | Value |
| --- | --- |
| `input.scientific_results` | `results/simulations/zno_ic_final` (`--scientific-results`) |
| `input.model_run` | `results/models/zno_md_grouped_local` (`--model-run`); no valid model run exists yet |
| `input.material` | `ZnO` |
| `methods.co2rr_corrections.variant` | `c`; used by `co2_reduction`, fig05 and table02 |
| `methods.co2_reduction.require_pathway_results` | `true` |
| `methods.neb.barrier_resolution_ev` | `null`: the value in the run's `config.yaml` (0.03 eV); a number overrides it |
| `methods.neb.smooth_band` / `dpi` | `true` / 600 |
| `methods.md_trajectory.desorption_threshold_ang` | 4.5 (passed to `md_trajectory_analysis`) |
| `methods.structures.geometry_dir` | `null`: `data/geometries/<material>/` of the run, else of the repository |
| `methods.structures.allow_geometry_warnings` | `false`; `true` keeps going and notes fig01 as a diagnostic draft |
| `methods.structures.render` / `dpi` | `auto` / 600; renderers: [docs/figures.md](../../docs/figures.md#fig01-renderer) |
| `methods.paper_figures.dpi` / `tddft_sigma_ev` | 600 / 0.10 |
| `methods.paper_figures.md_seed` | 1234: seed drawn in fig06 (integer without leading zeros) |
| `methods.paper_figures.figures` | `null` = fig03-fig07; or a non-empty subset, e.g. `[fig04]`; an empty list is an error |
| `methods.model_analysis.dpi` | 300 |

- Screening ranks successful DFT rows (`dft_status == ok`, carbonate excluded) by
  adsorption energy for the ML datasets. No composite score.
- Band gaps are PBE Kohn-Sham gaps; TDDFT values describe the isolated CO₂. Both are
  reported for information only.
- Science and model analyses are independent: a missing model run does not block
  `science` or `paper`.
