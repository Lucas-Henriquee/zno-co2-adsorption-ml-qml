# Analysis configuration

Each YAML connects one simulation run (and, for `models`, one model run); output goes to
a new directory below `results/analyses/`. Commands: [docs/analysis.md](../../docs/analysis.md).

| File | Use |
| --- | --- |
| `zno_current.yaml` | Default: `science`, `models`, `all` |
| `zno_paper.yaml` | `paper` alias: fig01, fig03-fig07, table02 (`--config configs/analysis/zno_paper.yaml`) |

Both share the schema and the method sections, so any method runs with either file.

| Key | Value |
| --- | --- |
| `input.scientific_results` | `results/simulations/zno_ic_final` (`--scientific-results`) |
| `input.model_run` | `results/models/zno_md_grouped_local` (`--model-run`); no valid model run exists yet |
| `input.material` | `ZnO` |
| `methods.co2rr_corrections.variant` | `c`; also used by `co2_reduction`, `paper_figures` (fig05) and `paper_tables` |
| `methods.co2_reduction.require_pathway_results` | `true` |
| `methods.neb.barrier_resolution_ev` | `null`: `methods.neb.barrier_resolution_ev` of the run's `config.yaml` (0.03 eV); a number overrides it |
| `methods.neb.smooth_band` / `dpi` | `true` / 600 in `zno_paper.yaml` (plot default in `zno_current.yaml`) |
| `methods.md_trajectory.desorption_threshold_ang` | 4.5; passed to `md_trajectory_analysis` and `paper_figures` |
| `methods.structures.geometry_dir` | `null`: `data/geometries/<material>/` of the run, else of the repository |
| `methods.structures.allow_geometry_warnings` | `false`; `true` renders the preview despite geometry warnings and notes fig01 (`paper/fig01_adsorption_structures.png`, files in `paper/fig01_pymol/`) as a diagnostic draft |
| `methods.structures.render` | `auto`: fig01 renderer of `export_pymol` (`pymol` = headless PyMOL, `ase` = ASE/matplotlib fallback, `auto` = `pymol` when `import pymol` works, else `ase`, `none` = PDB/CIF + `.pml` only); installing `pymol-open-source` (conda-forge) is optional and only changes the renderer |
| `methods.structures.dpi` | 600: fig01 PNGs and the matplotlib preview |
| `methods.paper_figures.tddft_sigma_ev` / `dpi` / `md_seed` / `figures` | 0.10 / 600 / 1234 (seed of fig06) / `null` (= fig03-fig07); fig05 uses `methods.co2rr_corrections.variant` |
| `methods.science_figures.dpi` / `methods.model_analysis.dpi` | 600 / 300 |

- Screening ranks successful DFT rows (`dft_status == ok`, carbonate excluded) by
  adsorption energy. No composite score.
- Band gaps are PBE Kohn-Sham gaps (`*_band_gap_gamma_ev`); TDDFT values describe the
  isolated CO₂. Both are reported for information only, not compared with
  photocatalysis targets.
- Science and model analyses are independent: a missing model run does not block
  `science` or `paper`.
