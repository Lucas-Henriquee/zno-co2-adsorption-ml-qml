# Analysis

`run_analysis.sh` builds tables, reports and figures from one simulation run and/or one
model run into `results/analyses/<run-id>/`. Configuration:
[configs/analysis/README.md](../configs/analysis/README.md).

```bash
RUN=results/simulations/<run-id>
./run_analysis.sh science --scientific-results $RUN --dry-run
./run_analysis.sh science --scientific-results $RUN --run-id zno_science_01
./run_analysis.sh paper --config configs/analysis/zno_paper.yaml --scientific-results $RUN --run-id zno_paper_01
./run_analysis.sh models --model-run results/models/<run-id> --run-id zno_models_analysis_01
./run_analysis.sh all --scientific-results $RUN --model-run results/models/<run-id>
```

`python -m scripts.workflows.run_analysis` takes the same arguments in an environment
that already has the analysis packages.

A = `results/analyses/<run-id>/`; RUN = the simulation run.

| Method | Writes into A | Needs (in RUN unless noted) |
| --- | --- | --- |
| `screening` | `science/screening/` (incl. `ranked_candidates.csv`) | `dft_summary.csv`, `tddft_summary.csv` |
| `co2rr_corrections` | `science/co2rr/`: `co2rr_corrected_summary.csv`, `co2rr_corrections_note.md`, `co2rr_pathway.png`, `co2rr_pathway_profile_table.csv` | `co2rr_pathway_summary.csv`, `pathways/**/energy.json` |
| `co2_reduction` | `science/co2_reduction/` | `dft_summary.csv`, `tddft_summary.csv`, `co2rr_pathway_summary.csv`; `co2rr_corrections` (A/`science/co2rr/co2rr_corrected_summary.csv`, configured variant) |
| `science_figures` | `science/figures/` | `co2_reduction`, `tddft/*/*/transitions.csv` |
| `neb` | `science/neb/`: `neb_image_energies.csv`, `neb_barriers.csv`, `neb_energy_table.csv`, `neb_summary_table.csv`, `migration_barrier_comparison.png`, `individual_profiles/` | `neb/<material>/*/neb_run_summary.json` (+ `neb_final_images.traj`) |
| `md_trajectory` | `science/md/`: `md_trajectory_analysis_summary.csv`, `md_desorption_sensitivity.csv`, `md/<material>/md_nearest_site_occupancy.png` | `md/<material>/*/seed_*/md_*.traj` |
| `structures` | `science/structures/pymol/` (`export_pymol --render <methods.structures.render>`: `<site>.pdb`, `<site>.cif`, `fig01_adsorption_structures.pml`, `fig01_<site>.png`, `fig01_adsorption_structures.png`, `render_info.json`, `README.txt`) and the preview `science/structures/zno_adsorption_sites_combined.png` | `dft_summary.csv`, `dft/<material>/<site>/<material>_<site>_optimized.xyz` (3 sites), `metadata.json` and `clean_slab.traj` of `data/geometries/<material>/` (the run's, else the repository's) |
| `paper_figures` | `paper/`: fig03-fig07 (`.png`, `.pdf`, `_data.csv`) | `dft_summary.csv`, `tddft_summary.csv`, `tddft/<material>/*/transitions.csv`, `md/<material>/*/md_frame_analysis.csv`; `co2rr_corrections`, `neb`, `md_trajectory` |
| `paper_tables` | `paper/table02_site_summary.{csv,md,tex}` | `dft_summary.csv`, `tddft_summary.csv`; `co2rr_corrections`, `neb`, `md_trajectory` |
| `model_analysis` | `models/tables/`, `models/figures/` | exactly one `benchmark_summary.csv`, and `confusion_matrix.csv` files, below the model run |

Dependencies are added automatically. Aliases:

| Alias | Methods |
| --- | --- |
| `science` | `screening`, `co2rr_corrections`, `co2_reduction`, `science_figures` |
| `paper` | `co2rr_corrections`, `co2_reduction`, `science_figures`, `neb`, `md_trajectory`, `structures`, `paper_figures`, `paper_tables`, then the paper collection |
| `models`, `model-analysis` | `model_analysis` |
| `all` | every method |

Paper collection (whenever `paper_figures` or `paper_tables` runs): the files of
A/`science/structures/pymol/` are copied into A/`paper/fig01_pymol/` and its combined
`fig01_adsorption_structures.png` into A/`paper/` (rendered by `export_pymol`: headless
PyMOL when `import pymol` works, else the ASE/matplotlib fallback; the registry note
names the renderer, read from `render_info.json`; installing `pymol-open-source` from
conda-forge is optional and only changes the renderer; with
`methods.structures.render: none` no PNG is written and the registry row
`fig01_adsorption_structures.png` has an empty `sha256` and the note "made in PyMOL from
fig01_pymol/fig01_adsorption_structures.pml"); fig03-fig07 are
drawn by `paper_figures` directly into A/`paper/` from A/`science/co2rr/`,
A/`science/neb/`, A/`science/md/` and the run files. A/`paper/figure_registry.csv` lists every file
(`item`, `file`, `producer_module`, `inputs`, `sha256`, `note`); a figure skipped by
`paper_figures` gets a row with an empty `sha256`. A structures figure with geometry
warnings makes `render_adsorption_structures` (the preview) exit with an error, so the
`structures` method fails and the paper methods after it do not run, unless
`methods.structures.allow_geometry_warnings` is true; the fig01 registry row is then noted
as a diagnostic draft.

`paper` needs a run with CO₂RR pathways, at least one NEB pair, at least one MD
trajectory and the three relaxed site geometries; a missing one stops the plan at input
validation. Partial MD (seeds still running) is accepted: the figures and tables report
the trajectories present (`n`).

| Option | Effect |
| --- | --- |
| `--scientific-results DIR` | Simulation run (default `input.scientific_results`) |
| `--model-run DIR` | Model run (default `input.model_run`) |
| `--config PATH` | Other YAML (`configs/analysis/zno_paper.yaml` for the paper) |
| `--run-id NAME` | Run directory name; default next `analysis_NN`; never reused |
| `--dry-run` | Validate inputs and print the commands and copies |
| `--list-methods` | Print methods and aliases |

- CO₂RR values come from the configured variant (c) of A/`science/co2rr/co2rr_corrected_summary.csv`.
- Each run holds its `config.yaml`, `logs/` and `run_manifest.json`: Git commit and dirty
  flag, environment, versions, sha256 of the simulation run's `run_manifest.json` and of
  every input file, per-method commands and output hashes, and the paper registry.
- Simulation and model runs are read only, except that `md_trajectory` writes
  `md/<material>/<site>/md_frame_analysis.csv` into RUN (the dataset builder reads it
  there; hashed in the manifest under `run_outputs`).
- `notebooks/zno-co2-analysis.ipynb` runs the same producers.

Model figures are descriptive. The single-split, few-trajectory comparison is not
evidence of quantum advantage or of generalization to unseen materials or trajectories.
