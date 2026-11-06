# Analysis

`run_analysis.sh` builds tables and figures from one simulation run (and, for `models`,
one model run) into a new directory `results/analyses/<run-id>/`. One configuration,
`configs/analysis/zno_current.yaml`, is the default for every method
([keys](../configs/analysis/README.md)).

```bash
RUN=results/simulations/<run-id>
./run_analysis.sh paper --scientific-results $RUN --dry-run
./run_analysis.sh paper --scientific-results $RUN --run-id zno_paper_01
./run_analysis.sh science --scientific-results $RUN --run-id zno_science_01
./run_analysis.sh models --model-run results/models/<run-id> --run-id zno_models_analysis_01
```

`python -m scripts.workflows.run_analysis` takes the same arguments in an environment
that already has the analysis packages. `./run_analysis.sh --help` works without conda.

## Aliases

| Alias | Methods |
| --- | --- |
| `paper` | `co2rr_corrections`, `neb`, `md_trajectory`, `structures`, `paper_figures`, `paper_tables`, then the paper collection |
| `science` | `screening`, `co2rr_corrections`, `co2_reduction` (tables only) |
| `models`, `model-analysis` | `model_analysis` |
| `all` | every method |

Dependencies are added automatically. `paper_tables` needs `co2rr_corrections`, `neb`
and `md_trajectory`; `paper_figures` needs only those its figures read (fig05:
`co2rr_corrections`, fig06: `md_trajectory`, fig07: `neb`).

## Methods

A = `results/analyses/<run-id>/`; RUN = the simulation run.

| Method | Writes into A | Needs (in RUN unless noted) |
| --- | --- | --- |
| `screening` | `science/screening/ranked_candidates.csv` (descriptor ranking; no dataset reads it) | `dft_summary.csv`, `tddft_summary.csv` |
| `co2rr_corrections` | `science/co2rr/`: `co2rr_corrected_summary.csv` (variants a-c), note, diagnostic pathway plot and table | `co2rr_pathway_summary.csv`, `pathways/**/energy.json` |
| `co2_reduction` | `science/co2_reduction/` (merged DFT/TDDFT/CO₂RR tables) | `dft_summary.csv`, `tddft_summary.csv`, `co2rr_pathway_summary.csv`; `co2rr_corrections` |
| `neb` | `science/neb/`: `neb_image_energies.csv`, `neb_barriers.csv` (read by the paper); diagnostic plots and tables of `plot_neb` | `neb/<material>/*/neb_run_summary.json` (+ `neb_final_images.traj`) |
| `md_trajectory` | `science/md/`: `md_trajectory_analysis_summary.csv`, `md_desorption_sensitivity.csv`, diagnostic `md_nearest_site_occupancy.png` | `md/<material>/*/seed_*/md_*.traj` |
| `structures` | `science/structures/pymol/` (fig01, `export_pymol`) and the diagnostic preview `science/structures/zno_adsorption_sites_combined.png` | `dft_summary.csv`, the three `dft/<material>/<site>/<material>_<site>_optimized.xyz`, `metadata.json` and `clean_slab.traj` of `data/geometries/<material>/` (the run's, else the repository's) |
| `paper_figures` | `paper/`: fig03-fig07 (`.png`, `.pdf`, `_data.csv`) | `dft_summary.csv`; for fig04 also `tddft_summary.csv` and `tddft/<material>/*/transitions.csv` |
| `paper_tables` | `paper/table02_site_summary.{csv,md,tex}` | `dft_summary.csv` (`tddft_summary.csv` optional) |
| `model_analysis` | `models/tables/`, `models/figures/` | exactly one `benchmark_summary.csv`, and `confusion_matrix.csv` files, below the model run |

Missing optional inputs make the producers skip a figure or leave a table column `n/a`
(`[skip]` in the log); a skipped figure is listed in the registry with an empty sha256.

## Paper collection

Runs after the methods whenever `paper_figures` or `paper_tables` ran:

- A/`science/structures/pymol/` is copied to A/`paper/fig01_pymol/`, and the two fig01
  PNGs (`fig01_adsorption_structures.png`, `fig01_adsorption_structures_contact.png`)
  to A/`paper/`. A rendered fig01 without both PNGs fails the collection.
- A/`paper/figure_registry.csv` has one row per file (`item`, `file`,
  `producer_module`, `inputs`, `sha256`, `note`).
- Geometry warnings of the structures preview (C more than 1.5 Å from its site, slab
  atoms closer than 1.0 Å) fail `structures` and stop the run, unless
  `methods.structures.allow_geometry_warnings` is true; fig01 is then noted as a
  diagnostic draft.

Items, producers and caveats: [figures](figures.md). Partial MD (seeds still running)
is accepted; figures and tables report the trajectories present.

## Options

| Option | Effect |
| --- | --- |
| `--scientific-results DIR` | Simulation run (default `input.scientific_results`) |
| `--model-run DIR` | Model run (default `input.model_run`) |
| `--config PATH` | Another YAML with the same schema |
| `--run-id NAME` | Run directory name; default next `analysis_NN`; never reused |
| `--dry-run` | Validate inputs and print the commands and copies |
| `--list-methods` | Print methods and aliases |

## Records

- Each analysis run holds `config.yaml`, `logs/` and `run_manifest.json`: Git commit and
  dirty flag, environment, versions, sha256 of the simulation run's `run_manifest.json`
  and of every input, per-method commands and output hashes, and the paper registry.
  Paths in the manifest and registry are relative to the project root (absolute when
  outside it); `commands` keep the absolute argv that ran.
- Simulation and model runs are read only, except that `md_trajectory` writes
  `md/<material>/<site>/md_frame_analysis.csv` into RUN (the dataset builder reads it
  there; hashed in the manifest under `run_outputs`).
- `notebooks/zno-co2-analysis.ipynb` runs the same `run_analysis` commands.

Model figures are descriptive. The single-split, few-trajectory comparison is not
evidence of quantum advantage or of generalization to unseen materials or trajectories.
