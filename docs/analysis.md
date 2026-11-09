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

## Site memory (time-resolved site identification)

`scripts/analysis/site_memory.py` asks, window by window from the start of the AIMD,
whether the starting site can still be predicted from the CO₂ geometry. It is not a
`run_analysis` method; run it in the modeling environment (scikit-learn, and Qiskit for
the QSVM).

1. Frame analysis of every simulation run (writes `md/<material>/<site>/md_frame_analysis.csv`
   into the run; `zno_ic_final` already has it):

   ```bash
   python -m scripts.analysis.md_trajectory_analysis --results-dir results/simulations/zno_md_100k \
     --material ZnO --output-dir results/analyses/site_memory/traj_100k
   ```

2. Memory curve. Runs are combined only with the same temperature, time step, log
   interval and equilibration, and distinct seeds:

   ```bash
   python -m scripts.analysis.site_memory --runs results/simulations/zno_md_100k \
     --output-dir results/analyses/site_memory/100k
   python -m scripts.analysis.site_memory \
     --runs results/simulations/zno_ic_final results/simulations/zno_md_300k_extra \
     --output-dir results/analyses/site_memory/300k
   ```

- Rows: one per trajectory per window (mean of its frames; default 40 fs windows);
  features: C height, shortest C–slab and CO₂ O–slab distances (`geometry3`). Site
  assignment columns are refused as features.
- Validation: leave one seed out; MinMax to [0, π/2] fitted on the training seeds
  (P(2x) of the ZZ map has period π); the same C (1) for every model, no tuning.
- Models: majority, nearest centroid, logistic, RBF SVM and QSVM (Qiskit
  `zz_feature_map`, statevector; `--qsvm-kernel zz_exact` computes the same kernel
  with numpy).
- Significance: site labels permuted within each seed, whole evaluation rerun. The
  smallest attainable p is 1/36 with 3 seeds and 1/7776 with 6.
- Memory time: end of the last window of the run of significant windows (p ≤ 0.05)
  that starts at t = 0. The first windows are identifiable by construction (every
  trajectory of a site starts from the same DFT geometry).
- Outputs: `site_memory_windows.csv`, `site_memory_predictions.csv`,
  `site_memory_features.csv`, `site_memory_summary.json`, `site_memory_curve.png`.

### On IBM Quantum hardware

`scripts/analysis/site_memory_ibm.py` evaluates the RBF SVM, the ideal QSVM and the
QSVM with kernels estimated on an IBM backend on a few windows chosen in advance, with
the same rows, splits, scaling and C:

```bash
python -m scripts.analysis.site_memory_ibm --runs results/simulations/zno_md_100k \
  --window-starts-fs 0 800 --output-dir results/analyses/site_memory/ibm_100k            # plan only
python -m scripts.analysis.site_memory_ibm --runs results/simulations/zno_md_100k \
  --window-starts-fs 0 800 --output-dir results/analyses/site_memory/ibm_100k --submit   # QPU
```

- Without `--submit` the script writes the plan (circuits, shots) and evaluates the
  classical and ideal models; it never contacts IBM. `--submit` needs a saved Qiskit
  Runtime account. `--reuse-hardware <output>/hardware_kernels` redoes the analysis from
  the saved kernels without the QPU, and refuses kernels computed on other data.
- `--scaling fixed` (default) maps every distance from 2.0–6.0 Å to [0, π/2] with no
  fit, so one kernel per window serves every split: N(N−1)/2 circuits (36 for 3 seeds,
  153 for 6). `--scaling train` fits MinMax per split as `site_memory.py` and needs one
  kernel per split.
- Hardware kernels: training diagonal 1, training matrix projected onto the PSD cone,
  test entries as measured. The off-diagonal deviation from the ideal kernel (MAE,
  RMSE, bias) and the reported quantum time are saved.
