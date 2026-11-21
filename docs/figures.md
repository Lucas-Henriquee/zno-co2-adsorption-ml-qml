# Figures

Manuscript figures and tables come only from an analysis run,
`results/analyses/<run-id>/paper/`. None of the producers trains models, runs DFT or
submits IBM jobs.

## Paper (simulation run)

```bash
./run_analysis.sh paper --scientific-results results/simulations/<run-id> --run-id <run-id>_paper
```

Every file in A/`paper/` is listed in `figure_registry.csv` (producer, inputs, sha256);
methods and the collection step: [analysis](analysis.md).

| Item | Producer | Shows | Caveat (caption or text) |
| --- | --- | --- | --- |
| fig01 | `export_pymol` (`structures`) | Relaxed geometries, (a) top metal, (b) top oxygen, (c) bridge, same tilted side view; CO₂ and the exposed surface layer. Two files: `fig01_adsorption_structures.png` (C–O lengths) and `fig01_adsorption_structures_contact.png` (shortest CO₂–surface contact, dashed line with its length in Å) | Site names are starting positions |
| fig03 | `paper_figures` | (a) E_ads, (b) C height above the topmost surface atom (`c_surface_distance_ang`), (c) Γ gap clean vs adsorbed; `dft_status` ok only | E_ads differences up to ~0.05 eV are not resolved; PBE without dispersion |
| fig04 | `paper_figures` | Broadened TDDFT spectra (sigma 0.10 eV, `methods.paper_figures.tddft_sigma_ev`), x from 3 eV to the last written transition + 4 sigma; the two main peaks labelled once | Isolated CO₂ in the adsorbed geometries, not the ZnO–CO₂ response |
| fig05 | `paper_figures` | ΔG at 0 V vs RHE, CO₂ → COOH* → CO* → CO, variant c; invalid sites labelled in the legend | U_L differences up to ~0.1 V are not resolved |
| fig06 | `paper_figures` | Shortest CO₂–surface distance (`c_o_min_slab_distance_ang`) vs production time, one line per site | One seed (`methods.paper_figures.md_seed`); state it in the caption |
| fig07 | `paper_figures` | NEB path top metal → bridge → top oxygen chained, energies relative to the first image | Barriers below 0.03 eV or with an endpoint that is not a minimum are not barriers |
| table02 | `paper_tables` | Per site: E_ads, shortest C–surface distance (`c_min_slab_distance_ang`), U_L, NEB, MD fractions as mean (per-seed range, n), TDDFT | No rank column |

Each figure has a `_data.csv` with the plotted numbers. ML items (fig02, fig08-fig10,
tables 1, 3, 5-8) are not produced here.

### fig01 renderer

`methods.structures.render` (default `auto`) selects the renderer of `export_pymol`:
`pymol` runs PyMOL headless with the commands of `fig01_pymol/fig01_adsorption_structures.pml`;
`ase` draws the same atoms and view with ASE and matplotlib; `auto` uses PyMOL when
`import pymol` works, else `ase`; `none` writes only PDB/CIF and the `.pml` (the two
registry rows then have an empty sha256). PyMOL (`pymol-open-source`, conda-forge) is
optional. The registry note names the renderer, read from `render_info.json`. To
re-render by hand: `pymol fig01_adsorption_structures.pml` in A/`paper/fig01_pymol/`.

## Site-memory study

`scripts.visualization.paper_site_memory_figures` draws the items of the site-memory
study from saved outputs only (no DFT, MD, model or IBM run):

```bash
python -m scripts.visualization.paper_site_memory_figures \
  --pbe-run results/simulations/zno_ic_final --d3-run results/simulations/zno_d3_stage1 \
  --memory-100k results/analyses/site_memory/100k_6seeds \
  --memory-300k results/analyses/site_memory/300k_v2 \
  --sensitivity results/analyses/site_memory/{100k_6seeds,300k}_{circular,full,reps2} \
  --ibm-summaries results/analyses/site_memory/ibm_summary_100k results/analyses/site_memory/ibm_summary_300k \
  --output-dir results/analyses/site_memory/paper
```

| Item | Shows | Caveat |
| --- | --- | --- |
| `adsorption_pbe_d3` | (a) E_ads, (b) C height above the topmost surface atom, PBE and PBE-D3 | Differences up to ~0.05 eV are not resolved |
| `neb_pbe_d3` | NEB path top metal → bridge → top oxygen, PBE and PBE-D3, each relative to its top-metal image | 3 internal images per band; no internal saddle in any band |
| `site_memory` | Balanced accuracy against time at 100 K and 300 K: SVM (RBF), QSVM (ZZ, standard), QSVM (ZZ, scaled); filled markers p ≤ 0.05; dashed line random guess (1/3) | Memory times are comparable only with the same number of seeds |
| `ibm_hardware` | Correct predictions of SVM, ideal QSVM and IBM QSVM per hardware window and temperature (512 shots, reps 1, first run of a repeated configuration) | 18 trajectories per window; differences of 1-2 are within the run-to-run variation |
| `table_site_memory`, `table_adsorption` | Memory time and accuracies per model and temperature; E_ads PBE and PBE-D3 | A memory that lasts to the last window is written as ">= ... (whole run)" |
| `supp_band_gap`, `supp_height_time`, `supp_qsvm_sensitivity`, `supp_ibm_runs` | Γ gap (PBE); C height against time (mean and seed range); QSVM feature-map sensitivity; every IBM window | Supplementary |

"Scaled" is the QSVM whose features are multiplied by λ ≤ 1, chosen from the training
features only so that its kernel is as wide as the RBF kernel (bandwidth; Shaydulin &
Wild 2022). Figures come as PNG and PDF with a `_data.csv`; `registry.csv` lists every
file, its sha256 and its inputs. The structures, TDDFT spectra and CO₂RR pathway come
from the paper stage above.

## Model run

```bash
./run_analysis.sh models --model-run results/models/<run-id> --run-id <run-id>_figures
```

`plot_current_model_results.py` renders the local model comparison and confusion
matrices.

## First-campaign producers

`scripts/visualization/benchmarks/` and the scripts in `scripts/legacy/visualization/`
read fixed paths of the first campaign, whose results are invalid
(`results/_invalid_ic_v1/`); do not use them for new figures. The saved figures are in
`results/_invalid_ic_v1/manuscript_figures/`. Details:
[scripts/legacy/README.md](../scripts/legacy/README.md).
