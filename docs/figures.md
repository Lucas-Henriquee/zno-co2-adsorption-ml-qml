# Figures

Figure producers for one simulation run and one model run. None of them trains models,
runs DFT or submits IBM jobs. Write outputs below `results/analyses/<run-id>/` and take
manuscript figures only from there.

## Paper (simulation run)

```bash
RUN=results/simulations/<run-id>
./run_analysis.sh paper --config configs/analysis/zno_paper.yaml --scientific-results $RUN --run-id <run-id>_paper
```

Everything goes to A = `results/analyses/<run-id>_paper/`; manuscript files are in
A/`paper/`, listed in `figure_registry.csv` with producer, inputs and sha256.

| Item | Files in A/`paper/` | Producer (method) | Notes |
| --- | --- | --- | --- |
| fig01 | `fig01_adsorption_structures.png`; `fig01_pymol/`: `<site>.pdb`, `<site>.cif`, `fig01_<site>.png`, `fig01_adsorption_structures.{pml,png}`, `render_info.json`, `README.txt` | `scripts.visualization.export_pymol` (`structures`) | 3D views rendered automatically by `export_pymol --render` (`methods.structures.render`, default `auto`): `pymol` = PyMOL run headless (`pymol2.PyMOL()`, else `pymol -qc`) executing the same command list as the `.pml`; `ase` = ball-and-stick fallback with ASE + matplotlib (same atoms, same rotation, element colours, spheres scaled by covalent radii with shading, sticks between atoms closer than 1.15 × the sum of covalent radii, C–O lengths); `auto` = `pymol` when `import pymol` works, else `ase`; `none` = no PNG. PyMOL is not a dependency: installing `pymol-open-source` (conda-forge) is optional and only changes the renderer. `fig01_adsorption_structures.png` = the panels side by side, (a) top metal, (b) top oxygen, (c) bridge, with an element legend. Relaxed structures of the sites with `dft_status` ok, shifted laterally by a lattice translation so that C sits at the cell centre; white background, orthographic, the same tilted side view for every site; only CO₂ and the exposed surface layer (slab atoms above a z threshold computed from each structure and written into the script). The `.pml` and PDB/CIF stay for re-rendering in PyMOL (`pymol fig01_adsorption_structures.pml`; uncomment the `ray`/`png` lines). The registry row of `fig01_adsorption_structures.png` has its sha256 and notes the renderer read from `render_info.json` (with `render: none`, an empty sha256 and "made in PyMOL from fig01_pymol/fig01_adsorption_structures.pml"). The matplotlib preview of `render_adsorption_structures` stays in A/`science/structures/`; its geometry warnings stop the method (with `allow_geometry_warnings` the fig01 row is noted as a diagnostic draft) |
| fig03 | `fig03_dft_descriptors.{png,pdf}`, `_data.csv` | `scripts.visualization.paper_figures` (`paper_figures`) | Three bar panels: (a) E_ads, (b) C–surface distance (`c_surface_distance_ang`, C height above the topmost slab atom), (c) Γ gap clean vs adsorbed; only `dft_status` ok. Panel (a): E_ads (negative = favourable) with a zero line and a symmetric y axis ±max(0.1, 1.2·max\|E_ads\|); values above positive and below negative bars |
| fig04 | `fig04_tddft_spectra.{png,pdf}`, `_data.csv` | `scripts.visualization.paper_figures` (`paper_figures`) | Gaussian-broadened spectra (sigma 0.10 eV) of the isolated CO₂ in the three adsorbed geometries, normalised by one common factor (global maximum); x = 3 eV to max(13 eV, last written transition + 4 sigma), top axis wavelength (400, 300, 200, 150, 120, 100 nm); shaded "UV (400–200 nm)" (3.10–6.20 eV) and "Vacuum UV (< 200 nm)" (> 6.20 eV) with dashed borders; legend `ZnO \| top-metal` etc. in the UV region; each curve's maximum labelled (`top-metal` / `X.XX eV`, stacked at 7 eV with a connector). A curve is truncated above its last written transition (`run_tddft --max-transitions`). Data: energy, curve, absorption and the labelled peak energy |
| fig05 | `fig05_co2rr_pathway.{png,pdf}`, `_data.csv` | `scripts.visualization.paper_figures` (`paper_figures`) | ΔG at U = 0 V vs RHE, CO₂(g) → COOH* → CO* → CO(g), configured variant (c); every site in its colour, solid line with markers; legend `Top metal (U_L = -1.98 V)` for `pathway_validity` ok, `(intermediates desorbed)` for `unbound`, `(not COOH*/CO*)` for `wrong_intermediate` |
| fig06 | `fig06_md_distance.{png,pdf}`, `_data.csv` | `scripts.visualization.paper_figures` (`paper_figures`) | CO₂–surface distance (`c_o_min_slab_distance_ang`) vs production time, one line per site, seed `methods.paper_figures.md_seed` (1234; else the site's first seed) |
| fig07 | `fig07_neb_profiles.{png,pdf}`, `_data.csv` | `scripts.visualization.paper_figures` (`paper_figures`) | One panel, the complete migration path: the NEB pairs of `neb_image_energies.csv` chained top_metal → bridge → top_oxygen (a pair stored in the other direction is reversed), energies relative to the first image of the chain, x = cumulative image index with thin vertical lines at the endpoint sites, one colour per segment. Pairs that cannot be chained are drawn on the same axes, each relative to its own initial image (`[skip]` note). Data: segment, pair, global index, endpoint site, relative energy |
| table02 | `table02_site_summary.{csv,md,tex}` | `scripts.analysis.paper_tables` (`paper_tables`) | No rank column |

The figures keep the simple style of the draft (one idea per figure, no error bars or
annotations). Caveats go into the captions, the text and Table 2: E_ads differences up
to ~0.05 eV and U_L differences up to ~0.1 V are not resolved (report them with their
uncertainty, not as a ranking); NEB barriers below the resolution or with an endpoint
that is not a minimum are not barriers; MD desorption and site occupancy are in Table 2. ML items (fig02, fig08-fig10,
tables 1, 3, 5-8) are not produced here.

Single steps, with `A=results/analyses/<run-id>` and `OUT=$A/science` (each writes into the
given `--output-dir`; these are the directories `paper_figures` and `paper_tables` read with
`--analysis-dir $A`):

| Content | Command |
| --- | --- |
| CO₂RR corrections (variants a-c) | `python -m scripts.analysis.co2rr_corrections --results-dir $RUN --output-dir $OUT/co2rr` |
| CO₂RR free-energy profile | `python -m scripts.visualization.plot_co2rr_pathway --summary $RUN/co2rr_pathway_summary.csv --corrected $OUT/co2rr/co2rr_corrected_summary.csv --variant c --output-dir $OUT/co2rr` |
| NEB image and barrier tables | `python -m scripts.analysis.neb_energy_table --base $RUN/neb/ZnO --output-dir $OUT/neb` |
| NEB profiles and barrier comparison | `python -m scripts.visualization.plot_neb --base $RUN/neb/ZnO --output-dir $OUT/neb --smooth-band` |
| MD site occupancy and desorption | `python -m scripts.analysis.md_trajectory_analysis --results-dir $RUN --material ZnO --output-dir $OUT/md` |
| Fig. 1 (PDB/CIF, PyMOL script, PNGs) | `python -m scripts.visualization.export_pymol --input-dir $RUN/dft/ZnO --summary $RUN/dft_summary.csv --output-dir $OUT/structures/pymol [--render auto\|pymol\|ase\|none] [--dpi 600] [--sites ...]` |
| Adsorption structures (matplotlib preview) | `python -m scripts.visualization.render_adsorption_structures --input-dir $RUN/dft/ZnO --summary $RUN/dft_summary.csv --output-dir $OUT/structures` |
| Earlier publication figures (heatmap, C-O activation, U_L, TDDFT) | `./run_analysis.sh science --scientific-results $RUN` |

Without `--output-dir`, the CO₂RR correction and MD summaries go into the run and the
NEB outputs into `neb/<material>/comparison_plots/`. `md_trajectory_analysis` always
writes `md/<material>/<site>/md_frame_analysis.csv` into the run.

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
