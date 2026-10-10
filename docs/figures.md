# Figures

## Adsorption structures

For PyMOL ray tracing, create the separate visualization environment:

```bash
conda env create -f environments/pymol.yml
conda run -n qml-zno-visualization python -m scripts.visualization.render_adsorption_structures \
  --view 3d --renderer pymol
```

The exporter saves a combined PNG, three individual PNGs, and editable PyMOL
sessions (`.pse`). Only the exposed layer from the saved structure is displayed,
with one CO2 molecule and no periodic copies. All panels use the same scale, with
view directions chosen to reduce overlap of the molecular atoms.
Dark halos mark the atoms defining the starting site. The labels report the two
C–O distances. The original DFT files are read without modification.

Render the saved DFT structures in 3D:

```bash
conda run -n qml-zno-simulation python -m scripts.visualization.render_adsorption_structures --view 3d
```

The PNG uses the same camera and scale for all sites, with labels below the
panels. Camera angles can be set with `--elevation` and `--azimuth`.
The outlines mark the surface atoms that defined each starting site.
Only the exposed surface region is displayed; coordinates are read from the
saved DFT XYZ files and checked against `results/dft_summary.csv`.

If geometry warnings are reported, review them before using the figure in the
paper. To inspect a draft, add `--allow-geometry-warnings`; its filename includes
`_diagnostic`. The earlier projections remain available through `--view combined`.

## New analysis runs

```bash
./run_analysis.sh science --run-id science_figures_01
./run_analysis.sh models --model-run results/models/MODEL_RUN --run-id model_figures_01
```

`render_publication_figures.py` renders the scientific analysis tables and TDDFT
transitions. `plot_current_model_results.py` renders local model comparisons and
confusion matrices. Outputs go to the new analysis directory in PNG, PDF, and SVG.

## Saved manuscript benchmarks

Rebuild all four benchmark figures in a new directory:

```bash
conda run -n qml-zno-modeling python -m scripts.visualization.benchmarks.render_saved_figures \
  --output-dir results/analyses/manuscript_benchmarks_01
```

Use `--plots generalization qubits configuration ibm` to select figures.
These scripts read saved results and do not train models or submit IBM jobs.

| Figure | Source below `results/benchmarks/` |
| --- | --- |
| Generalization comparison | `svm_expanded`, `qsvm_expanded`, `svm_unique_windows`, `qsvm_unique_windows` |
| Qubit sweep | `qsvm_qubit_comparison` |
| Circuit configuration comparison | `qsvm_entanglement_comparison`, `qsvm_repetition_comparison` |
| IBM comparison | `ibm_qubit_comparison`, `ibm_shot_comparison`, `final_matched_n12` |

The individual historical scripts still default to `figures/`; use the command
above to preserve the selected manuscript images.

## Circuit diagrams

```bash
conda run -n qml-zno-modeling python -m scripts.visualization.benchmarks.export_paper_circuits \
  --output-dir results/analyses/manuscript_circuits_01
```

The exporter reconstructs the saved three- and six-qubit logical feature maps,
checks their fingerprints and text diagrams, and writes PNG and SVG files.
It also draws a compute--uncompute block schematic.
The diagrams are not archived hardware-transpiled circuits.

## Earlier scientific plots

`scripts/legacy/visualization/analysis_plot2.py` contains the DFT descriptor
and TDDFT optical plots used during manuscript preparation. Its
`scripts/visualization/analysis_plot2.py` wrapper preserves the former command.
It uses fixed campaign paths and an output directory defined in the script.
The other earlier MD and CO₂RR plots are listed in
[the historical plotting guide](../scripts/legacy/README.md).

Keep run outputs under `results/analyses/`; copy selected final figures into
`figures/` when preparing the manuscript.
