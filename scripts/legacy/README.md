# First-campaign scripts

Kept for provenance only. They read fixed paths of the first campaign, whose results
are invalid and now lie below `results/_invalid_ic_v1/`
([README](../../results/_invalid_ic_v1/README.md)). Do not use them for new figures or
runs; they are not part of `run_all.sh`.

| Script | Reads |
| --- | --- |
| `visualization/analysis_plot2.py` | DFT, TDDFT, MD, CO₂RR summaries and `results/neb_iddp`; writes `results/my_analysis` |
| `visualization/plot_md_comparison.py` | `results/my_analysis/md/ZnO` |
| `visualization/plot_ml_results.py` | 17 first-campaign model run folders in `results/` |
| `visualization/render_publication_figures.py` | Tables of `analysis_co2_reduction` and `tddft/*/*/transitions.csv`; the former `run_analysis` method `science_figures` (screening heatmap, C–O activation, E_ads vs TDDFT, U_L ranking, TDDFT spectra), retired from the workflow; the paper figures are drawn by `scripts/visualization/paper_figures.py` |
| `datasets/export_qml_dataset.py` | `ranked_candidates.csv` of the screening; one row per site (3 rows, one per class), which cannot be split into training and test data; the former `run_datasets` method `qml_features` |
| `workflows/run_pipeline.py` | Flat `results/` layout of the first campaign |

`analysis_plot2.py` also plots TDDFT energies in Hartree labelled as eV and |me| as the
oscillator strength. The directory spellings are kept because saved records refer to
them. Current producers: [docs/figures.md](../../docs/figures.md).
