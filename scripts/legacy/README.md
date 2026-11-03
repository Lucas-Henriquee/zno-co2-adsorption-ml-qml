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
| `workflows/run_pipeline.py` | Flat `results/` layout of the first campaign |

`analysis_plot2.py` also plots TDDFT energies in Hartree labelled as eV and |me| as the
oscillator strength. The directory spellings are kept because saved records refer to
them. Current producers: [docs/figures.md](../../docs/figures.md).
