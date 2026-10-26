# Invalid results from the first IC campaign (archived)

Everything in this folder was produced before the 2026-10 audit and is **not valid**.
It is kept only for provenance; do not use it for analysis, figures or the manuscript.

Main reasons:

- **Wrong ZnO bulk basis.** Zn and O were placed on Wyckoff 2a (0, 0, z) instead of
  2b (1/3, 2/3, z), which gives Zn-O distances of 0.61 A instead of 1.97-1.99 A. Every slab,
  adsorption energy and descriptor derived from it is unphysical (clean slab +317 eV).
- **1x1 surface cell.** The CO2 was 0.93 A from its own periodic image.
- **Site and convergence problems.** The bridge site used a subsurface O; some relaxations
  stopped at the step limit and were reported as converged.
- **Method-level errors** (see the audit for details): TDDFT energies in Hartree labelled as eV
  and |me| used as the oscillator strength; MD friction in the wrong units and no equilibration;
  NEB barriers taken over the endpoints; CO2RR electronic energies labelled as dG, with O* in the
  profile; ML datasets built from a cartesian product of 3 calculations (960 rows), with
  trajectory windows leaking between train and test.

Contents:

| Path | Origin |
| --- | --- |
| `dft/`, `tddft/`, `md/`, `neb*/`, `pathways/`, `*_summary.csv` | Earlier simulation campaigns |
| `svm*/`, `qsvm*/`, `benchmarks/`, `models/` | Model experiments on the leaked datasets |
| `datasets/`, `tables/`, `figures/`, `my_analysis/`, `analysis/` | Derived data and analyses |
| `data/geometries_ZnO/` | Old input geometries (former `data/geometries/ZnO/`) |
| `data/qml_datasets/` | Old TDDFT QML datasets (former `data/qml_datasets/`) |

The corrected geometry is in `data/geometries/ZnO/` (2x1 ZnO(10-10) supercell, 16 atoms) and new
campaigns are written to `results/simulations/<run-id>/`.

The scripts under `scripts/visualization/benchmarks/` and `scripts/visualization/plot_ml_results.py`
still point at the old benchmark paths and will fail until they are regenerated from new runs.
