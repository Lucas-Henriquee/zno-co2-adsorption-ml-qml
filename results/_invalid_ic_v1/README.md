# First IC campaign (invalid, archived)

Everything in this folder was produced before the 2026-10 audit and is not valid. It is
kept for provenance only; do not use it for analyses, figures or the manuscript.

## Errors

- ZnO bulk basis: Zn and O on Wyckoff 2a (0, 0, z) instead of 2b (1/3, 2/3, z), giving
  Zn-O distances of 0.61 Å instead of 1.97-1.99 Å. Every slab, adsorption energy and
  descriptor derived from it is unphysical (clean slab +317 eV).
- 1×1 surface cell (8 atoms): the CO₂ was 0.93 Å from its periodic image.
- Sites and convergence: the bridge site used a subsurface O; some relaxations stopped at
  the step limit and were reported as converged.
- Methods: TDDFT energies in Hartree labelled as eV and |me| used as oscillator
  strength; MD friction in wrong units and no equilibration; NEB barriers taken over the
  endpoints; CO₂RR electronic energies labelled as ΔG, with O* in the profile; ML
  datasets built as a cartesian product of 3 calculations (960 rows), with trajectory
  windows leaking between train and test.

## Contents

| Path | Origin |
| --- | --- |
| `dft/`, `tddft/`, `md/`, `neb*/`, `pathways/`, `*_summary.csv` | Simulation campaigns |
| `svm*/`, `qsvm*/`, `benchmarks/` | Model experiments on the leaked datasets |
| `datasets/`, `tables/`, `figures/`, `my_analysis/`, `analysis/`, `gaussian_broadening/` | Derived data and analyses |
| `data/geometries_ZnO/` | Old input geometries (former `data/geometries/ZnO/`) |
| `data/qml_datasets/` | Old TDDFT QML datasets (former `data/qml_datasets/`) |
| `manuscript_figures/` | ML figures and tables of the draft manuscript (former root `figures/`) |

The corrected geometry is `data/geometries/ZnO/` (2×1 ZnO(10-10), 16 atoms); new runs
go to `results/simulations/<run-id>/`. `scripts/visualization/benchmarks/` and
`scripts/legacy/visualization/plot_ml_results.py` still point at the old benchmark paths
and fail. Untracked pre-audit runs can be kept in `results/_local_archive_pre_audit/`
(ignored by Git).
