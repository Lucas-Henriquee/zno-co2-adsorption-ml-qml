# Dataset Configuration

Dataset generation is configured independently from simulations and models.
`zno_existing_results.yaml` reads one isolated simulation run,
`results/simulations/<run-id>/`, and never writes back into `results/`. The
legacy `results/` tree was produced with a wrong geometry and must not be used.

Relative input paths are resolved below `input.results_dir`. Select another run
without editing YAML:

```bash
./run_datasets.sh all \
  --source-results results/simulations/zno_run_01 \
  --ranking results/analyses/<analysis-run>/science/screening/ranked_candidates.csv \
  --run-id zno_run_01_datasets
```

## Expected inputs (below the run directory)

- `dft_summary.csv`: rows with `dft_status == ok` only; the exploratory site
  `carbonate` is excluded. Band gaps are read from `*_band_gap_gamma_ev` (PBE
  Kohn-Sham gaps, not comparable to the experimental 3.37 eV).
- `tddft_summary.csv` (optional): `co2_isolated_first_excitation_ev`,
  `co2_isolated_onset_ev`, `co2_isolated_peak_ev`, `co2_isolated_total_f`, rows
  with `tddft_status == ok`. These describe the isolated CO2 molecule.
- `co2rr_pathway_summary.csv` (optional): rows with `pathway_status == ok`;
  step energies are read from `dG_*` or `dE_*` according to `energy_basis` and
  stored as `feature_co2rr_d_<step>_ev` (basis kept in `co2rr_energy_basis`),
  plus `limiting_potential_v`.
- `md/<material>/<site>/md_production_frames.csv`: production frames of every
  seed, with `seed`, `phase`, `time_fs` and the descriptors
  `c_surface_distance_ang`, `c_min_slab_distance_ang`,
  `o_min_metal_distance_ang`, `co_bond_avg_ang`, `oco_angle_deg`. Only rows with
  `phase == production` are used.

## Outputs (`multitech`)

- `multitech_selected_features_dataset.csv`: one row per successful DFT
  calculation, other methods joined one to one (no cartesian product).
- `md_window_unique_dataset.csv`: one row per block of production frames. The
  block length is at least `block_tau_factor` times the integrated
  autocorrelation time (`tau_int_fs`, also reported with the lag-1
  autocorrelation) and at least `min_block_length_fs`; `md_windows_per_site`
  caps the number of blocks per trajectory (0 = no cap). `group` =
  `material/site/seed_<n>` identifies the trajectory; the label is the starting
  site.
- `md_trajectory_dataset.csv`: one row per independent trajectory.

The former `expanded` target multiplied TDDFT transitions, MD windows and
per-site DFT/CO2RR constants (960 rows from 3 calculations) and leaked the site
label; it was removed.

`qml_features` exports the ranked DFT table without zero imputation (rows with a
missing selected feature are dropped) and with the site label as the target.
