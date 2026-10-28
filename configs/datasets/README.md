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
  with `tddft_status == ok`. These describe the isolated CO2 molecule, are
  nearly identical for every site and are therefore never classification
  features: the summary keeps them as `tddft_co2_isolated_*` reference columns
  (no `feature_` prefix) and the MD datasets do not contain them. The `tddft`
  method tables carry `group` = `material/site` and print the same warning.
- `md/<material>/<site>/md_frame_analysis.csv` (optional): written by
  `python -m scripts.analysis.md_trajectory_analysis --results-dir <run>`; adds
  the window site diagnostics below.
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
- `md_window_unique_dataset.csv`: one row per block of production frames. All
  trajectories share ONE block length, chosen from the frames before labels are
  attached: `block_tau_factor` times the largest statistical inefficiency
  g = 1 + 2 sum rho_k over trajectories and descriptors (Sokal's tau_int is
  g/2), or `block_length_fs` when set, and at least `min_block_length_fs`;
  `md_windows_per_site` caps the number of blocks per trajectory (0 = no cap) by
  lengthening the common block. Each row reports its trajectory's g
  (`statistical_inefficiency_fs`, formerly `tau_int_fs`) and lag-1
  autocorrelation; the choice is recorded in
  `md_window_unique_dataset_metadata.json`. `group` = `material/site/seed_<n>`
  identifies the trajectory; the label is the starting site. Rows are
  pseudo-replicates of their trajectory: the trajectory is the statistical unit.
  With `md_frame_analysis.csv`, the non-feature columns `window_nearest_site`
  (majority), `window_nearest_site_fraction`, `window_start_site_fraction`,
  `window_desorbed_fraction` and `window_site_analysis_coverage` show whether
  the CO2 is still at its starting site.
- `md_trajectory_dataset.csv`: one row per independent trajectory (same site
  diagnostics over the whole production run).

The former `expanded` target multiplied TDDFT transitions, MD windows and
per-site DFT/CO2RR constants (960 rows from 3 calculations) and leaked the site
label; it was removed.

`qml_features` exports the ranked DFT table without zero imputation (rows with a
missing selected feature are dropped) and with the site label as the target.
