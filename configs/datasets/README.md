# Dataset configuration

`zno_existing_results.yaml` reads one simulation run (`input.results_dir`, default
`results/simulations/zno_ic_final`) and writes below `data/generated/`; it never writes
into `results/`. Commands: [docs/datasets.md](../../docs/datasets.md).

Relative input paths resolve below `input.results_dir`. The first-campaign results in
`results/_invalid_ic_v1/` used a wrong geometry and must not be used as input.

## Inputs (below the run directory)

| File | Use |
| --- | --- |
| `dft_summary.csv` | Required. Rows with `dft_status == ok`; exploratory site `carbonate` excluded. Band gaps from `*_band_gap_gamma_ev` (PBE Kohn-Sham, not comparable to the experimental 3.37 eV) |
| `tddft_summary.csv` | Optional. Rows with `tddft_status == ok`; see below |
| `co2rr_pathway_summary.csv` | Optional. Rows with `pathway_status == ok`; step energies from `dG_*` or `dE_*` according to `energy_basis`, stored as `feature_co2rr_d_<step>_ev` (basis in `co2rr_energy_basis`), plus `limiting_potential_v` |
| `md/<material>/<site>/md_production_frames.csv` | Required for `md_windows`. All seeds; columns `seed`, `phase`, `time_fs`, `c_surface_distance_ang`, `c_min_slab_distance_ang`, `o_min_metal_distance_ang`, `co_bond_avg_ang`, `oco_angle_deg`; only `phase == production` rows are used |
| `md/<material>/<site>/md_frame_analysis.csv` | Optional. Written by `python -m scripts.analysis.md_trajectory_analysis --results-dir <run>`; adds the window site diagnostics |

TDDFT columns read: `co2_isolated_first_excitation_ev`,
`co2_isolated_first_bright_excitation_ev` (lowest transition with f >= threshold, not an
absorption onset), `co2_isolated_peak_ev`, `co2_isolated_peak_f`,
`co2_isolated_ks_ionization_threshold_ev` (-ε_HOMO), `trk_sum` (total oscillator
strength).

- Older files: `co2_isolated_onset_ev` is read as
  `co2_isolated_first_bright_excitation_ev`; `co2_isolated_total_f` (truncated sum) is
  ignored.
- The values describe the isolated CO₂, are nearly equal for every site and are never
  classification features. The summary keeps them as `tddft_*` reference columns (no
  `feature_` prefix); the MD datasets do not contain them; `qml_features` offers no
  TDDFT column.
- The `tddft` method writes reference tables only (no `feature_*` columns, no site
  label), with `group` = `material/site`.

## Outputs of `multitech`

| File | Row |
| --- | --- |
| `multitech_selected_features_dataset.csv` | One successful DFT calculation; other methods joined one to one |
| `md_window_unique_dataset.csv` | One block of production frames |
| `md_trajectory_dataset.csv` | One trajectory (site diagnostics over the whole production run) |

Block length of `md_window_unique_dataset.csv` (one value for all trajectories, chosen
from the frames before labels are attached):

- `block_tau_factor` (2.0) × the largest statistical inefficiency
  g = 1 + 2 Σ ρ_k over trajectories and descriptors (Sokal's τ_int = g/2), or
  `block_length_fs` when set; at least `min_block_length_fs`.
- `md_windows_per_site` (10) caps the blocks per trajectory by lengthening the common
  block; 0 = no cap.
- The choice is recorded in `md_window_unique_dataset_metadata.json`. Each row carries
  its trajectory's g (`statistical_inefficiency_fs`, formerly `tau_int_fs`) and lag-1
  autocorrelation.
- `group` = `material/site/seed_<n>`; the label is the starting site. Rows are
  pseudo-replicates: the trajectory is the statistical unit.
- With `md_frame_analysis.csv`: non-feature columns `window_nearest_site` (majority),
  `window_nearest_site_fraction`, `window_start_site_fraction`,
  `window_desorbed_fraction`, `window_site_analysis_coverage`.

The former `expanded` target (cartesian product of TDDFT transitions, MD windows and
per-site constants, 960 rows from 3 calculations) leaked the site label and was
removed; the workflow rejects it.

## `qml_features`

Exports the ranked DFT table (`adsorption_energy_ev`, `adsorbed_band_gap_gamma_ev`)
with the site label as target. Rows with a missing feature are dropped (no zero
imputation).
