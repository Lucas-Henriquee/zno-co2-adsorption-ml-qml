# Dataset and feature reference

Columns written by `scripts/datasets/prepare_multitech_dataset.py` (method `multitech`
of `run_datasets.sh`). Inputs and options: [configs/datasets/README.md](../configs/datasets/README.md).
Feature sets used by the models: [configs/models/README.md](../configs/models/README.md).

## Datasets

| File (below `data/generated/<run-id>/multitech/`) | One row is | Use |
| --- | --- | --- |
| `multitech_selected_features_dataset.csv` | One successful DFT calculation (material, site) | Descriptor table; 3 rows for ZnO, too few for classification |
| `md_window_unique_dataset.csv` | One block of production MD frames of one trajectory | Classification input (`configs/models/zno_local.yaml`) |
| `md_trajectory_dataset.csv` | One MD trajectory (all production frames) | Same descriptors as the windows, over the whole trajectory |

Each file has `<name>_feature_columns.txt` and `<name>_label_columns.txt` next to it;
the MD-window dataset also has `md_window_unique_dataset_metadata.json` (block length,
statistical inefficiency of every trajectory).

## Identifiers and labels

| Column | Meaning |
| --- | --- |
| `material`, `site` | Simulated case; `site` is the starting site of the DFT relaxation or MD trajectory |
| `group` | Independent unit: `material/site` (summary) or `material/site/seed_<n>` (MD). Train/test splits and CV folds keep a group on one side |
| `seed` | MD seed of the trajectory |
| `md_window_id` | Block index within the trajectory (0, 1, …) |
| `label_site` | Starting site |
| `label_site_id` | `top_metal` 0, `top_oxygen` 1, `bridge` 2 |

The label is the starting site, not the site occupied during the block: the
physisorbed CO₂ moves away from it during the run.

## MD features (windows and trajectories)

Means over the frames of the block (window dataset) or of the whole production run
(trajectory dataset). All distances use the minimum-image convention.

| Column | Meaning / unit |
| --- | --- |
| `feature_md_window_c_surface_distance_mean_a` | Height of C above the topmost slab atom, mean / Å |
| `feature_md_window_c_surface_distance_std_a` | Same height, standard deviation (ddof = 1) / Å |
| `feature_md_window_c_min_slab_distance_mean_a` | Shortest C–slab-atom distance, mean / Å |
| `feature_md_window_o_min_metal_distance_mean_a` | Shortest O(CO₂)–Zn distance, mean / Å |
| `feature_md_window_co_bond_avg_mean_a` | Mean of the two C–O bond lengths / Å |
| `feature_md_window_mean_oco_angle_deg` | O–C–O angle, mean / degrees |

Model feature sets (`scripts/models/benchmarks/run_benchmarks.py`):

- `md3`: C height mean and standard deviation, O–C–O angle.
- `md_mic5`: shortest C–slab and O–Zn distances, C height, C–O bond length, O–C–O angle.

Temperature is not a feature.

## MD diagnostics (not features)

| Column | Meaning |
| --- | --- |
| `window_start_fs`, `window_end_fs`, `window_n_frames` | Time span and size of the block |
| `block_length_fs` | Common block length of the dataset |
| `statistical_inefficiency_fs` | g = 1 + 2 Σ ρ_k of the trajectory, largest over the descriptors, in fs |
| `lag1_autocorrelation` | ρ₁ of the descriptor that sets g |
| `window_nearest_site`, `window_nearest_site_fraction` | Majority nearest site of the frames and its fraction |
| `window_start_site_fraction` | Fraction of frames whose nearest site is the starting site |
| `window_desorbed_fraction` | Fraction of frames with the CO₂ beyond the desorption threshold |
| `window_site_analysis_coverage` | Fraction of frames with a site assignment (`md_frame_analysis.csv`) |
| `n_production_frames`, `production_length_fs` | Trajectory dataset only |

They would leak the label if used as features. No window is filtered by them.

## Summary columns

| Column | Meaning / unit |
| --- | --- |
| `feature_dft_adsorption_energy_ev` | E_ads / eV (PBE, no dispersion) |
| `feature_dft_c_surface_distance_a` | Height of C above the topmost slab atom / Å |
| `feature_dft_c_min_slab_distance_a` | Shortest C–slab-atom distance / Å |
| `feature_dft_o_min_metal_distance_a` | Shortest O(CO₂)–Zn distance / Å |
| `feature_dft_oco_angle_deg` | O–C–O angle / degrees |
| `feature_dft_pbe_gap_change_ev` | Adsorbed minus clean PBE Kohn-Sham gap at Γ / eV |
| `feature_md_c_surface_distance_mean_a`, `_std_a`, `feature_md_mean_oco_angle_deg` | Over all production frames of all seeds |
| `md_n_trajectories` | Trajectories behind the MD columns |
| `feature_co2rr_d_<step>_ev` | Step energy of the run summary (variant a), ΔG or ΔE according to `co2rr_energy_basis` / eV |
| `feature_co2rr_limiting_potential_v` | U_L / V; blank when `co2rr_pathway_validity` is not `ok` |
| `co2rr_pathway_validity` | `ok`, `unbound` (COOH* or CO* desorbed) or `wrong_intermediate` |
| `tddft_co2_isolated_*`, `tddft_trk_sum` | Isolated CO₂ references (first excitation, first bright excitation, peak, −ε_HOMO, TRK sum); not features |
| `dft_final_state` | `physisorbed` or `chemisorbed` |

The paper reports the corrected CO₂RR variant c (Table 2), not these values.

## Run `zno_ic_final`

- 3 sites × 3 seeds = 9 trajectories, 76 production frames each (20 fs apart).
- Largest g = 27.7 frames (`top_oxygen/seed_1234`, shortest C–slab distance), so the
  block is 2 × 27.7 → 56 frames (1120 fs) and each trajectory gives one block,
  0–1100 fs. The window dataset has 9 rows and leaves out the last 20 frames of each
  trajectory; the trajectory dataset has the same 9 rows over all 76 frames.
- The sample size of any model is 9 trajectories, 3 per class.
