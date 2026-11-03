# Dataset and feature reference

> Out of date: describes the first-campaign datasets, including TDDFT and CO₂RR
> feature groups that are no longer ML features (TDDFT is a reference only; the CO₂RR
> columns are now free energies). Current sources:
> `scripts/datasets/prepare_multitech_dataset.py`,
> [configs/datasets/README.md](../configs/datasets/README.md) and the feature sets in
> [configs/models/README.md](../configs/models/README.md). To be rewritten with the
> ML/QML review.

Source definitions: `scripts/datasets/prepare_multitech_dataset.py` and
`scripts/models/benchmarks/run_benchmarks.py`. These names describe locally
constructed data and feature subsets, not externally standardized datasets.

| Dataset under a generated run | Meaning of a row |
| --- | --- |
| `multitech/multitech_selected_features_dataset.csv` | One combined material/site summary |
| `multitech/multitech_expanded_ml_dataset_clean.csv` | A TDDFT transition combined with an MD window for the same material/site |
| `multitech/md_window_unique_dataset.csv` | One distinct MD window for a material/site |

`material` and `site` identify the simulated case. `md_window_id` identifies a
window within its site's trajectory. `label_site` is the site name;
`label_site_id` maps `top_metal` to 0, `top_oxygen` to 1, and `bridge` to 2.
The configured window count is ten per site in the current dataset YAML.
Expanded combinations are not independent simulations. Class labels identify
sites; they are not labels of catalytic quality.

## Feature groups

`static6` contains these six columns:

| Column | Meaning / unit |
| --- | --- |
| `feature_dft_adsorption_energy_ev` | DFT adsorption energy / eV |
| `feature_dft_c_surface_distance_a` | Carbon-to-surface distance / Å |
| `feature_dft_gap_change_ev` | Adsorbed minus clean-slab band gap / eV |
| `feature_co2rr_deltaG_CO2_to_COOH_ev` | Electronic reaction-energy estimate, CO₂* to COOH* / eV |
| `feature_co2rr_deltaG_COOH_to_CO_ev` | Electronic reaction-energy estimate, COOH* to CO* / eV |
| `feature_co2rr_limiting_potential_v` | Historical column name for the largest non-negative reaction-energy estimate |

The preserved reaction routine uses electronic energies without zero-point or
thermal corrections. The `limiting_potential_v` column retains its original
name and values; the implemented maximum is an energy descriptor, not a
rigorous electrochemical limiting potential.

`md3` contains `feature_md_window_c_surface_distance_mean_a`,
`feature_md_window_c_surface_distance_std_a`, and
`feature_md_window_mean_oco_angle_deg`: window distance mean and standard
deviation in Å, and mean O–C–O angle in degrees.

`tddft3` contains `feature_tddft_transition_energy_ev`,
`feature_tddft_transition_oscillator_strength`, and
`feature_tddft_transition_delta_from_peak_ev`: transition energy in eV,
dimensionless oscillator-strength field, and signed transition-minus-peak
energy in eV. Interpret these fields using the exact TDDFT source version that
generated the selected campaign; historical extraction behavior is preserved.

`dynamic6` joins `tddft3` and `md3`. `dynamic4` keeps transition energy,
oscillator strength, window mean distance, and window distance standard deviation.
`all12` joins `static6` and `dynamic6`. The supplied local-model YAML selects
`md3`; no feature group was changed by this documentation.

For the paper appendix, choose the actual submitted dataset before inserting
five to ten example rows and the final row. Include the complete selected data
in the supplement with its hash, generation command, and feature/label columns.
No example values or manuscript row counts are invented here.
