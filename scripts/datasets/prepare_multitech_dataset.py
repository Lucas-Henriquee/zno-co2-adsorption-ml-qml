"""Build classification datasets from DFT, TDDFT, CO2RR and MD outputs of one run.

No row is duplicated by a join (audit findings D1 and M3):

* ``multitech_selected_features_dataset.csv`` (target ``summary``): one row per
  successful DFT calculation (material, site). The other methods are joined one to
  one; nothing is multiplied. It is a descriptor table, too small for classification.
  The TDDFT descriptors of the isolated CO2 molecule are kept there only as
  ``tddft_co2_isolated_*`` reference columns, NOT as ``feature_*`` columns: the
  physisorbed CO2 has nearly the same geometry on every site, so they carry no site
  information.
* ``md_window_unique_dataset.csv`` (target ``md_windows``): one row per block of
  production MD frames, with ``group`` = material/site/seed, the id of its
  trajectory. Every trajectory uses ONE common block length, chosen from the
  frames alone before any label is attached (``choose_common_block_length``) and
  recorded in ``md_window_unique_dataset_metadata.json``.
* ``md_trajectory_dataset.csv`` (written with ``md_windows``): one row per
  independent MD trajectory.

Statistical unit. The blocks of one trajectory are pseudo-replicates: block means
a few statistical inefficiencies apart are nearly uncorrelated, but they share the
initial structure and the starting site of their trajectory. The independent unit
is the trajectory (``group``); validation must split by group, and the number of
groups, not the number of rows, is the sample size.

Site occupancy. When ``md/<material>/<site>/md_frame_analysis.csv`` (written by
scripts/analysis/md_trajectory_analysis.py) exists, every window also gets the
majority nearest site of its frames (``window_nearest_site``) and its fraction, the
fraction of frames still at the starting site and the fraction of desorbed frames.
They are diagnostics, not features (they would leak the label): the label stays
the starting site, and windows whose CO2 migrated or desorbed must be filtered or
discussed.

The former ``expanded`` target (TDDFT transitions x MD windows x per-site DFT and
CO2RR constants) was a cartesian product that leaked the site label and is removed.
The exploratory DFT sites (``carbonate``, ``tridentate``) are excluded from every dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.md_inputs import find_md_summaries
from scripts.common.project_config import EXPLORATORY_SITES, RESULTS_DIR

# Legacy option: now the maximum number of blocks per trajectory (0 = no cap).
DEFAULT_MD_WINDOWS_PER_SITE = 10
# Common block length = BLOCK_TAU_FACTOR x the largest statistical inefficiency
# g = 1 + 2 sum rho_k (in frames) over trajectories and descriptors. In Sokal's
# convention the integrated autocorrelation time is tau_int = g / 2. Means of
# blocks a few g long are approximately independent.
DEFAULT_BLOCK_TAU_FACTOR = 2.0

SITE_LABELS = {
    "top_metal": 0,
    "top_oxygen": 1,
    "bridge": 2,
}
EXCLUDED_SITES = frozenset(EXPLORATORY_SITES)

PRODUCTION_PHASE = "production"

# Standard MD descriptor -> accepted column names in the per-frame CSV.
MD_DESCRIPTOR_ALIASES = {
    "c_surface_distance_ang": ("c_surface_distance_ang", "c_height_above_slab_plane_ang"),
    "c_min_slab_distance_ang": ("c_min_slab_distance_ang",),
    "o_min_metal_distance_ang": ("o_min_metal_distance_ang",),
    "co_bond_avg_ang": ("co_bond_avg_ang",),
    "oco_angle_deg": ("oco_angle_deg",),
}
REQUIRED_MD_DESCRIPTORS = ("c_surface_distance_ang", "oco_angle_deg")
MD_WINDOW_FEATURES = {
    "c_surface_distance_ang": "feature_md_window_c_surface_distance_mean_a",
    "c_min_slab_distance_ang": "feature_md_window_c_min_slab_distance_mean_a",
    "o_min_metal_distance_ang": "feature_md_window_o_min_metal_distance_mean_a",
    "co_bond_avg_ang": "feature_md_window_co_bond_avg_mean_a",
    "oco_angle_deg": "feature_md_window_mean_oco_angle_deg",
}
MD_WINDOW_STD_FEATURE = "feature_md_window_c_surface_distance_std_a"

# TDDFT descriptors of the isolated CO2 molecule (in its adsorbed geometry). They
# are not excitations of the ZnO/CO2 system and must not be compared with a
# photocatalyst absorption target. They are (almost) constant across sites, so
# they are reference columns of the summary table only, never classification
# features (no ``feature_`` prefix), and never enter the MD datasets.
TDDFT_COLUMNS = {
    "co2_isolated_first_excitation_ev": "tddft_co2_isolated_first_excitation_ev",
    "co2_isolated_onset_ev": "tddft_co2_isolated_onset_ev",
    "co2_isolated_peak_ev": "tddft_co2_isolated_peak_ev",
    "co2_isolated_total_f": "tddft_co2_isolated_total_f",
}

# Per-frame site analysis written by scripts/analysis/md_trajectory_analysis.py.
FRAME_ANALYSIS_FILENAME = "md_frame_analysis.csv"
WINDOW_SITE_COLUMNS = [
    "window_nearest_site",
    "window_nearest_site_fraction",
    "window_start_site_fraction",
    "window_desorbed_fraction",
    "window_site_analysis_coverage",
]
BLOCK_METADATA_SUFFIX = "_metadata.json"

SUMMARY_COLUMNS = [
    "material",
    "site",
    "group",
    "feature_dft_adsorption_energy_ev",
    "feature_dft_c_surface_distance_a",
    "feature_dft_c_min_slab_distance_a",
    "feature_dft_o_min_metal_distance_a",
    "feature_dft_oco_angle_deg",
    "feature_dft_pbe_gap_change_ev",
    *TDDFT_COLUMNS.values(),
    "feature_md_c_surface_distance_mean_a",
    "feature_md_c_surface_distance_std_a",
    "feature_md_mean_oco_angle_deg",
    "md_n_trajectories",
    "label_site",
    "label_site_id",
]

MD_UNIQUE_COLUMNS = [
    "material",
    "site",
    "group",
    "seed",
    "md_window_id",
    "window_start_fs",
    "window_end_fs",
    "window_n_frames",
    "block_length_fs",
    "statistical_inefficiency_fs",
    "lag1_autocorrelation",
    *MD_WINDOW_FEATURES.values(),
    MD_WINDOW_STD_FEATURE,
    *WINDOW_SITE_COLUMNS,
    "label_site",
    "label_site_id",
]

MD_TRAJECTORY_COLUMNS = [
    "material",
    "site",
    "group",
    "seed",
    "n_production_frames",
    "production_length_fs",
    "statistical_inefficiency_fs",
    "lag1_autocorrelation",
    *MD_WINDOW_FEATURES.values(),
    MD_WINDOW_STD_FEATURE,
    *WINDOW_SITE_COLUMNS,
    "label_site",
    "label_site_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build datasets with one row per independent DFT calculation or MD block."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Simulation run directory, e.g. results/simulations/<run-id> (default: QML_ZNO_RESULTS_DIR).",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "datasets")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["summary", "expanded", "md_windows"],
        default=["summary", "md_windows"],
        help="'expanded' was removed (cartesian product, label leakage) and is rejected.",
    )
    parser.add_argument(
        "--md-windows-per-site",
        "--max-windows-per-trajectory",
        dest="md_windows_per_site",
        type=int,
        default=DEFAULT_MD_WINDOWS_PER_SITE,
        help="Maximum number of blocks per MD trajectory; 0 = as many as the correlation time allows.",
    )
    parser.add_argument(
        "--block-tau-factor",
        type=float,
        default=DEFAULT_BLOCK_TAU_FACTOR,
        help=(
            "Common block length in units of the largest statistical inefficiency "
            "g = 1 + 2 sum rho_k over all trajectories and descriptors."
        ),
    )
    parser.add_argument(
        "--block-length-fs",
        type=float,
        default=None,
        help="Fixed common block length in fs for every trajectory (replaces the g-based length).",
    )
    parser.add_argument(
        "--min-block-length-fs",
        type=float,
        default=0.0,
        help="Lower bound on the block length in fs.",
    )
    parser.add_argument(
        "--equilibration-fs",
        type=float,
        default=0.0,
        help="Frames before this time are discarded, only for MD files without a 'phase' column.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}. Available: {list(df.columns)}")


def normalize_site(site: str) -> str:
    aliases = {
        "topMetal": "top_metal",
        "top_metal": "top_metal",
        "topOxygen": "top_oxygen",
        "top_oxygen": "top_oxygen",
        "bridge": "bridge",
    }
    return aliases.get(str(site), str(site))


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = df.copy()
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def keep_ok(df: pd.DataFrame, status_column: str) -> pd.DataFrame:
    """Keep rows whose status is 'ok'. Tables without a status column are kept whole."""
    if status_column not in df.columns:
        return df.copy()
    return df[df[status_column].astype(str).str.strip().str.lower() == "ok"].copy()


def drop_excluded_sites(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["site"] = data["site"].map(normalize_site)
    return data[~data["site"].isin(EXCLUDED_SITES)].copy()


def add_site_labels(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["label_site"] = data["site"]
    data["label_site_id"] = data["site"].map(SITE_LABELS)

    if data["label_site_id"].isna().any():
        unknown = sorted(data.loc[data["label_site_id"].isna(), "site"].unique())
        raise ValueError(f"Unknown site names: {unknown}")

    data["label_site_id"] = data["label_site_id"].astype(int)
    return data


def save_dataset(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    data = df[[col for col in columns if col in df.columns]].copy()
    data.to_csv(path, index=False)

    feature_cols = [col for col in data.columns if col.startswith("feature_")]
    label_cols = [col for col in data.columns if col.startswith("label_")]

    (path.parent / f"{path.stem}_feature_columns.txt").write_text(
        "\n".join(feature_cols) + "\n",
        encoding="utf-8",
    )
    (path.parent / f"{path.stem}_label_columns.txt").write_text(
        "\n".join(label_cols) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved: {path}")
    print(f"Samples: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Labels: {len(label_cols)}")
    if "group" in data.columns:
        print(f"Groups (independent trajectories or calculations): {data['group'].nunique()}")
        print(data.groupby("label_site")["group"].nunique().rename("groups").sort_index())
    print(data["label_site"].value_counts().sort_index())


# --------------------------------------------------------------------------- DFT


def load_dft_features(results_dir: Path) -> pd.DataFrame:
    """One row per successful DFT calculation, without the exploratory sites.

    Band gaps are PBE Kohn-Sham gaps on a Gamma-centred grid (*_band_gap_gamma_ev).
    They underestimate the experimental ZnO gap (3.37 eV) and are used only as a
    gap change on CO2 adsorption, never compared with an absorption target.
    """
    path = require_file(results_dir / "dft_summary.csv")
    df = pd.read_csv(path)

    require_columns(
        df,
        [
            "material",
            "site",
            "dft_status",
            "adsorption_energy_ev",
            "c_surface_distance_ang",
            "clean_band_gap_gamma_ev",
            "adsorbed_band_gap_gamma_ev",
        ],
        "DFT summary",
    )
    df = drop_excluded_sites(keep_ok(df, "dft_status"))

    optional = [
        col
        for col in ["c_min_slab_distance_ang", "o_min_metal_distance_ang", "oco_angle_deg"]
        if col in df.columns
    ]
    cols = [
        "adsorption_energy_ev",
        "c_surface_distance_ang",
        "clean_band_gap_gamma_ev",
        "adsorbed_band_gap_gamma_ev",
        *optional,
    ]
    df = numeric(df, cols)

    out = df[["material", "site"]].copy()
    out["feature_dft_adsorption_energy_ev"] = df["adsorption_energy_ev"]
    out["feature_dft_c_surface_distance_a"] = df["c_surface_distance_ang"]
    if "c_min_slab_distance_ang" in df.columns:
        out["feature_dft_c_min_slab_distance_a"] = df["c_min_slab_distance_ang"]
    if "o_min_metal_distance_ang" in df.columns:
        out["feature_dft_o_min_metal_distance_a"] = df["o_min_metal_distance_ang"]
    if "oco_angle_deg" in df.columns:
        out["feature_dft_oco_angle_deg"] = df["oco_angle_deg"]
    out["feature_dft_pbe_gap_change_ev"] = (
        df["adsorbed_band_gap_gamma_ev"] - df["clean_band_gap_gamma_ev"]
    )
    if "final_state" in df.columns:
        out["dft_final_state"] = df["final_state"]
    return out


def load_tddft_summary(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "tddft_summary.csv"
    if not path.is_file():
        print(f"[dataset] No TDDFT summary at {path}; TDDFT descriptors omitted.")
        return None
    df = pd.read_csv(path)
    require_columns(df, ["material", "site", *TDDFT_COLUMNS], "TDDFT summary")
    df = drop_excluded_sites(keep_ok(df, "tddft_status"))
    df = numeric(df, list(TDDFT_COLUMNS))

    out = df[["material", "site"]].copy()
    for source, feature in TDDFT_COLUMNS.items():
        out[feature] = df[source]
    return out


CO2RR_STEPS = ("CO2_adsorption", "CO2_to_COOH", "COOH_to_CO", "CO_desorption", "overall")
CO2RR_PREFIX = {"free_energy_CHE": "dG_", "electronic_energies_only": "dE_"}


def load_co2rr_features(results_dir: Path) -> pd.DataFrame | None:
    """Reaction-step energies and limiting potential, rows with pathway_status == ok.

    Each row is read from its dG_* or dE_* columns according to ``energy_basis``;
    the features carry a neutral name (feature_co2rr_d_<step>_ev) and the basis is
    kept in ``co2rr_energy_basis``, so a dE is never presented as a dG.
    """
    path = results_dir / "co2rr_pathway_summary.csv"
    if not path.is_file():
        print(f"[dataset] No CO2RR summary at {path}; CO2RR descriptors omitted.")
        return None
    df = pd.read_csv(path)
    require_columns(df, ["material", "site", "pathway_status", "energy_basis"], "CO2RR summary")
    df = drop_excluded_sites(keep_ok(df, "pathway_status"))
    unknown = sorted(set(df["energy_basis"].astype(str)) - set(CO2RR_PREFIX))
    if unknown:
        raise ValueError(f"CO2RR summary: unknown energy_basis values {unknown}")

    out = df[["material", "site"]].copy()
    out["co2rr_energy_basis"] = df["energy_basis"].astype(str)
    for step in CO2RR_STEPS:
        values = []
        for _, row in df.iterrows():
            column = f"{CO2RR_PREFIX[str(row['energy_basis'])]}{step}_ev"
            values.append(pd.to_numeric(row.get(column), errors="coerce"))
        out[f"feature_co2rr_d_{step}_ev"] = values
    if "limiting_potential_v" in df.columns:
        out["feature_co2rr_limiting_potential_v"] = pd.to_numeric(df["limiting_potential_v"], errors="coerce")
    return out


# ---------------------------------------------------------------------------- MD


def find_md_trajectory_files(results_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (material, site, path) for md/<material>/<site>/md_production_frames.csv.

    Each site-level table holds the production frames of every seed; the seed
    column separates the independent trajectories. seed_<n>/ folders are not read.
    """
    return [
        (path.parent.parent.name, normalize_site(path.parent.name), path)
        for path in find_md_summaries(results_dir)
    ]


def _resolve_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "co_bond_avg_ang" not in data.columns and {"co_bond_1_ang", "co_bond_2_ang"}.issubset(data.columns):
        data["co_bond_avg_ang"] = (
            pd.to_numeric(data["co_bond_1_ang"], errors="coerce")
            + pd.to_numeric(data["co_bond_2_ang"], errors="coerce")
        ) / 2.0
    for standard, aliases in MD_DESCRIPTOR_ALIASES.items():
        if standard in data.columns:
            continue
        for alias in aliases:
            if alias in data.columns:
                data[standard] = data[alias]
                break
    return data


def _time_fs(df: pd.DataFrame, name: str) -> pd.Series:
    if "time_fs" in df.columns:
        return pd.to_numeric(df["time_fs"], errors="coerce")
    raise ValueError(f"{name}: needs a 'time_fs' column.")


def _seed_label(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _time_key(times: pd.Series) -> pd.Series:
    # Times are multiples of log_interval x timestep; rounding absorbs CSV round-off.
    return pd.to_numeric(times, errors="coerce").round(3)


def load_frame_site_analysis(site_dir: Path) -> pd.DataFrame | None:
    """Production rows of md_frame_analysis.csv (seed, time key, nearest site, desorbed)."""
    path = site_dir / FRAME_ANALYSIS_FILENAME
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    require_columns(df, ["seed", "phase", "time_fs", "nearest_site", "desorbed"], f"MD frame analysis {path}")
    df = df[df["phase"].astype(str).str.strip().str.lower() == PRODUCTION_PHASE].copy()
    out = pd.DataFrame({
        "_seed": df["seed"].map(_seed_label),
        "_time_key": _time_key(df["time_fs"]),
        "nearest_site": df["nearest_site"].astype(str).map(normalize_site),
        "desorbed": pd.to_numeric(df["desorbed"], errors="coerce"),
    })
    return out.drop_duplicates(subset=["_seed", "_time_key"])


def _attach_site_analysis(frames: pd.DataFrame, analysis: pd.DataFrame | None, seed) -> pd.DataFrame:
    if analysis is None:
        return frames
    selected = analysis if seed is None else analysis[analysis["_seed"] == str(seed)]
    selected = selected.drop(columns="_seed").drop_duplicates(subset="_time_key")
    merged = frames.assign(_time_key=_time_key(frames["time_fs"])).merge(
        selected, on="_time_key", how="left", validate="one_to_one"
    )
    return merged.drop(columns="_time_key")


def load_md_trajectories(
    results_dir: Path,
    equilibration_fs: float = 0.0,
) -> list[dict]:
    """Read every MD trajectory, keeping only production frames.

    Returns one dict per trajectory (one seed) with material, site, seed,
    group = material/site/seed_<seed> and a frame table sorted by time. Rows with
    ``phase`` != production are dropped; without a ``phase`` column, frames before
    ``equilibration_fs`` are discarded instead. A table without a ``seed`` column
    is treated as a single trajectory with group = material/site. When the site
    folder has md_frame_analysis.csv, its ``nearest_site`` and ``desorbed``
    columns are joined frame by frame (seed, time_fs); frames not analysed yet
    keep NaN.
    """
    trajectories: list[dict] = []
    for material, site, path in find_md_trajectory_files(results_dir):
        if site in EXCLUDED_SITES:
            continue
        df = _resolve_descriptors(pd.read_csv(path))
        require_columns(df, list(REQUIRED_MD_DESCRIPTORS), f"MD summary {path}")
        df["time_fs"] = _time_fs(df, f"MD summary {path}")
        # The table's own copies would collide with the frame analysis columns.
        df = df.drop(columns=[col for col in ("nearest_site", "desorbed") if col in df.columns])
        site_analysis = load_frame_site_analysis(path.parent)

        if "phase" in df.columns:
            df = df[df["phase"].astype(str).str.strip().str.lower() == PRODUCTION_PHASE]
        else:
            print(
                f"[dataset] {path} has no 'phase' column; discarding frames before "
                f"{equilibration_fs:g} fs as equilibration."
            )
            df = df[df["time_fs"] >= equilibration_fs]

        descriptors = [name for name in MD_DESCRIPTOR_ALIASES if name in df.columns]
        df = numeric(df, descriptors)

        if "seed" in df.columns and df["seed"].notna().any():
            seeds = df["seed"].map(_seed_label)
            parts = [(seed, part) for seed, part in df.groupby(seeds, sort=True)]
        else:
            parts = [(None, df)]

        for seed, frames in parts:
            frames = frames.dropna(subset=["time_fs"]).sort_values("time_fs")
            frames = frames.drop_duplicates(subset="time_fs").reset_index(drop=True)
            frames = _attach_site_analysis(frames, site_analysis, seed)
            if site_analysis is not None and len(frames):
                coverage = float(frames["nearest_site"].notna().mean())
                if coverage < 1.0:
                    print(
                        f"[dataset] {path.parent / FRAME_ANALYSIS_FILENAME} seed={seed}: site analysis "
                        f"covers {coverage:.0%} of the production frames; rerun md_trajectory_analysis."
                    )
            if frames.empty:
                print(f"[dataset] {path} seed={seed}: no production frames; skipped.")
                continue
            group = f"{material}/{site}" if seed is None else f"{material}/{site}/seed_{seed}"
            trajectories.append(
                {
                    "material": material,
                    "site": site,
                    "seed": seed,
                    "group": group,
                    "path": path,
                    "descriptors": descriptors,
                    "frames": frames,
                }
            )

    groups = [item["group"] for item in trajectories]
    duplicated = sorted({name for name in groups if groups.count(name) > 1})
    if duplicated:
        raise ValueError(f"Duplicated MD trajectory ids: {duplicated}")
    return trajectories


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation rho_k for k = 0..max_lag (biased estimator)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.array([1.0])
    mean = float(x.mean())
    x = x - mean
    variance = float(np.dot(x, x)) / n
    # A constant descriptor (up to round-off) carries no correlation information.
    if variance <= 1e-12 * max(1.0, mean * mean):
        return np.array([1.0])
    max_lag = min(max_lag, n - 1)
    return np.array([float(np.dot(x[: n - k], x[k:])) / n / variance for k in range(max_lag + 1)])


def statistical_inefficiency(values: np.ndarray) -> tuple[float, float]:
    """Return (statistical inefficiency g in frames, lag-1 autocorrelation).

    g = 1 + 2 sum_k rho_k is the number of consecutive frames worth one
    independent sample (the variance of a mean of N frames is g sigma^2 / N). In
    Sokal's convention the integrated autocorrelation time is tau_int = g / 2;
    this module always reports g. The sum stops at the first non-positive rho_k
    (initial positive sequence), at most at N/2.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 1.0, float("nan")
    rho = autocorrelation(x, max_lag=len(x) // 2)
    lag1 = float(rho[1]) if len(rho) > 1 else float("nan")
    g = 1.0
    for value in rho[1:]:
        if value <= 0.0:
            break
        g += 2.0 * float(value)
    return g, lag1


# Former name: the quantity is the statistical inefficiency g, not Sokal's tau_int.
integrated_autocorrelation_time = statistical_inefficiency


def trajectory_correlation(frames: pd.DataFrame, descriptors: list[str]) -> dict:
    """Largest statistical inefficiency g over the descriptors (in frames and fs)."""
    times = frames["time_fs"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else float("nan")
    g_frames, lag1, driver = 1.0, float("nan"), None
    for name in descriptors:
        g, rho1 = statistical_inefficiency(frames[name].to_numpy())
        if driver is None or g > g_frames:
            g_frames, lag1, driver = g, rho1, name
    return {
        "timestep_fs": dt,
        "statistical_inefficiency_frames": g_frames,
        "statistical_inefficiency_fs": g_frames * dt if math.isfinite(dt) else float("nan"),
        "lag1_autocorrelation": lag1,
        "g_descriptor": driver,
    }


def choose_common_block_length(
    trajectories: list[dict],
    max_windows: int = DEFAULT_MD_WINDOWS_PER_SITE,
    block_tau_factor: float = DEFAULT_BLOCK_TAU_FACTOR,
    min_block_length_fs: float = 0.0,
    block_length_fs: float | None = None,
) -> dict:
    """Choose ONE block length for every trajectory, from the frames alone.

    It is fixed before any label is attached and is the same for every site, so
    the block size cannot differ between classes. Default: ``block_tau_factor``
    times the largest statistical inefficiency g over all trajectories and
    descriptors; ``block_length_fs`` replaces it with a fixed value. The length
    is then raised to ``min_block_length_fs`` and, when ``max_windows`` > 0, so
    that no trajectory yields more than ``max_windows`` blocks. Every trajectory
    must have the same frame interval.
    """
    if not trajectories:
        raise ValueError("No MD trajectories to choose a block length from.")
    correlations = {
        item["group"]: trajectory_correlation(item["frames"], item["descriptors"])
        for item in trajectories
    }
    steps = sorted({
        round(value["timestep_fs"], 6)
        for value in correlations.values()
        if math.isfinite(value["timestep_fs"])
    })
    if len(steps) > 1:
        raise ValueError(
            f"MD trajectories have different frame intervals {steps} fs; a common block "
            "length in frames needs one interval (check log_interval and timestep_fs)."
        )
    dt = steps[0] if steps else float("nan")
    driver_group = max(correlations, key=lambda group: correlations[group]["statistical_inefficiency_frames"])
    g_max = correlations[driver_group]["statistical_inefficiency_frames"]

    if block_length_fs is not None:
        if block_length_fs <= 0:
            raise ValueError("block_length_fs must be positive.")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("A fixed block length in fs needs trajectories with at least two frames.")
        length = max(int(round(block_length_fs / dt)), 1)
        source = "fixed_block_length_fs"
    else:
        length = max(math.ceil(block_tau_factor * g_max), 1)
        source = "block_tau_factor_x_max_statistical_inefficiency"
    if min_block_length_fs > 0 and math.isfinite(dt) and dt > 0:
        length = max(length, math.ceil(min_block_length_fs / dt))
    capped = False
    if max_windows > 0:
        for item in trajectories:
            n_frames = len(item["frames"])
            if n_frames // length > max_windows:
                # Fewer, longer blocks for every trajectory: never shorter than above.
                length = max(length, n_frames // max_windows)
                capped = True
        for item in trajectories:  # n // (n // m) can still exceed m for small n.
            if len(item["frames"]) // length > max_windows:
                length = math.ceil(len(item["frames"]) / max_windows)

    short = sorted(item["group"] for item in trajectories if len(item["frames"]) < length)
    return {
        "block_length_frames": int(length),
        "block_length_fs": length * dt if math.isfinite(dt) else float("nan"),
        "frame_interval_fs": dt,
        "source": source,
        "requested_block_length_fs": block_length_fs,
        "block_tau_factor": block_tau_factor,
        "min_block_length_fs": min_block_length_fs,
        "max_windows_per_trajectory": max_windows,
        "capped_by_max_windows": capped,
        "max_statistical_inefficiency_frames": g_max,
        "max_statistical_inefficiency_fs": correlations[driver_group]["statistical_inefficiency_fs"],
        "max_statistical_inefficiency_group": driver_group,
        "max_statistical_inefficiency_descriptor": correlations[driver_group]["g_descriptor"],
        "statistical_inefficiency_definition": "g = 1 + 2 sum_k rho_k (Sokal tau_int = g / 2)",
        "trajectories_shorter_than_block": short,
        "per_trajectory": {
            group: {
                "n_production_frames": int(len(item["frames"])),
                "statistical_inefficiency_frames": correlations[group]["statistical_inefficiency_frames"],
                "statistical_inefficiency_fs": correlations[group]["statistical_inefficiency_fs"],
                "lag1_autocorrelation": correlations[group]["lag1_autocorrelation"],
                "g_descriptor": correlations[group]["g_descriptor"],
            }
            for group, item in ((entry["group"], entry) for entry in trajectories)
        },
    }


def _site_fractions(trajectory: dict, window: pd.DataFrame) -> dict:
    """Nearest-site diagnostics of a window (NaN when no frame has been analysed)."""
    if "nearest_site" not in window.columns:
        return {}
    sites = window["nearest_site"].dropna().astype(str)
    sites = sites[sites.str.lower() != "nan"]
    row = {name: float("nan") for name in WINDOW_SITE_COLUMNS}
    row["window_nearest_site"] = ""
    row["window_site_analysis_coverage"] = float(len(sites) / len(window)) if len(window) else float("nan")
    if len(sites):
        counts = sites.value_counts()
        row["window_nearest_site"] = str(counts.index[0])
        row["window_nearest_site_fraction"] = float(counts.iloc[0] / len(sites))
        row["window_start_site_fraction"] = float((sites == trajectory["site"]).mean())
    if "desorbed" in window.columns:
        desorbed = pd.to_numeric(window["desorbed"], errors="coerce").dropna()
        if len(desorbed):
            row["window_desorbed_fraction"] = float(desorbed.mean())
    return row


def _window_row(trajectory: dict, window: pd.DataFrame, window_id: int) -> dict:
    row = {
        "material": trajectory["material"],
        "site": trajectory["site"],
        "group": trajectory["group"],
        "seed": trajectory["seed"],
        "md_window_id": int(window_id),
        "window_start_fs": float(window["time_fs"].iloc[0]),
        "window_end_fs": float(window["time_fs"].iloc[-1]),
        "window_n_frames": int(len(window)),
    }
    for name in trajectory["descriptors"]:
        row[MD_WINDOW_FEATURES[name]] = float(window[name].mean())
    # Single-frame windows have no spread; NaN is kept rather than imputed.
    row[MD_WINDOW_STD_FEATURE] = float(window["c_surface_distance_ang"].std(ddof=1))
    row.update(_site_fractions(trajectory, window))
    return row


def load_md_windows(
    results_dir: Path,
    n_windows: int = DEFAULT_MD_WINDOWS_PER_SITE,
    block_tau_factor: float = DEFAULT_BLOCK_TAU_FACTOR,
    min_block_length_fs: float = 0.0,
    equilibration_fs: float = 0.0,
    block_length_fs: float | None = None,
) -> pd.DataFrame:
    """One row per block of production frames, with the trajectory id in ``group``.

    Blocks are contiguous and non-overlapping, and every trajectory uses the same
    block length (``choose_common_block_length``, chosen before labels exist);
    ``n_windows`` caps the number of blocks per trajectory (0 = no cap). A
    trajectory shorter than one block gives a single row covering all its
    production frames. The block choice is stored in ``DataFrame.attrs["block"]``.
    Rows are pseudo-replicates of their trajectory; the trajectory (``group``)
    is the statistical unit.
    """
    trajectories = load_md_trajectories(results_dir, equilibration_fs)
    block = choose_common_block_length(
        trajectories, n_windows, block_tau_factor, min_block_length_fs, block_length_fs
    )
    length = block["block_length_frames"]
    print(
        f"[dataset] common block length: {length} frames = {block['block_length_fs']:g} fs "
        f"({block['source']}; max g = {block['max_statistical_inefficiency_frames']:.1f} frames in "
        f"{block['max_statistical_inefficiency_group']} / {block['max_statistical_inefficiency_descriptor']})"
    )
    rows = []
    for trajectory in trajectories:
        frames = trajectory["frames"]
        stats = block["per_trajectory"][trajectory["group"]]
        n_blocks = max(len(frames) // length, 1)
        if len(frames) < length:
            print(
                f"[dataset] {trajectory['group']}: {len(frames)} production frames, shorter than the "
                f"common block ({length} frames); one window for the whole trajectory."
            )
        for window_id in range(n_blocks):
            window = frames.iloc[window_id * length:(window_id + 1) * length]
            row = _window_row(trajectory, window, window_id)
            row["block_length_fs"] = block["block_length_fs"]
            row["statistical_inefficiency_fs"] = stats["statistical_inefficiency_fs"]
            row["lag1_autocorrelation"] = stats["lag1_autocorrelation"]
            rows.append(row)

    windows = pd.DataFrame(rows)
    windows.attrs["block"] = block
    return windows


def load_md_trajectory_features(results_dir: Path, equilibration_fs: float = 0.0) -> pd.DataFrame:
    """One row per independent trajectory (means over its production frames)."""
    rows = []
    for trajectory in load_md_trajectories(results_dir, equilibration_fs):
        frames = trajectory["frames"]
        correlation = trajectory_correlation(frames, trajectory["descriptors"])
        row = _window_row(trajectory, frames, 0)
        rows.append(
            {
                "material": row["material"],
                "site": row["site"],
                "group": row["group"],
                "seed": row["seed"],
                "n_production_frames": row["window_n_frames"],
                "production_length_fs": row["window_end_fs"] - row["window_start_fs"],
                "statistical_inefficiency_fs": correlation["statistical_inefficiency_fs"],
                "lag1_autocorrelation": correlation["lag1_autocorrelation"],
                **{key: value for key, value in row.items() if key.startswith("feature_")},
                **{key: value for key, value in row.items() if key in WINDOW_SITE_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def load_md_summary_features(results_dir: Path, equilibration_fs: float = 0.0) -> pd.DataFrame:
    """One row per (material, site): averages over all production frames of all seeds."""
    rows = []
    trajectories = load_md_trajectories(results_dir, equilibration_fs)
    keys = sorted({(item["material"], item["site"]) for item in trajectories})
    for material, site in keys:
        selected = [
            item for item in trajectories if (item["material"], item["site"]) == (material, site)
        ]
        frames = pd.concat([item["frames"] for item in selected], ignore_index=True)
        rows.append(
            {
                "material": material,
                "site": site,
                "feature_md_c_surface_distance_mean_a": frames["c_surface_distance_ang"].mean(),
                "feature_md_c_surface_distance_std_a": frames["c_surface_distance_ang"].std(ddof=1),
                "feature_md_mean_oco_angle_deg": frames["oco_angle_deg"].mean(),
                "md_n_trajectories": len(selected),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- datasets


def build_summary_dataset(results_dir: Path, equilibration_fs: float = 0.0) -> pd.DataFrame:
    """One row per successful DFT calculation; other methods joined one to one."""
    dataset = load_dft_features(results_dir)
    if dataset.duplicated(subset=["material", "site"]).any():
        raise ValueError("dft_summary.csv has more than one successful row per (material, site).")

    tables = [load_tddft_summary(results_dir), load_co2rr_features(results_dir)]
    try:
        tables.append(load_md_summary_features(results_dir, equilibration_fs))
    except FileNotFoundError as exc:
        print(f"[dataset] {exc}; MD descriptors omitted from the summary.")

    for table in tables:
        if table is None or table.empty:
            continue
        # validate= rejects any many-to-many join, the source of the former
        # 960-row cartesian product (D1).
        dataset = dataset.merge(table, on=["material", "site"], how="left", validate="one_to_one")

    dataset["group"] = dataset["material"] + "/" + dataset["site"]
    return add_site_labels(dataset)


def build_unique_md_window_dataset(
    results_dir: Path,
    md_windows_per_site: int,
    block_tau_factor: float = DEFAULT_BLOCK_TAU_FACTOR,
    min_block_length_fs: float = 0.0,
    equilibration_fs: float = 0.0,
    block_length_fs: float | None = None,
) -> pd.DataFrame:
    # The block length is chosen inside load_md_windows, before labels are added.
    md_windows = load_md_windows(
        results_dir,
        md_windows_per_site,
        block_tau_factor,
        min_block_length_fs,
        equilibration_fs,
        block_length_fs,
    )
    block = md_windows.attrs.get("block")
    md_windows = add_site_labels(md_windows)
    md_windows = md_windows.sort_values(["site", "group", "md_window_id"]).reset_index(drop=True)
    md_windows.attrs["block"] = block
    return md_windows


def build_md_trajectory_dataset(results_dir: Path, equilibration_fs: float = 0.0) -> pd.DataFrame:
    data = add_site_labels(load_md_trajectory_features(results_dir, equilibration_fs))
    return data.sort_values(["site", "group"]).reset_index(drop=True)


def write_block_metadata(path: Path, block: dict, dataset: pd.DataFrame) -> None:
    """Record the common block length and the per-trajectory g next to the dataset."""
    payload = {
        **block,
        "statistical_unit": "trajectory (group = material/site/seed); rows are pseudo-replicates",
        "n_rows": int(len(dataset)),
        "n_groups": int(dataset["group"].nunique()) if "group" in dataset.columns else None,
        "site_analysis_available": bool(
            "window_nearest_site" in dataset.columns
            and (dataset["window_nearest_site"].fillna("").astype(str) != "").any()
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    targets = list(dict.fromkeys(args.targets))

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")
    if args.md_windows_per_site < 0:
        raise SystemExit("--md-windows-per-site must be zero (no cap) or positive.")
    if args.block_tau_factor <= 0:
        raise SystemExit("--block-tau-factor must be positive.")
    if args.block_length_fs is not None and args.block_length_fs <= 0:
        raise SystemExit("--block-length-fs must be positive.")
    if "expanded" in targets:
        raise SystemExit(
            "The 'expanded' dataset was removed: it was a cartesian product of TDDFT "
            "transitions, MD windows and per-site DFT/CO2RR constants (960 rows from 3 "
            "calculations), which leaks the site label. Use 'md_windows' (grouped MD blocks) and 'summary' instead."
        )

    if "summary" in targets:
        summary = build_summary_dataset(results_dir, args.equilibration_fs)
        save_dataset(
            summary,
            output_dir / "multitech_selected_features_dataset.csv",
            SUMMARY_COLUMNS
            + [col for col in summary.columns if col.startswith("feature_co2rr_")]
            + [col for col in ["co2rr_energy_basis", "dft_final_state"] if col in summary.columns],
        )

    if "md_windows" in targets:
        md_unique = build_unique_md_window_dataset(
            results_dir,
            args.md_windows_per_site,
            args.block_tau_factor,
            args.min_block_length_fs,
            args.equilibration_fs,
            args.block_length_fs,
        )
        window_path = output_dir / "md_window_unique_dataset.csv"
        save_dataset(md_unique, window_path, MD_UNIQUE_COLUMNS)
        write_block_metadata(
            window_path.with_name(window_path.stem + BLOCK_METADATA_SUFFIX),
            md_unique.attrs["block"],
            md_unique,
        )
        trajectories = build_md_trajectory_dataset(results_dir, args.equilibration_fs)
        save_dataset(
            trajectories,
            output_dir / "md_trajectory_dataset.csv",
            MD_TRAJECTORY_COLUMNS,
        )


if __name__ == "__main__":
    main()
