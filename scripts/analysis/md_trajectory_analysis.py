"""Site occupancy, height and desorption of CO2 along the AIMD trajectories.

CO2 on ZnO(10-10) is physisorbed (E_ads close to zero) and the surface offers no
diffusion barrier, so during the MD the molecule may migrate to another site or
leave the surface. The starting site used as the class label is then no longer a
description of the frames. This script reads the ASE trajectories written by
run_md.py and, frame by frame, records where the molecule actually is.

Inputs (below --results-dir, one simulation run):

* ``md/<material>/<site>/seed_<n>/md_equilibration.traj`` and
  ``md_production.traj``: one frame every ``log_interval`` steps, frame 0 is the
  start of the phase. Trajectories still being written are read up to their last
  complete frame; missing seeds or phases are reported and skipped.
* ``config.yaml`` of the run (``methods.md``): ``log_interval``, ``timestep_fs``,
  ``equilibration_steps``, ``production_steps`` and ``seeds``; command-line
  options override it.

Per frame, with the minimum-image convention in the surface plane:

* ``nearest_site``: type (top_metal, top_oxygen, bridge) of the nearest of ALL
  symmetry-equivalent top-plane sites of the 2x1 cell, by lateral (xy) distance
  of the C atom. The sites follow the frame: top_metal / top_oxygen are the
  current xy positions of the outermost-plane Zn / O atoms and bridge is the
  midpoint of each surface Zn-O dimer (the Zn and its nearest outermost-plane O).
* ``start_site_lateral_ang``: lateral distance to the nearest site of the
  starting type.
* ``c_height_top_plane_ang``: z of C minus the mean z of the OUTERMOST plane
  only. The plane atoms are taken from a reference clean slab (atoms within
  0.3 A of its highest z; by default the ideal slab of build_geometries, from
  which every DFT start was built). run_md.py reports
  ``c_height_above_slab_plane_ang`` against a 1.0 A window that also takes the
  second plane (0.94 A lower), which biases that height by about +0.47 A; this
  column is the corrected value. run_md.py itself is not changed.
* ``c_min_slab_distance_ang``: minimum-image C-slab distance; ``desorbed`` = 1
  when it exceeds ``--desorption-threshold``.
* ``msd_lateral_ang2``: squared lateral displacement of C from the first
  production frame (positions unwrapped frame to frame). It is a single-trajectory
  quantity, not an ensemble average; NaN during equilibration.

Outputs:

* ``md/<material>/<site>/md_frame_analysis.csv``: one row per frame and seed.
* ``md_trajectory_analysis_summary.csv`` (run level): one row per trajectory.
* ``md/<material>/md_nearest_site_occupancy.png`` (optional): nearest site
  against time, one panel per trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import project_config

DEFAULT_DESORPTION_THRESHOLD_ANG = 4.5
# Outermost plane: atoms within this distance of the highest slab atom of the
# reference slab. On ZnO(10-10) the second plane lies 0.94 A lower.
TOP_PLANE_WINDOW_ANG = 0.3
ADSORBATE_ATOMS = 3  # CO2 is the last three atoms: O, C, O.
SITE_TYPES = ("top_metal", "top_oxygen", "bridge")
PHASES = ("equilibration", "production")
TRAJECTORY_FILES = {
    "equilibration": "md_equilibration.traj",
    "production": "md_production.traj",
}
FRAME_ANALYSIS_FILENAME = "md_frame_analysis.csv"
SUMMARY_FILENAME = "md_trajectory_analysis_summary.csv"
PLOT_FILENAME = "md_nearest_site_occupancy.png"
SURFACE_PBC = (True, True, False)

# Fallbacks when the run has no config.yaml (they mirror run_md.DEFAULTS).
MD_DEFAULTS = {
    "log_interval": 20,
    "timestep_fs": 1.0,
    "equilibration_steps": 1000,
    "production_steps": 2000,
}

FRAME_COLUMNS = [
    "seed",
    "phase",
    "frame_index",
    "step",
    "time_fs",
    "nearest_site",
    "nearest_site_lateral_ang",
    "start_site_lateral_ang",
    "c_height_top_plane_ang",
    "c_min_slab_distance_ang",
    "desorbed",
    "msd_lateral_ang2",
]

SUMMARY_COLUMNS = [
    "material",
    "site",
    "seed",
    "status",
    "n_equilibration_frames",
    "n_production_frames",
    "expected_production_frames",
    "production_time_fs",
    "nearest_site_at_production_start",
    "fraction_production_at_start_site",
    "fraction_production_desorbed",
    "first_leave_start_site_fs",
    "first_desorbed_fs",
    "final_lateral_displacement_ang",
    "max_lateral_displacement_ang",
    "mean_c_height_top_plane_ang",
    "mean_c_min_slab_distance_ang",
    "mean_start_site_lateral_ang",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nearest-site occupancy, top-plane height and desorption along the MD trajectories."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_config.RESULTS_DIR,
        help="Simulation run directory, results/simulations/<run-id> (default: QML_ZNO_RESULTS_DIR).",
    )
    parser.add_argument("--material", default="ZnO")
    parser.add_argument(
        "--desorption-threshold",
        type=float,
        default=DEFAULT_DESORPTION_THRESHOLD_ANG,
        help="C-slab minimum distance (A) above which a frame counts as desorbed.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="MD steps between saved frames (default: methods.md.log_interval of the run config, else 20).",
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=None,
        help="MD time step in fs (default: methods.md.timestep_fs of the run config, else 1.0).",
    )
    parser.add_argument(
        "--production-steps",
        type=int,
        default=None,
        help="Expected production steps, only used to flag partial trajectories (default: run config).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run configuration with methods.md (default: <results-dir>/config.yaml when present).",
    )
    parser.add_argument(
        "--reference-slab",
        type=Path,
        default=None,
        help=(
            "Clean slab whose outermost plane (atoms within 0.3 A of its highest z) defines the "
            "top-plane atoms; same atom order as the MD slab. Default: the ideal slab of "
            "build_geometries (data/geometries/<material>/clean_slab.traj). A relaxed slab "
            "whose surface Zn sank by more than 0.3 A would drop Zn from the plane and is rejected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory of the run-level summary and plot (default: --results-dir).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not write the occupancy plot.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------- configuration


def load_md_settings(results_dir: Path, config_path: Path | None, args: argparse.Namespace) -> dict:
    """Return log_interval, timestep_fs, production_steps and seeds (CLI > config > defaults)."""
    md: dict = {}
    path = config_path if config_path is not None else results_dir / "config.yaml"
    if path.is_file():
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        md = ((payload.get("methods") or {}).get("md")) or {}
        print(f"[md-analysis] configuration: {path}")
    elif config_path is not None:
        raise SystemExit(f"[md-analysis] Configuration file does not exist: {config_path}")
    else:
        print(f"[md-analysis] no config.yaml in {results_dir}; using defaults or command-line values.")

    def pick(cli_value, key, cast):
        if cli_value is not None:
            return cast(cli_value)
        if md.get(key) is not None:
            return cast(md[key])
        return cast(MD_DEFAULTS[key])

    seeds = md.get("seeds") or ([md["seed"]] if md.get("seed") is not None else [])
    production = args.production_steps
    if production is None:
        production = md.get("production_steps", md.get("steps"))
    settings = {
        "log_interval": pick(args.log_interval, "log_interval", int),
        "timestep_fs": pick(args.timestep_fs, "timestep_fs", float),
        "production_steps": int(production) if production is not None else MD_DEFAULTS["production_steps"],
        "seeds": [str(int(seed)) for seed in seeds],
    }
    if settings["log_interval"] <= 0 or settings["timestep_fs"] <= 0:
        raise SystemExit("[md-analysis] log_interval and timestep_fs must be positive.")
    return settings


# --------------------------------------------------------------------- geometry


def lateral_vectors(vectors: np.ndarray, cell) -> np.ndarray:
    """Minimum-image vectors in the surface plane (z set to zero)."""
    from ase.geometry import find_mic

    flat = np.array(vectors, dtype=float).reshape(-1, 3)
    flat[:, 2] = 0.0
    mic, _ = find_mic(flat, cell, pbc=SURFACE_PBC)
    return mic


def lateral_distances(point: np.ndarray, targets: np.ndarray, cell) -> np.ndarray:
    return np.linalg.norm(lateral_vectors(np.asarray(targets) - np.asarray(point), cell), axis=1)


def top_plane_indices(slab, window: float = TOP_PLANE_WINDOW_ANG) -> list[int]:
    """Indices of the slab atoms within ``window`` of the highest slab atom."""
    z = slab.positions[:, 2]
    return [int(i) for i in np.flatnonzero(z >= float(z.max()) - window)]


class SiteModel:
    """Symmetry-equivalent top-plane sites, evaluated on the positions of each frame.

    The atom indices come from a reference clean slab: the outermost-plane metal
    atoms (top_metal), outermost-plane O atoms (top_oxygen) and, for every such
    metal, its nearest outermost-plane O (the surface dimer; its midpoint is a
    bridge site). Lateral distances use the minimum image of ``cell``.
    """

    def __init__(self, reference, metal_symbol: str, window: float = TOP_PLANE_WINDOW_ANG):
        self.n_slab = len(reference)
        self.symbols = list(reference.get_chemical_symbols())
        self.cell = np.array(reference.cell)
        self.top_indices = top_plane_indices(reference, window)
        self.metal_indices = [i for i in self.top_indices if self.symbols[i] == metal_symbol]
        self.oxygen_indices = [i for i in self.top_indices if self.symbols[i] == "O"]
        if not self.metal_indices or not self.oxygen_indices:
            raise ValueError(
                f"The outermost plane of the reference slab (window {window} A) must contain "
                f"{metal_symbol} and O atoms; found {[self.symbols[i] for i in self.top_indices]}. "
                "Use the ideal (unrelaxed) slab as reference."
            )
        positions = reference.positions
        pairs = []
        for metal in self.metal_indices:
            distances = lateral_distances(positions[metal], positions[self.oxygen_indices], self.cell)
            pairs.append((metal, self.oxygen_indices[int(np.argmin(distances))]))
        self.bridge_pairs = sorted(set(pairs))

    def check_frame(self, atoms) -> None:
        """Reject frames whose slab does not match the reference (order, size or cell)."""
        if len(atoms) != self.n_slab + ADSORBATE_ATOMS:
            raise ValueError(
                f"Frame has {len(atoms)} atoms; expected {self.n_slab} slab atoms + CO2."
            )
        if list(atoms.get_chemical_symbols()[: self.n_slab]) != self.symbols:
            raise ValueError("Slab atom order of the frame differs from the reference slab.")
        if list(atoms.get_chemical_symbols()[self.n_slab:]) != ["O", "C", "O"]:
            raise ValueError("The last three atoms of the frame are not O, C, O.")
        if not np.allclose(np.array(atoms.cell)[:2, :2], self.cell[:2, :2], atol=1e-3):
            raise ValueError("Surface cell of the frame differs from the reference slab.")

    def sites(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        metal = positions[self.metal_indices]
        oxygen = positions[self.oxygen_indices]
        bridge = np.array([
            positions[m] + 0.5 * lateral_vectors(positions[o] - positions[m], self.cell)[0]
            for m, o in self.bridge_pairs
        ])
        return {"top_metal": metal, "top_oxygen": oxygen, "bridge": bridge}

    def plane_z(self, positions: np.ndarray) -> float:
        return float(np.mean(positions[self.top_indices, 2]))


def analyze_frame(atoms, model: SiteModel, start_site: str, threshold: float) -> dict:
    from ase.geometry import get_distances

    positions = atoms.get_positions()
    carbon = positions[model.n_slab + 1]
    site_distances = {
        name: float(lateral_distances(carbon, points, model.cell).min())
        for name, points in model.sites(positions).items()
    }
    nearest = min(SITE_TYPES, key=lambda name: site_distances[name])
    _, slab_distances = get_distances(
        carbon[None, :], positions[: model.n_slab], cell=model.cell, pbc=SURFACE_PBC
    )
    c_min = float(slab_distances.min())
    return {
        "nearest_site": nearest,
        "nearest_site_lateral_ang": site_distances[nearest],
        "start_site_lateral_ang": site_distances.get(start_site, float("nan")),
        "c_height_top_plane_ang": float(carbon[2]) - model.plane_z(positions),
        "c_min_slab_distance_ang": c_min,
        "desorbed": int(c_min > threshold),
        "_carbon_xyz": carbon,
    }


def load_reference_slab(material: str, path: Path | None):
    if path is not None:
        from ase.io import read

        if not path.is_file():
            raise SystemExit(f"[md-analysis] Reference slab not found: {path}")
        return read(path)
    from scripts.simulations.dft.build_geometries import build_slab

    return build_slab(material)


# ------------------------------------------------------------------ trajectories


def read_frames(path: Path) -> list:
    """Every complete frame of an ASE trajectory; [] when it is missing or unreadable.

    A trajectory still being written may end with an incomplete frame: reading
    stops there, the frames before it are kept.
    """
    if not path.is_file():
        return []
    from ase.io.trajectory import Trajectory

    try:
        traj = Trajectory(str(path), "r")
    except Exception as exc:  # truncated header, file being created
        print(f"[md-analysis] cannot open {path}: {type(exc).__name__}: {exc}")
        return []
    frames = []
    try:
        for index in range(len(traj)):
            try:
                frames.append(traj[index])
            except Exception as exc:
                print(f"[md-analysis] {path}: stopped at frame {index} ({type(exc).__name__}); partial file.")
                break
    finally:
        traj.close()
    return frames


def analyze_trajectory(
    seed_dir: Path,
    seed: str,
    model: SiteModel,
    start_site: str,
    threshold: float,
    log_interval: int,
    timestep_fs: float,
) -> pd.DataFrame:
    """Frame table of one seed (equilibration then production)."""
    rows = []
    n_equilibration = 0
    for phase in PHASES:
        frames = read_frames(seed_dir / TRAJECTORY_FILES[phase])
        if phase == "equilibration":
            n_equilibration = len(frames)
        unwrapped = None
        origin = None
        previous = None
        for index, atoms in enumerate(frames):
            model.check_frame(atoms)
            data = analyze_frame(atoms, model, start_site, threshold)
            carbon = data.pop("_carbon_xyz")
            msd = float("nan")
            if phase == "production":
                if previous is None:
                    unwrapped = carbon[:2].copy()
                    origin = unwrapped.copy()
                else:
                    unwrapped = unwrapped + lateral_vectors(carbon - previous, model.cell)[0, :2]
                msd = float(np.sum((unwrapped - origin) ** 2))
                previous = carbon
            offset = 0 if phase == "equilibration" else max(n_equilibration - 1, 0) * log_interval
            rows.append({
                "seed": seed,
                "phase": phase,
                "frame_index": index,
                "step": offset + index * log_interval,
                "time_fs": index * log_interval * timestep_fs,
                **data,
                "msd_lateral_ang2": msd,
            })
    return pd.DataFrame(rows, columns=FRAME_COLUMNS)


def _seed_key(name: str) -> tuple[int, str]:
    value = name.removeprefix("seed_")
    return (int(value), value) if value.isdigit() else (10**9, value)


def discover_trajectories(results_dir: Path, material: str) -> dict[str, list[tuple[str, Path]]]:
    """{site: [(seed, seed_dir), ...]} for md/<material>/<site>/seed_<n>/."""
    base = results_dir / "md" / material
    found: dict[str, list[tuple[str, Path]]] = {}
    if not base.is_dir():
        return found
    for site_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        if site_dir.name not in SITE_TYPES:
            continue
        seeds = sorted(
            (path for path in site_dir.glob("seed_*") if path.is_dir()),
            key=lambda path: _seed_key(path.name),
        )
        found[site_dir.name] = [(path.name.removeprefix("seed_"), path) for path in seeds]
    return found


def summarize_trajectory(
    material: str,
    site: str,
    seed: str,
    frames: pd.DataFrame,
    expected_production_frames: int | None,
) -> dict:
    production = frames[frames["phase"] == "production"].reset_index(drop=True)
    n_production = len(production)
    row = {name: float("nan") for name in SUMMARY_COLUMNS}
    row.update({
        "material": material,
        "site": site,
        "seed": seed,
        "n_equilibration_frames": int((frames["phase"] == "equilibration").sum()),
        "n_production_frames": n_production,
        "expected_production_frames": expected_production_frames,
        "nearest_site_at_production_start": "",
    })
    if n_production == 0:
        row["status"] = "no_production" if len(frames) else "missing"
        return row
    complete = expected_production_frames is not None and n_production >= expected_production_frames
    row["status"] = "complete" if complete else "partial"
    at_start = production["nearest_site"] == site
    desorbed = production["desorbed"].astype(int) == 1
    displacement = np.sqrt(production["msd_lateral_ang2"].to_numpy(dtype=float))
    row.update({
        "production_time_fs": float(production["time_fs"].iloc[-1]),
        "nearest_site_at_production_start": str(production["nearest_site"].iloc[0]),
        "fraction_production_at_start_site": float(at_start.mean()),
        "fraction_production_desorbed": float(desorbed.mean()),
        "first_leave_start_site_fs": (
            float(production.loc[~at_start, "time_fs"].iloc[0]) if (~at_start).any() else float("nan")
        ),
        "first_desorbed_fs": (
            float(production.loc[desorbed, "time_fs"].iloc[0]) if desorbed.any() else float("nan")
        ),
        "final_lateral_displacement_ang": float(displacement[-1]),
        "max_lateral_displacement_ang": float(np.nanmax(displacement)),
        "mean_c_height_top_plane_ang": float(production["c_height_top_plane_ang"].mean()),
        "mean_c_min_slab_distance_ang": float(production["c_min_slab_distance_ang"].mean()),
        "mean_start_site_lateral_ang": float(production["start_site_lateral_ang"].mean()),
    })
    return row


def analyze_run(
    results_dir: Path,
    material: str,
    model: SiteModel,
    settings: dict,
    threshold: float = DEFAULT_DESORPTION_THRESHOLD_ANG,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Analyse every seed of every site; write md_frame_analysis.csv per site.

    Returns ({site: frame table}, summary table). Seeds listed in the run
    configuration without a trajectory appear in the summary as ``missing``.
    """
    log_interval = settings["log_interval"]
    production_steps = settings.get("production_steps")
    expected = production_steps // log_interval + 1 if production_steps else None
    site_frames: dict[str, pd.DataFrame] = {}
    summary_rows = []
    discovered = discover_trajectories(results_dir, material)
    if not discovered:
        print(f"[md-analysis] no seed_<n>/ trajectories below {results_dir / 'md' / material}")
    for site, seeds in discovered.items():
        tables = []
        present = {seed for seed, _ in seeds}
        for seed, seed_dir in seeds:
            try:
                frames = analyze_trajectory(
                    seed_dir, seed, model, site, threshold, log_interval, settings["timestep_fs"]
                )
            except ValueError as exc:
                print(f"[md-analysis] {material}/{site}/seed_{seed}: skipped ({exc})")
                row = summarize_trajectory(material, site, seed, pd.DataFrame(columns=FRAME_COLUMNS), expected)
                row["status"] = "error"
                summary_rows.append(row)
                continue
            summary_rows.append(summarize_trajectory(material, site, seed, frames, expected))
            if frames.empty:
                print(f"[md-analysis] {material}/{site}/seed_{seed}: no readable frames yet.")
                continue
            tables.append(frames)
            last = summary_rows[-1]
            print(
                f"[md-analysis] {material}/{site}/seed_{seed}: {last['n_equilibration_frames']} eq + "
                f"{last['n_production_frames']} production frames ({last['status']}); "
                f"at start site {last['fraction_production_at_start_site']:.2f}, "
                f"desorbed {last['fraction_production_desorbed']:.2f}"
            )
        for seed in settings.get("seeds", []):
            if seed not in present:
                summary_rows.append(
                    summarize_trajectory(material, site, seed, pd.DataFrame(columns=FRAME_COLUMNS), expected)
                )
                print(f"[md-analysis] {material}/{site}/seed_{seed}: not found (missing).")
        if tables:
            table = pd.concat(tables, ignore_index=True)
            site_frames[site] = table
            path = results_dir / "md" / material / site / FRAME_ANALYSIS_FILENAME
            table.to_csv(path, index=False)
            print(f"[md-analysis] wrote {path} ({len(table)} frames)")
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    return site_frames, summary


def plot_occupancy(site_frames: dict[str, pd.DataFrame], path: Path, timestep_fs: float, log_interval: int) -> bool:
    """Nearest site against time (production starts at 0 fs; equilibration is negative)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[md-analysis] matplotlib not available; occupancy plot skipped.")
        return False
    panels = [
        (site, seed, frames)
        for site in SITE_TYPES
        if site in site_frames
        for seed, frames in site_frames[site].groupby("seed", sort=False)
    ]
    if not panels:
        return False
    n_cols = max(sum(1 for site_, _, _ in panels if site_ == site) for site in site_frames)
    n_rows = len(site_frames)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.4 * n_rows), squeeze=False, sharey=True)
    for ax in axes.ravel():
        ax.set_visible(False)
    level = {name: index for index, name in enumerate(SITE_TYPES)}
    rows = {site: index for index, site in enumerate(site for site in SITE_TYPES if site in site_frames)}
    columns = {site: 0 for site in site_frames}
    for site, seed, frames in panels:
        ax = axes[rows[site], columns[site]]
        columns[site] += 1
        ax.set_visible(True)
        n_eq = int((frames["phase"] == "equilibration").sum())
        shift = np.where(
            frames["phase"] == "equilibration", -max(n_eq - 1, 0) * log_interval * timestep_fs, 0.0
        )
        time = frames["time_fs"].to_numpy(dtype=float) + shift
        y = frames["nearest_site"].map(level).to_numpy(dtype=float)
        ax.step(time, y, where="post", lw=1.0, color="tab:blue")
        desorbed = frames["desorbed"].astype(int).to_numpy() == 1
        if desorbed.any():
            ax.plot(time[desorbed], y[desorbed], "x", color="tab:red", ms=3, label="desorbed")
            ax.legend(loc="upper right", fontsize=7)
        ax.axvline(0.0, color="0.6", lw=0.8, ls="--")
        ax.axhline(level[site], color="tab:green", lw=0.8, alpha=0.4)
        ax.set_title(f"{site} / seed {seed}", fontsize=9)
        ax.set_yticks(range(len(SITE_TYPES)), SITE_TYPES, fontsize=7)
        ax.set_xlabel("time from production start (fs)", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[md-analysis] wrote {path}")
    return True


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"[md-analysis] Results directory not found: {results_dir}")
    if args.desorption_threshold <= 0:
        raise SystemExit("[md-analysis] --desorption-threshold must be positive.")
    output_dir = (args.output_dir or results_dir).expanduser().resolve()
    settings = load_md_settings(results_dir, args.config, args)
    metal_symbol = project_config.MATERIALS[args.material].metal_symbol
    try:
        model = SiteModel(load_reference_slab(args.material, args.reference_slab), metal_symbol)
    except ValueError as exc:
        raise SystemExit(f"[md-analysis] {exc}") from exc
    print(
        f"[md-analysis] top plane atoms {model.top_indices}; bridge pairs {model.bridge_pairs}; "
        f"log_interval={settings['log_interval']} timestep={settings['timestep_fs']} fs; "
        f"desorption threshold {args.desorption_threshold} A"
    )
    site_frames, summary = analyze_run(
        results_dir, args.material, model, settings, args.desorption_threshold
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)
    print(f"[md-analysis] wrote {summary_path} ({len(summary)} trajectories)")
    if not args.no_plot:
        plot_occupancy(
            site_frames,
            output_dir / "md" / args.material / PLOT_FILENAME,
            settings["timestep_fs"],
            settings["log_interval"],
        )
    return 0 if site_frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
