"""MD trajectory post-processing on synthetic trajectories of the real 2x1 ZnO(10-10) slab."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.io.trajectory import Trajectory

from scripts.analysis import md_trajectory_analysis as mta
from scripts.simulations.dft.build_geometries import add_co2_to_site, build_slab, select_adsorption_sites

HEIGHT = 2.2  # C above the outermost plane
LOG_INTERVAL = 20
TIMESTEP_FS = 1.0
SECOND_PLANE_DROP = 0.938  # ZnO(10-10): the second plane lies 0.94 A lower


@pytest.fixture(scope="module")
def slab():
    return build_slab("ZnO")


@pytest.fixture(scope="module")
def model(slab):
    return mta.SiteModel(slab, "Zn")


def adsorbed(slab, xy, height=HEIGHT):
    atoms = add_co2_to_site(slab, "top_metal", select_adsorption_sites(slab, "Zn")["top_metal"])
    z_top = slab.positions[:, 2].max()
    shift = np.array([xy[0], xy[1], z_top + height]) - atoms.positions[-2]
    atoms.positions[-3:] += shift
    return atoms


def write_traj(path: Path, slab, path_xy, heights=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    heights = heights if heights is not None else [HEIGHT] * len(path_xy)
    with Trajectory(str(path), "w") as traj:
        for xy, height in zip(path_xy, heights):
            traj.write(adsorbed(slab, xy, height))


def settings(**overrides):
    base = {"log_interval": LOG_INTERVAL, "timestep_fs": TIMESTEP_FS, "production_steps": 200, "seeds": []}
    base.update(overrides)
    return base


def test_reference_sites_cover_every_equivalent_site(slab, model):
    sites = model.sites(slab.positions)
    assert {name: len(points) for name, points in sites.items()} == {"top_metal": 2, "top_oxygen": 2, "bridge": 2}
    # Only the outermost plane: four atoms, the second plane (0.94 A lower) is excluded.
    z = slab.positions[:, 2]
    assert len(model.top_indices) == 4
    assert np.allclose(z[model.top_indices], z.max())
    # The sites used to build the DFT starts are among the equivalent sites.
    chosen = select_adsorption_sites(slab, "Zn")
    for name in mta.SITE_TYPES:
        assert mta.lateral_distances(chosen[name], sites[name], model.cell).min() < 1e-6


def test_nearest_site_uses_minimum_image(slab, model):
    zn = slab.positions[model.metal_indices[0]]
    bridge = model.sites(slab.positions)["bridge"][0]
    cell_y = slab.cell[1, 1]
    cell_x = slab.cell[0, 0]
    # C just across the periodic boundary from a Zn: far in Cartesian y, close by mic.
    across = (zn[0] + cell_x, zn[1] - cell_y + 0.1)
    result = mta.analyze_frame(adsorbed(slab, across), model, "top_metal", 4.5)
    assert result["nearest_site"] == "top_metal"
    assert result["nearest_site_lateral_ang"] == pytest.approx(0.1, abs=1e-6)
    assert result["start_site_lateral_ang"] == pytest.approx(0.1, abs=1e-6)
    shifted = (bridge[0] - cell_x, bridge[1] + cell_y - 0.05)
    result = mta.analyze_frame(adsorbed(slab, shifted), model, "top_metal", 4.5)
    assert result["nearest_site"] == "bridge"
    assert result["start_site_lateral_ang"] > 0.9


def test_height_uses_only_the_outermost_plane(slab, model):
    result = mta.analyze_frame(adsorbed(slab, slab.positions[model.metal_indices[0], :2]), model, "top_metal", 4.5)
    assert result["c_height_top_plane_ang"] == pytest.approx(HEIGHT, abs=1e-6)
    # run_md's 1.0 A window mixes both planes and reports about +0.47 A more.
    z = slab.positions[:, 2]
    two_planes = slab.positions[z >= z.max() - 1.0, 2].mean()
    assert z.max() - two_planes == pytest.approx(SECOND_PLANE_DROP / 2, abs=0.01)
    assert result["c_min_slab_distance_ang"] == pytest.approx(HEIGHT, abs=1e-6)
    assert result["desorbed"] == 0


def test_relaxed_reference_without_surface_metal_is_rejected(slab):
    buckled = slab.copy()
    z = buckled.positions[:, 2]
    for index, atom in enumerate(buckled):
        if atom.symbol == "Zn" and z[index] > z.max() - 0.1:
            buckled.positions[index, 2] -= 0.4
    with pytest.raises(ValueError, match="outermost plane"):
        mta.SiteModel(buckled, "Zn")


def make_run(tmp_path, slab, model):
    sites = model.sites(slab.positions)
    zn = sites["top_metal"][0]
    bridge = sites["bridge"][0]
    md = tmp_path / "md" / "ZnO"
    # top_metal seed 1: 2 equilibration frames at Zn, then 6 production frames walking to the bridge.
    seed1 = md / "top_metal" / "seed_1"
    write_traj(seed1 / "md_equilibration.traj", slab, [zn[:2], zn[:2]])
    walk = [zn[:2] + f * (bridge[:2] - zn[:2]) for f in np.linspace(0.0, 1.0, 6)]
    write_traj(seed1 / "md_production.traj", slab, walk)
    # top_metal seed 2: stays laterally, rises above the desorption threshold.
    seed2 = md / "top_metal" / "seed_2"
    write_traj(seed2 / "md_equilibration.traj", slab, [zn[:2]])
    write_traj(seed2 / "md_production.traj", slab, [zn[:2]] * 6, heights=[2.2, 3.0, 4.0, 5.0, 6.0, 7.0])
    # bridge seed 3: still in equilibration (no production file yet).
    seed3 = md / "bridge" / "seed_3"
    write_traj(seed3 / "md_equilibration.traj", slab, [bridge[:2]] * 3)
    return tmp_path


def test_run_analysis_tracks_migration_and_desorption(tmp_path, slab, model):
    root = make_run(tmp_path, slab, model)
    site_frames, summary = mta.analyze_run(root, "ZnO", model, settings(seeds=["1", "2", "3", "4"]), 4.5)

    frames = pd.read_csv(root / "md/ZnO/top_metal/md_frame_analysis.csv")
    assert list(frames.columns) == mta.FRAME_COLUMNS
    production = frames[(frames["seed"] == 1) & (frames["phase"] == "production")]
    assert production["nearest_site"].tolist() == ["top_metal"] * 3 + ["bridge"] * 3
    assert production["time_fs"].tolist() == [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    assert production["step"].iloc[0] == LOG_INTERVAL  # after one equilibration interval
    step = np.linalg.norm(model.sites(slab.positions)["bridge"][0][:2] - model.sites(slab.positions)["top_metal"][0][:2])
    assert production["msd_lateral_ang2"].iloc[-1] == pytest.approx(step**2, rel=1e-6)
    assert frames.loc[frames["phase"] == "equilibration", "msd_lateral_ang2"].isna().all()

    desorbing = frames[(frames["seed"] == 2) & (frames["phase"] == "production")]
    assert desorbing["desorbed"].tolist() == [0, 0, 0, 1, 1, 1]
    assert (desorbing["c_height_top_plane_ang"].diff().dropna() > 0).all()

    rows = summary.set_index(["site", "seed"])
    seed1 = rows.loc[("top_metal", "1")]
    assert seed1["fraction_production_at_start_site"] == pytest.approx(0.5)
    assert seed1["first_leave_start_site_fs"] == pytest.approx(60.0)
    assert seed1["final_lateral_displacement_ang"] == pytest.approx(step, rel=1e-6)
    assert seed1["status"] == "partial"  # 6 frames < 200 / 20 + 1
    assert seed1["mean_c_height_top_plane_ang"] == pytest.approx(HEIGHT, abs=1e-6)
    seed2 = rows.loc[("top_metal", "2")]
    assert seed2["fraction_production_desorbed"] == pytest.approx(0.5)
    assert seed2["first_desorbed_fs"] == pytest.approx(60.0)
    assert rows.loc[("bridge", "3"), "status"] == "no_production"
    assert rows.loc[("bridge", "3"), "n_equilibration_frames"] == 3
    assert rows.loc[("top_metal", "4"), "status"] == "missing"
    assert "bridge" in site_frames  # equilibration frames are still written


def test_truncated_trajectory_is_read_up_to_the_last_complete_frame(tmp_path, slab, model):
    path = tmp_path / "md_production.traj"
    write_traj(path, slab, [slab.positions[model.metal_indices[0], :2]] * 5)
    data = path.read_bytes()
    path.write_bytes(data[:-200])  # frame 5 is being written
    assert len(mta.read_frames(path)) == 4
    assert mta.read_frames(tmp_path / "absent.traj") == []


def test_main_writes_summary_and_plot(tmp_path, slab, model):
    root = make_run(tmp_path, slab, model)
    (root / "config.yaml").write_text(
        "methods:\n  md:\n    log_interval: 20\n    timestep_fs: 1.0\n    production_steps: 200\n"
        "    seeds: [1, 2]\n",
        encoding="utf-8",
    )
    assert mta.main(["--results-dir", str(root)]) == 0
    summary = pd.read_csv(root / mta.SUMMARY_FILENAME)
    assert set(summary["status"]) >= {"partial", "no_production"}
    assert list(summary.columns) == mta.SUMMARY_COLUMNS
    pytest.importorskip("matplotlib")
    assert (root / "md" / "ZnO" / mta.PLOT_FILENAME).is_file()


def test_frame_analysis_feeds_dataset_windows(tmp_path, slab, model):
    """md_frame_analysis.csv joins md_production_frames.csv on (seed, time_fs)."""
    from scripts.datasets import prepare_multitech_dataset as pm

    root = make_run(tmp_path, slab, model)
    mta.analyze_run(root, "ZnO", model, settings(), 4.5)
    rows = [
        {"seed": seed, "phase": "production", "step": 20 * k, "time_fs": 20.0 * k,
         "c_surface_distance_ang": 2.2 + 0.01 * k, "oco_angle_deg": 179.0 - k}
        for seed in (1, 2) for k in range(6)
    ]
    pd.DataFrame(rows).to_csv(root / "md/ZnO/top_metal/md_production_frames.csv", index=False)
    windows = pm.load_md_windows(root, 0, block_length_fs=60.0)
    seed1 = windows[windows["seed"] == "1"]
    assert seed1["window_nearest_site"].tolist() == ["top_metal", "bridge"]
    assert seed1["window_start_site_fraction"].tolist() == [1.0, 0.0]
    seed2 = windows[windows["seed"] == "2"]
    assert seed2["window_desorbed_fraction"].tolist() == [0.0, 1.0]
    assert (windows["window_site_analysis_coverage"] == 1.0).all()


def test_md_defaults_mirror_run_md():
    from scripts.simulations.md import run_md

    for key in ("log_interval", "timestep_fs", "equilibration_steps", "production_steps"):
        assert mta.MD_DEFAULTS[key] == run_md.DEFAULTS[key], key


def test_co2_slab_distance_sees_the_oxygen_atoms(slab, model):
    # Upright CO2 standing on one O: C is far from the slab, the lower O is not.
    zn = slab.positions[model.metal_indices[0]]
    atoms = adsorbed(slab, zn[:2], height=4.0)
    carbon = atoms.positions[-2].copy()
    atoms.positions[-3] = carbon + [0.0, 0.0, -1.16]
    atoms.positions[-1] = carbon + [0.0, 0.0, 1.16]
    result = mta.analyze_frame(atoms, model, "top_metal", 3.5)
    assert result["c_min_slab_distance_ang"] == pytest.approx(4.0, abs=1e-6)
    assert result["c_o_min_slab_distance_ang"] == pytest.approx(4.0 - 1.16, abs=1e-6)
    assert result["desorbed"] == 1  # C-only, single frame: unchanged definition


def test_persistent_mask_requires_consecutive_frames():
    above = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    assert mta.persistent_mask(above, 3).astype(int).tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert mta.persistent_mask(above, 1).astype(int).tolist() == above
    assert mta.persistent_mask([], 3).tolist() == []


def trench_point(slab, model):
    """Midway between the O of a dimer and the Zn of the next row (periodic image in y)."""
    sites = model.sites(slab.positions)
    zn = sites["top_metal"][0]
    return (zn[0], 0.5 * (sites["top_oxygen"][0][1] + zn[1] + slab.cell[1, 1]))


def test_site_assignment_marks_the_trench_as_other(slab, model):
    zn = model.sites(slab.positions)["top_metal"][0]
    trench = trench_point(slab, model)
    result = mta.analyze_frame(adsorbed(slab, trench), model, "top_metal", 4.5)
    assert result["nearest_site_lateral_ang"] > 1.0
    assert result["nearest_site"] in mta.SITE_TYPES  # still forced to a type
    assert result["site_assignment"] == "other"
    near = mta.analyze_frame(adsorbed(slab, (zn[0] + 0.3, zn[1])), model, "top_metal", 4.5)
    assert near["site_assignment"] == "top_metal"
    wide = mta.analyze_frame(adsorbed(slab, trench), model, "top_metal", 4.5, site_radius=2.0)
    assert wide["site_assignment"] == wide["nearest_site"]


def test_new_columns_summary_and_sensitivity(tmp_path, slab, model):
    zn = model.sites(slab.positions)["top_metal"][0]
    trench = trench_point(slab, model)
    seed_dir = tmp_path / "md" / "ZnO" / "top_metal" / "seed_1"
    write_traj(seed_dir / "md_equilibration.traj", slab, [zn[:2]])
    # Production: a one-frame spike to 4.8 A, back, three frames at 4.8 A, two at 5.5 A over the trench.
    heights = [2.2, 4.8, 2.2, 4.8, 4.8, 4.8, 5.5, 5.5]
    path_xy = [zn[:2]] * 6 + [trench] * 2
    write_traj(seed_dir / "md_production.traj", slab, path_xy, heights=heights)
    site_frames, summary = mta.analyze_run(tmp_path, "ZnO", model, settings(sites=["top_metal"]), 4.5)

    frames = site_frames["top_metal"]
    assert list(pd.read_csv(tmp_path / "md/ZnO/top_metal/md_frame_analysis.csv").columns) == mta.FRAME_COLUMNS
    production = frames[frames["phase"] == "production"]
    assert production["desorbed_persistent"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert production["site_assignment"].tolist() == ["top_metal"] * 6 + ["other"] * 2
    assert production["nearest_site"].iloc[-1] in mta.SITE_TYPES
    row = summary.set_index(["site", "seed"]).loc[("top_metal", "1")]
    assert row["fraction_production_other"] == pytest.approx(0.25)
    assert row["fraction_production_desorbed_persistent"] == pytest.approx(5 / 8)
    assert row["first_desorbed_persistent_fs"] == pytest.approx(60.0)
    assert row["fraction_production_desorbed"] == pytest.approx(6 / 8)  # C-based single frame, as before
    # Adsorbed at the start site: site_assignment == start AND not persistently desorbed (frames 0-2).
    assert row["fraction_production_adsorbed_at_start_site"] == pytest.approx(3 / 8)
    assert row["fraction_production_at_start_site"] == pytest.approx(6 / 8)  # nearest_site, desorbed included

    table = mta.desorption_sensitivity("ZnO", site_frames, [4.0, 4.5, 5.0], 3)
    assert list(table.columns) == mta.SENSITIVITY_COLUMNS
    by_threshold = table.set_index("threshold")
    assert by_threshold.loc[4.5, "fraction_desorbed"] == pytest.approx(6 / 8)
    assert by_threshold.loc[4.5, "fraction_desorbed_persistent"] == pytest.approx(5 / 8)
    assert by_threshold.loc[5.0, "fraction_desorbed"] == pytest.approx(2 / 8)
    assert by_threshold.loc[5.0, "fraction_desorbed_persistent"] == 0.0  # only 2 consecutive frames


def test_missing_sites_and_not_started_seeds(tmp_path, slab, model, capsys):
    zn = model.sites(slab.positions)["top_metal"][0]
    md = tmp_path / "md" / "ZnO"
    write_traj(md / "top_metal" / "seed_1" / "md_equilibration.traj", slab, [zn[:2]] * 2)
    write_traj(md / "top_metal" / "seed_1" / "md_production.traj", slab, [zn[:2]] * 3)
    not_started = md / "bridge" / "seed_1"
    not_started.mkdir(parents=True)
    (not_started / "md_equilibration.traj").write_bytes(b"")
    (tmp_path / "config.yaml").write_text(
        "selection:\n  sites: [top_metal, top_oxygen, bridge]\n"
        "methods:\n  md:\n    log_interval: 20\n    timestep_fs: 1.0\n    production_steps: 40\n",
        encoding="utf-8",
    )
    assert mta.main(["--results-dir", str(tmp_path), "--no-plot"]) == 0
    assert "cannot open" not in capsys.readouterr().out
    summary = pd.read_csv(tmp_path / mta.SUMMARY_FILENAME, dtype={"seed": str})
    status = dict(zip(summary["site"], summary["status"]))
    assert status == {"top_metal": "complete", "top_oxygen": "missing", "bridge": "not_started"}
    sensitivity = pd.read_csv(tmp_path / mta.SENSITIVITY_FILENAME)
    assert sorted(sensitivity["threshold"]) == [4.0, 4.5, 5.0]
    assert (sensitivity["fraction_desorbed"] == 0.0).all()


def test_configured_sites_fallbacks():
    assert mta.configured_sites({}) == list(mta.SITE_TYPES)
    assert mta.configured_sites({"selection": {"sites": ["bridge", "carbonate"]}}) == ["bridge"]
    assert mta.configured_sites({"methods": {"dft": {"sites": ["top_oxygen"]}}}) == ["top_oxygen"]


def test_output_dir_holds_run_level_files_and_frames_stay_in_the_run(tmp_path, slab, model):
    root = make_run(tmp_path / "run", slab, model)
    out = tmp_path / "analysis" / "science" / "md"
    before = {path for path in tmp_path.rglob("*") if path.is_file()}
    assert mta.main(["--results-dir", str(root), "--output-dir", str(out), "--log-interval", "20",
                     "--timestep-fs", "1.0"]) == 0
    created = {path for path in tmp_path.rglob("*") if path.is_file()} - before
    in_out = {path.relative_to(out) for path in created if out in path.parents}
    in_run = {path.relative_to(root) for path in created if root in path.parents}
    assert created == {out / path for path in in_out} | {root / path for path in in_run}
    expected_out = {Path(mta.SUMMARY_FILENAME), Path(mta.SENSITIVITY_FILENAME)}
    import importlib.util

    if importlib.util.find_spec("matplotlib") is not None:
        expected_out.add(Path(mta.PLOT_FILENAME))  # directly in OUT, not OUT/md/ZnO/
    assert in_out == expected_out
    assert in_run == {Path("md/ZnO/top_metal") / mta.FRAME_ANALYSIS_FILENAME,
                      Path("md/ZnO/bridge") / mta.FRAME_ANALYSIS_FILENAME}
    assert not (root / mta.SUMMARY_FILENAME).exists()


def test_unreadable_trajectory_is_not_reported_as_missing(tmp_path, slab, model, capsys):
    """A file with content that ASE cannot open (e.g. two writers) is 'unreadable', not 'missing'."""
    zn = model.sites(slab.positions)["top_metal"][0]
    md = tmp_path / "md" / "ZnO" / "top_metal"
    write_traj(md / "seed_1" / "md_equilibration.traj", slab, [zn[:2]] * 2)
    write_traj(md / "seed_1" / "md_production.traj", slab, [zn[:2]] * 3)
    broken = md / "seed_2" / "md_equilibration.traj"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"- of UlmASE-Trajectory" + b"{not json" * 20)
    with pytest.raises(mta.UnreadableTrajectory):
        mta.read_frames(broken)
    _, summary = mta.analyze_run(tmp_path, "ZnO", model, settings(sites=["top_metal"], production_steps=40), 4.5)
    status = dict(zip(summary["seed"], summary["status"]))
    assert status == {"1": "complete", "2": "unreadable"}
    assert "UNREADABLE" in capsys.readouterr().out


def test_more_production_frames_than_expected_is_overfull(tmp_path, slab, model):
    zn = model.sites(slab.positions)["top_metal"][0]
    seed_dir = tmp_path / "md" / "ZnO" / "top_metal" / "seed_1"
    write_traj(seed_dir / "md_equilibration.traj", slab, [zn[:2]])
    write_traj(seed_dir / "md_production.traj", slab, [zn[:2]] * 6)  # e.g. two appending writers
    _, summary = mta.analyze_run(tmp_path, "ZnO", model, settings(sites=["top_metal"], production_steps=40), 4.5)
    assert summary["status"].tolist() == ["overfull"]  # 6 frames > 40 / 20 + 1
    _, summary = mta.analyze_run(tmp_path, "ZnO", model, settings(sites=["top_metal"], production_steps=100), 4.5)
    assert summary["status"].tolist() == ["complete"]


def test_stale_frame_file_is_removed_when_no_seed_is_readable(tmp_path, slab, model):
    seed_dir = tmp_path / "md" / "ZnO" / "bridge" / "seed_1"
    seed_dir.mkdir(parents=True)
    (seed_dir / "md_equilibration.traj").write_bytes(b"")  # not started
    stale = tmp_path / "md" / "ZnO" / "bridge" / mta.FRAME_ANALYSIS_FILENAME
    stale.write_text("seed,phase\n1,production\n", encoding="utf-8")
    site_frames, _ = mta.analyze_run(tmp_path, "ZnO", model, settings(sites=["bridge"]), 4.5)
    assert "bridge" not in site_frames
    assert not stale.exists()
