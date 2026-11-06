"""MD driver checks with an EMT stand-in for GPAW (no GPAW needed)."""

import csv
import fcntl
import os
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import fcc100
from ase.calculators.emt import EMT
from ase.io import write

from scripts.common import project_config
from scripts.simulations.md import run_md


def adsorbed_structure():
    slab = fcc100("Cu", size=(2, 2, 2), vacuum=6.0)
    top = slab.positions[:, 2].max()
    x, y = slab.positions[-1, :2]
    co2 = Atoms("OCO", positions=[(x - 1.16, y, top + 2.5), (x, y, top + 2.5), (x + 1.16, y, top + 2.5)])
    return slab + co2


def make_results(tmp_path, sites=None):
    """Fake run directory: dft_summary.csv plus one optimized structure per site."""
    sites = sites or {"bridge": ("ok", "base")}
    with (tmp_path / "dft_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["material", "site", "dft_status", "dft_profile"])
        writer.writeheader()
        for site, (status, profile) in sites.items():
            writer.writerow({"material": "ZnO", "site": site, "dft_status": status, "dft_profile": profile})
            case = tmp_path / "dft" / "ZnO" / site
            case.mkdir(parents=True)
            write(case / f"ZnO_{site}_optimized.xyz", adsorbed_structure())
    return tmp_path


@pytest.fixture
def calls():
    return []


@pytest.fixture
def results(tmp_path, monkeypatch, calls):
    def fake_calculator(**kwargs):
        calls.append(kwargs)
        return EMT()

    monkeypatch.setattr(project_config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_md, "build_calculator", fake_calculator)
    return tmp_path


FAST = ["--equilibration-steps", "4", "--production-steps", "6", "--log-interval", "2", "--timestep", "0.5"]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(argv):
    return run_md.run_md(run_md.parse_args(argv))


def test_profile_and_parameters_come_from_dft_summary(results, calls):
    make_results(results, {"bridge": ("ok", "robust")})
    assert run(["--site", "bridge", *FAST]) == 0
    call = calls[0]
    assert call["smearing"] == 0.05 and call["maxiter"] == 280 and call["mixer"] == (0.03, 7, 100)
    assert call["mode"] == "pw" and call["ecut"] == 500.0 and call["kpts"] == (2, 3, 1)
    summary = read_rows(results / "md/ZnO/bridge/md_site_summary.csv")
    assert summary[0]["dft_profile"] == "robust" and summary[0]["seed"] == "1234"


def test_friction_is_converted_from_inverse_fs(results, monkeypatch):
    from ase.md import langevin

    frictions = []
    original = langevin.Langevin

    class Recording(original):
        def __init__(self, *args, **kwargs):
            frictions.append(kwargs["friction"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(langevin, "Langevin", Recording)
    make_results(results)
    assert run(["--site", "bridge", "--friction", "0.02", *FAST]) == 0
    assert frictions == [pytest.approx(0.02 / units.fs)] * 2
    assert run_md.DEFAULTS["friction_per_fs"] == 0.01


def test_equilibration_frames_are_excluded_from_descriptors(results):
    make_results(results)
    assert run(["--site", "bridge", *FAST]) == 0
    seed_dir = results / "md/ZnO/bridge/seed_1234"
    equilibration = read_rows(seed_dir / "md_equilibration_frames.csv")
    production = read_rows(seed_dir / "md_production_frames.csv")
    assert {row["phase"] for row in equilibration} == {"equilibration"}
    assert {row["phase"] for row in production} == {"production"}
    assert [int(row["step"]) for row in production] == [4, 6, 8, 10]
    assert [float(row["time_fs"]) for row in production] == [0.0, 1.0, 2.0, 3.0]
    assert (seed_dir / "md_equilibration.traj").is_file() and (seed_dir / "md_production.traj").is_file()

    summary = read_rows(results / "md/ZnO/bridge/md_site_summary.csv")[0]
    assert int(summary["n_production_frames"]) == 4
    assert int(summary["force_evaluations"]) == 4 + 6 + 1
    expected = np.mean([float(row["temperature_k"]) for row in production])
    assert float(summary["temperature_k_mean"]) == pytest.approx(expected)
    site_frames = read_rows(results / "md/ZnO/bridge/md_production_frames.csv")
    assert site_frames == production


def test_same_seed_reproduces_trajectory(results):
    make_results(results)
    assert run(["--site", "bridge", "--seeds", "7", "8", *FAST]) == 0
    first = read_rows(results / "md/ZnO/bridge/seed_7/md_production_frames.csv")
    other = read_rows(results / "md/ZnO/bridge/seed_8/md_production_frames.csv")
    assert first[-1]["epot_ev"] != other[-1]["epot_ev"]

    assert run(["--site", "bridge", "--seed", "7", "--overwrite", *FAST]) == 0
    again = read_rows(results / "md/ZnO/bridge/seed_7/md_production_frames.csv")
    assert again == first
    # Seed 8 from the first invocation is kept; seed 7 was recomputed.
    summary = read_rows(results / "md/ZnO/bridge/md_site_summary.csv")
    assert sorted(row["seed"] for row in summary) == ["7", "8"]


def test_adding_a_seed_later_keeps_the_earlier_trajectories(results):
    make_results(results)
    assert run(["--site", "bridge", "--seeds", "1234", "2345", *FAST]) == 0
    before = [r for r in read_rows(results / "md/ZnO/bridge/md_production_frames.csv")]
    assert run(["--site", "bridge", "--seeds", "3456", *FAST]) == 0
    frames = read_rows(results / "md/ZnO/bridge/md_production_frames.csv")
    summary = read_rows(results / "md/ZnO/bridge/md_site_summary.csv")
    assert sorted({r["seed"] for r in frames}) == ["1234", "2345", "3456"]
    assert sorted(r["seed"] for r in summary) == ["1234", "2345", "3456"]
    assert [r for r in frames if r["seed"] != "3456"] == before


def test_descriptors_use_minimum_image():
    atoms = Atoms(
        "ZnOOCO",
        positions=[(0.2, 1.0, 5.0), (2.0, 1.0, 5.0), (9.6, 1.0, 7.3), (9.8, 2.1, 7.3), (9.8, 3.26, 7.3)],
        cell=(10.0, 10.0, 20.0),
        pbc=(True, True, False),
    )
    descriptors = run_md.frame_descriptors(atoms, top_layer=[0, 1])
    # C at x=9.8 is 0.4 A from Zn at x=0.2 through the periodic boundary.
    assert descriptors["c_min_slab_distance_ang"] == pytest.approx(np.hypot(0.4, np.hypot(1.1, 2.3)))
    assert descriptors["o_min_metal_distance_ang"] == pytest.approx(np.hypot(0.6, 2.3))
    assert descriptors["c_height_above_slab_plane_ang"] == pytest.approx(2.3)
    assert descriptors["co_bond_2_ang"] == pytest.approx(1.16)


def test_failed_site_is_skipped_and_exit_code_is_nonzero(results):
    make_results(results, {"bridge": ("ok", "base"), "top_metal": ("ads_failed", "none")})
    with pytest.raises(SystemExit) as exit_info:
        run_md.main(["--sites", "top_metal", "bridge", *FAST])
    assert exit_info.value.code == 1
    assert (results / "md/ZnO/bridge/md_site_summary.csv").is_file()
    assert not (results / "md/ZnO/top_metal").exists()
    assert run(["--site", "top_oxygen", *FAST]) == 1  # not in dft_summary.csv


def test_settings_from_run_config_and_legacy_steps(results):
    config = {"methods": {"md": {"steps": 50, "seeds": [3, 4], "friction_per_fs": 0.02},
                          "dft": {"ecut_ev": 500.0, "kpoints": [2, 3, 1]}}}
    settings = run_md.resolve_settings(run_md.parse_args([]), config)
    assert settings["production_steps"] == 50 and settings["seeds"] == [3, 4]
    assert settings["friction_per_fs"] == 0.02 and settings["equilibration_steps"] == run_md.DEFAULTS["equilibration_steps"]

    config["methods"]["md"]["production_steps"] = 80
    assert run_md.resolve_settings(run_md.parse_args(["--steps", "50"]), config)["production_steps"] == 80

    config["methods"]["dft"]["kpoints"] = [3, 3, 1]
    with pytest.raises(SystemExit, match="k-points"):
        run_md.resolve_settings(run_md.parse_args([]), config)


def test_defaults_match_production_yaml():
    import yaml

    config = yaml.safe_load((project_config.ROOT_DIR / "configs/simulations/zno_production.yaml").read_text())
    md = config["methods"]["md"]
    for key, value in run_md.DEFAULTS.items():
        if key != "seeds":
            assert md[key] == value, key
    assert md["kpoints"] == config["methods"]["dft"]["kpoints"]
    assert md["ecut_ev"] == config["methods"]["dft"]["ecut_ev"]


def test_nve_test_reports_drift(results):
    make_results(results)
    assert run(["--site", "bridge", "--nve-test", "--nve-test-steps", "5"]) == 0
    rows = read_rows(results / "md/ZnO/bridge/nve_test/nve_test_summary.csv")
    assert [float(row["timestep_fs"]) for row in rows] == [0.5, 1.0]
    assert all(np.isfinite(float(row["drift_mev_per_atom_per_ps"])) for row in rows)
    assert not (results / "md/ZnO/bridge/md_site_summary.csv").exists()


def test_held_lock_stops_the_seed_before_writing(results, calls, capsys):
    make_results(results)
    seed_dir = results / "md/ZnO/bridge/seed_1234"
    seed_dir.mkdir(parents=True)
    with (seed_dir / run_md.LOCK_FILENAME).open("w") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        holder.write("4242\n")
        holder.flush()
        assert run(["--site", "bridge", *FAST]) == 1
    assert "locked by another run_md (PID 4242" in capsys.readouterr().out
    assert calls == [] and [p.name for p in seed_dir.iterdir()] == [run_md.LOCK_FILENAME]
    assert not (results / "md/ZnO/bridge/md_site_summary.csv").exists()
    # Released: the same seed now runs and the lock records this process.
    assert run(["--site", "bridge", *FAST]) == 0
    assert (seed_dir / run_md.LOCK_FILENAME).read_text().strip() == str(os.getpid())


def test_lock_outcome_is_broadcast_to_every_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(run_md, "is_master", lambda: False)
    monkeypatch.setattr(run_md, "broadcast", lambda value: "locked by another run_md (PID 7)")
    with pytest.raises(run_md.TrajectoryLocked, match="PID 7"):
        with run_md.trajectory_lock(tmp_path / "seed_1"):
            pass
    assert not (tmp_path / "seed_1").exists()  # only rank 0 touches the file


def test_complete_seed_is_skipped_unless_overwrite(results, calls, capsys):
    make_results(results)
    assert run(["--site", "bridge", "--seeds", "1", "2", *FAST]) == 0
    frames = read_rows(results / "md/ZnO/bridge/md_production_frames.csv")
    assert run(["--site", "bridge", "--seeds", "1", "2", *FAST]) == 0
    assert len(calls) == 2 and "seed 1: complete" in capsys.readouterr().out
    assert read_rows(results / "md/ZnO/bridge/md_production_frames.csv") == frames
    assert run(["--site", "bridge", "--seeds", "2", "--overwrite", *FAST]) == 0
    assert len(calls) == 3
    assert sorted(r["seed"] for r in read_rows(results / "md/ZnO/bridge/md_site_summary.csv")) == ["1", "2"]


def test_incomplete_or_different_seed_needs_overwrite(results, calls, capsys):
    make_results(results)
    # An interrupted trajectory: files in seed_5, no rows in the site tables.
    seed_dir = results / "md/ZnO/bridge/seed_5"
    seed_dir.mkdir(parents=True)
    (seed_dir / "md_equilibration.traj").write_bytes(b"partial")
    assert run(["--site", "bridge", "--seeds", "5", *FAST]) == 1
    out = capsys.readouterr().out
    assert "incomplete production (0 of 4 production frames" in out and "--overwrite" in out
    assert calls == [] and (seed_dir / "md_equilibration.traj").read_bytes() == b"partial"
    assert run(["--site", "bridge", "--seeds", "5", "--overwrite", *FAST]) == 0
    assert len(calls) == 1

    # Complete, but with another friction: not skipped and not silently recomputed.
    assert run(["--site", "bridge", "--seeds", "5", "--friction", "0.02", *FAST]) == 1
    assert "recorded with different friction_per_fs" in capsys.readouterr().out
    assert len(calls) == 1


def test_seed_state_counts_frames_and_settings(results):
    make_results(results)
    assert run(["--site", "bridge", "--seeds", "3", *FAST]) == 0
    site_dir = results / "md/ZnO/bridge"
    settings = run_md.resolve_settings(run_md.parse_args(FAST), {})
    assert run_md.seed_state(site_dir, 3, settings, "base")[0] == "complete"
    assert run_md.seed_state(site_dir, 4, settings, "base")[0] == "absent"
    assert run_md.seed_state(site_dir, 3, settings, "robust") == ("different", "recorded with different dft_profile")
    assert run_md.seed_state(site_dir, 3, {**settings, "rng_stream": "per_site"}, "base")[0] == "different"
    frames = read_rows(site_dir / "md_production_frames.csv")
    with (site_dir / "md_production_frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frames[0]))
        writer.writeheader()
        writer.writerows(frames[:-1])
    assert run_md.seed_state(site_dir, 3, settings, "base") == (
        "incomplete", "3 of 4 production frames in md_production_frames.csv")


def test_rng_stream_defaults_to_shared_and_is_recorded(results):
    sites = {"bridge": ("ok", "base"), "top_metal": ("ok", "base")}
    make_results(results, sites)
    assert run(["--sites", "bridge", "top_metal", "--seeds", "9", *FAST]) == 0
    bridge = read_rows(results / "md/ZnO/bridge/md_production_frames.csv")
    top = read_rows(results / "md/ZnO/top_metal/md_production_frames.csv")
    # Same structure and seed: the shared stream gives the same trajectory on both sites.
    assert [r["epot_ev"] for r in bridge] == [r["epot_ev"] for r in top]
    assert {r["rng_stream"] for r in bridge} == {"shared"}
    summary = read_rows(results / "md/ZnO/bridge/md_site_summary.csv")[0]
    assert summary["rng_stream"] == "shared" and summary["omp_num_threads"] == os.environ["OMP_NUM_THREADS"]


def test_per_site_rng_stream_decorrelates_sites(results):
    make_results(results, {"bridge": ("ok", "base"), "top_metal": ("ok", "base")})
    (results / "config.yaml").write_text("methods:\n  md:\n    rng_stream: per_site\n")
    assert run(["--sites", "bridge", "top_metal", "--seeds", "9", *FAST]) == 0
    bridge = read_rows(results / "md/ZnO/bridge/md_production_frames.csv")
    top = read_rows(results / "md/ZnO/top_metal/md_production_frames.csv")
    assert [r["epot_ev"] for r in bridge] != [r["epot_ev"] for r in top]
    assert read_rows(results / "md/ZnO/bridge/md_site_summary.csv")[0]["rng_stream"] == "per_site"

    expected = np.random.default_rng(np.random.SeedSequence([9, run_md.SITE_RNG_INDEX["bridge"]])).random(3)
    assert np.array_equal(run_md.make_rng(9, "bridge", "per_site").random(3), expected)
    assert np.array_equal(run_md.make_rng(9, "bridge", "shared").random(3), np.random.default_rng(9).random(3))
    assert set(project_config.DFT_SITES) <= set(run_md.SITE_RNG_INDEX)
    assert len(set(run_md.SITE_RNG_INDEX.values())) == len(run_md.SITE_RNG_INDEX)
    with pytest.raises(SystemExit, match="rng_stream"):
        run_md.resolve_settings(run_md.parse_args([]), {"methods": {"md": {"rng_stream": "random"}}})


def test_production_yaml_uses_per_site_rng_stream():
    import yaml

    config = yaml.safe_load((project_config.ROOT_DIR / "configs/simulations/zno_production.yaml").read_text())
    assert config["methods"]["md"]["rng_stream"] == "per_site"
    assert run_md.resolve_settings(run_md.parse_args([]), config)["rng_stream"] == "per_site"
    assert run_md.resolve_settings(run_md.parse_args([]), {})["rng_stream"] == "shared"


@pytest.mark.parametrize("module", [
    "scripts.simulations.md.run_md",
    "scripts.simulations.dft.run_dft",
    "scripts.simulations.neb.run_neb",
    "scripts.simulations.neb.relax_neb_image",
    "scripts.simulations.tddft.run_tddft",
    "scripts.simulations.co2rr.run_co2rr_pathways",
])
def test_entry_points_default_to_one_openmp_thread(module):
    code = f"import os, {module}; print(os.environ['OMP_NUM_THREADS'])"
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    run_once = lambda env: subprocess.run([sys.executable, "-c", code], cwd=project_config.ROOT_DIR, env=env,
                                          capture_output=True, text=True, check=True).stdout.strip()
    assert run_once(env) == "1"
    assert run_once({**env, "OMP_NUM_THREADS": "4"}) == "4"  # an explicit value is kept
