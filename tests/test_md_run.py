"""MD driver checks with an EMT stand-in for GPAW (no GPAW needed)."""

import csv

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
    assert call["smearing"] == 0.15 and call["maxiter"] == 280 and call["mixer"] == (0.03, 7, 100)
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

    assert run(["--site", "bridge", "--seed", "7", *FAST]) == 0
    again = read_rows(results / "md/ZnO/bridge/seed_7/md_production_frames.csv")
    assert again == first
    summary = read_rows(results / "md/ZnO/bridge/md_site_summary.csv")
    assert [row["seed"] for row in summary] == ["7"]


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
    assert settings["friction_per_fs"] == 0.02 and settings["equilibration_steps"] == 1000

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
