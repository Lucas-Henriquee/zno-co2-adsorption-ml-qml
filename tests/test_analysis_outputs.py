"""Explicit inputs and output directories of the analysis CLIs (paper analysis stage).

Every explicit option is honoured or fails; none falls back silently. Without the new
options the outputs stay where they were (RUN or <base>/comparison_plots).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from scripts.analysis import analysis_co2_reduction as co2
from scripts.analysis import neb_energy_table
from scripts.analysis.co2rr_corrections import CORRECTED_CSV
from scripts.visualization import plot_neb
from scripts.visualization import render_adsorption_structures as render

ROOT = Path(__file__).resolve().parents[1]
GEOMETRIES = ROOT / "data" / "geometries" / "ZnO"


def _files(root: Path) -> set[Path]:
    return {path for path in root.rglob("*") if path.is_file()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _quiet(func, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args)


# --------------------------------------------------------------- analysis_co2_reduction

SITES = ("top_metal", "bridge")


def _corrected_rows(u_l: float) -> list[dict]:
    return [
        {
            "material": "ZnO", "site": site, "variant": variant, "status": "ok", "message": "",
            "variant_description": "test", "geometry_flags": "",
            "limiting_potential_v": str(u_l - 0.1 * i), "limiting_step": "CO2->COOH*",
        }
        for i, site in enumerate(SITES)
        for variant in ("a", "c")
    ]


def _co2_run(tmp_path: Path, with_corrected: bool = True) -> Path:
    run = tmp_path / "run"
    _write_csv(run / "dft_summary.csv", [
        {
            "material": "ZnO", "site": site, "dft_status": "ok", "dft_profile": "base",
            "adsorption_energy_ev": "-0.3", "co_bond_1_ang": "1.2", "co_bond_2_ang": "1.2",
            "oco_angle_deg": "150.0", "c_surface_distance_ang": "2.0", "final_state": "physisorbed",
        }
        for site in SITES
    ])
    _write_csv(run / "tddft_summary.csv", [{"material": "ZnO", "site": site} for site in SITES])
    _write_csv(run / "co2rr_pathway_summary.csv", [
        {
            "material": "ZnO", "site": site, "pathway_status": "ok", "energy_basis": "free_energy_CHE",
            "geometry_flags": "", "limiting_potential_v": "-9.0", "limiting_step": "uncorrected",
        }
        for site in SITES
    ])
    if with_corrected:
        _write_csv(run / CORRECTED_CSV, _corrected_rows(-1.0))
    return run


def _limiting_potentials(out: Path) -> dict[str, float]:
    return {row["site"]: float(row["limiting_potential_v"])
            for row in _read_csv(out / "co2rr_pathway_template.csv")}


def test_co2_reduction_default_reads_corrected_summary_of_the_run(tmp_path):
    run = _co2_run(tmp_path)
    out = tmp_path / "analysis" / "science" / "co2_reduction"
    _quiet(co2.main, ["--results-dir", str(run), "--output-dir", str(out)])
    assert _limiting_potentials(out) == {"top_metal": -1.0, "bridge": -1.1}


def test_co2_reduction_default_without_corrected_summary_uses_pathway_summary(tmp_path):
    run = _co2_run(tmp_path, with_corrected=False)
    out = tmp_path / "out"
    _quiet(co2.main, ["--results-dir", str(run), "--output-dir", str(out)])
    assert _limiting_potentials(out) == {"top_metal": -9.0, "bridge": -9.0}


def test_co2_reduction_explicit_corrected_summary_wins(tmp_path):
    run = _co2_run(tmp_path)
    analysis = tmp_path / "analysis"
    corrected = analysis / "science" / "co2rr" / CORRECTED_CSV
    _write_csv(corrected, _corrected_rows(-0.5))
    out = analysis / "science" / "co2_reduction"
    before = _files(tmp_path)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        co2.main([
            "--results-dir", str(run), "--output-dir", str(out),
            "--co2rr-variant", "c", "--corrected-summary", str(corrected),
        ])
    assert _limiting_potentials(out) == {"top_metal": -0.5, "bridge": -0.6}
    assert str(corrected) in log.getvalue()
    created = _files(tmp_path) - before
    assert created and all(out in path.parents for path in created)


def test_co2_reduction_missing_explicit_corrected_summary_fails(tmp_path):
    run = _co2_run(tmp_path)  # the run has its own corrected summary: it must not be used
    out = tmp_path / "out"
    missing = tmp_path / "analysis" / CORRECTED_CSV
    with pytest.raises(SystemExit, match="--corrected-summary"):
        _quiet(co2.main, ["--results-dir", str(run), "--output-dir", str(out),
                          "--corrected-summary", str(missing)])
    assert not (out / "co2rr_pathway_template.csv").exists()


def test_co2_reduction_corrected_summary_excludes_uncorrected(tmp_path):
    run = _co2_run(tmp_path)
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _quiet(co2.main, ["--results-dir", str(run), "--output-dir", str(tmp_path / "out"),
                          "--uncorrected", "--corrected-summary", str(run / CORRECTED_CSV)])


# --------------------------------------------------------------- NEB table and plots

def _neb_base(tmp_path: Path) -> Path:
    atoms = read(GEOMETRIES / "co2_top_metal.traj")
    base = tmp_path / "run" / "neb" / "ZnO"
    profiles = {
        ("top_metal", "bridge"): [0.0, 0.1, 0.3, 0.1, 0.05],
        ("bridge", "top_oxygen"): [0.0, 0.0003, 0.0130, 0.0287, 0.0136],
    }
    for (initial, final), energies in profiles.items():
        folder = base / f"{initial}_to_{final}"
        folder.mkdir(parents=True)
        (folder / "neb_run_summary.json").write_text(json.dumps({
            "material": "ZnO", "initial_site": initial, "final_site": final, "status": "converged",
            "climb_activated": True, "climb_skipped": False,
        }))
        frames = []
        for index, energy in enumerate(energies):
            image = atoms.copy()
            image.positions[-3:, 0] += 0.2 * index  # distinct images along the band
            image.calc = SinglePointCalculator(image, energy=energy, forces=np.zeros((len(image), 3)))
            frames.append(image)
        write(folder / "neb_final_images.traj", frames)
    return base


def test_neb_energy_table_default_output_is_comparison_plots(tmp_path):
    base = _neb_base(tmp_path)
    _quiet(neb_energy_table.main, ["--base", str(base), "--barrier-resolution", "0.03"])
    assert (base / "comparison_plots" / "neb_image_energies.csv").is_file()
    assert (base / "comparison_plots" / "neb_barriers.csv").is_file()


def test_neb_energy_table_output_dir(tmp_path):
    base = _neb_base(tmp_path)
    out = tmp_path / "analysis" / "science" / "neb"
    before = _files(tmp_path)
    _quiet(neb_energy_table.main, ["--base", str(base), "--output-dir", str(out), "--barrier-resolution", "0.03"])
    created = _files(tmp_path) - before
    assert created == {out / "neb_image_energies.csv", out / "neb_barriers.csv"}
    assert not (base / "comparison_plots").exists()
    barriers = {row["pair"]: row for row in _read_csv(out / "neb_barriers.csv")}
    assert float(barriers["top_metal_to_bridge"]["forward_barrier_eV"]) == pytest.approx(0.3)
    assert barriers["bridge_to_top_oxygen"]["barrier_resolved"] == "False"


def test_plot_neb_default_output_is_comparison_plots(tmp_path):
    base = _neb_base(tmp_path)
    _quiet(plot_neb.main, ["--base", str(base), "--dpi", "20", "--barrier-resolution", "0.03"])
    assert (base / "comparison_plots" / "migration_barrier_comparison.png").is_file()
    assert (base / "comparison_plots" / "neb_summary_table.csv").is_file()


@pytest.mark.parametrize("smooth", [False, True])
def test_plot_neb_output_dir(tmp_path, smooth):
    base = _neb_base(tmp_path)
    out = tmp_path / "analysis" / "science" / "neb"
    before = _files(tmp_path)
    argv = ["--base", str(base), "--output-dir", str(out), "--dpi", "20", "--barrier-resolution", "0.03"]
    _quiet(plot_neb.main, argv + (["--smooth-band"] if smooth else []))
    created = _files(tmp_path) - before
    assert all(out in path.parents for path in created)
    assert {path.name for path in created} >= {
        "neb_energy_table.csv", "neb_summary_table.csv", "migration_barrier_comparison.png",
        "neb_profile_top_metal_to_bridge.png", "neb_profile_bridge_to_top_oxygen.png",
    }
    assert (out / "individual_profiles" / "neb_profile_top_metal_to_bridge.png").is_file()
    assert not (base / "comparison_plots").exists()


# --------------------------------------------------------------- adsorption structures

def _structure_run(tmp_path: Path) -> tuple[Path, Path]:
    """RUN/dft/ZnO/<site>/ZnO_<site>_optimized.xyz and RUN/dft_summary.csv from the start geometries."""
    run = tmp_path / "run"
    rows = []
    for site in render.SITES:
        atoms = read(GEOMETRIES / f"co2_{site}.traj")
        # Tilt CO2 out of the x axis (as after a relaxation) so the side view shows both bonds.
        carbon = atoms.positions[-2].copy()
        angle = np.radians(30.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]])
        atoms.positions[-3:] = (atoms.positions[-3:] - carbon) @ rotation.T + carbon
        folder = run / "dft" / "ZnO" / site
        folder.mkdir(parents=True)
        write(folder / f"ZnO_{site}_optimized.xyz", atoms)
        rows.append({"material": "ZnO", "site": site,
                     **{key: f"{value:.6f}" for key, value in render.geometry_metrics(atoms).items()}})
    _write_csv(run / "dft_summary.csv", rows)
    return run / "dft" / "ZnO", run / "dft_summary.csv"


def test_render_structures_with_explicit_run_inputs(tmp_path):
    input_dir, summary = _structure_run(tmp_path)
    out = tmp_path / "analysis" / "science" / "structures"
    before = _files(tmp_path)
    _quiet(render.main, [
        "--input-dir", str(input_dir), "--summary", str(summary), "--output-dir", str(out),
        "--site-metadata", str(GEOMETRIES / "metadata.json"),
        "--initial-slab", str(GEOMETRIES / "clean_slab.traj"), "--dpi", "30",
    ])
    assert _files(tmp_path) - before == {out / "zno_adsorption_sites_combined.png"}


def test_render_structures_missing_inputs_fail(tmp_path):
    input_dir, summary = _structure_run(tmp_path)
    (input_dir / "bridge" / "ZnO_bridge_optimized.xyz").unlink()
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as excinfo:
        _quiet(render.main, [
            "--input-dir", str(input_dir), "--summary", str(tmp_path / "absent.csv"),
            "--output-dir", str(out), "--site-metadata", str(GEOMETRIES / "metadata.json"),
            "--initial-slab", str(GEOMETRIES / "clean_slab.traj"),
        ])
    message = str(excinfo.value)
    assert "--summary" in message and "absent.csv" in message
    assert "ZnO_bridge_optimized.xyz" in message
    assert "ZnO_top_metal_optimized.xyz" not in message
    assert not out.exists()
