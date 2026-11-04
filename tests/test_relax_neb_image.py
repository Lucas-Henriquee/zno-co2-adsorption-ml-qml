"""relax_neb_image: relax one NEB image with the run's DFT settings (EMT stands in for GPAW)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import write

from scripts.simulations.dft import run_dft
from scripts.simulations.neb import relax_neb_image as rni

ROOT = Path(__file__).resolve().parents[1]
MAT = "ZnO"
PAIR = "top_metal_to_bridge"


def _slab() -> Atoms:
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=6.0)
    slab.pbc = (True, True, False)
    slab.info.pop("adsorbate_info", None)  # not writable to extxyz
    slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 3]))  # bottom layer
    return slab


def _with_co2(x_shift: float) -> Atoms:
    slab = _slab()
    z = slab.positions[:, 2].max() + 2.6
    x, y = slab.positions[-1, 0] + x_shift, slab.positions[-1, 1]
    co2 = Atoms("OCO", positions=[(x - 1.17, y, z), (x, y, z), (x + 1.17, y, z)])
    atoms = slab + co2
    atoms.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 3]))
    return atoms


def _energy(atoms: Atoms) -> float:
    probe = atoms.copy()
    probe.calc = EMT()
    return float(probe.get_potential_energy())


@pytest.fixture()
def run(tmp_path):
    run = tmp_path / "RUN"
    run.mkdir()
    (run / "config.yaml").write_text(
        "methods:\n  dft: {fmax_ev_per_angstrom: 0.05, ecut_ev: 500.0, kpoints: [2, 3, 1], "
        "scf_force_convergence: null}\n",
        encoding="utf-8",
    )
    endpoints = {"top_metal": _with_co2(0.0), "bridge": _with_co2(1.3)}
    clean = _energy(_slab())
    co2 = _energy(Atoms("OCO", positions=[(0, 0, 0), (1.17, 0, 0), (2.34, 0, 0)], cell=[12, 12, 12]))
    rows = []
    for site, atoms in endpoints.items():
        folder = run / "dft" / MAT / site
        folder.mkdir(parents=True)
        write(folder / f"{MAT}_{site}_optimized.xyz", atoms)
        energy = _energy(atoms)
        rows.append({"material": MAT, "site": site, "dft_status": "ok", "dft_profile": "base",
                     "adsorbed_energy_ev": energy, "clean_slab_energy_ev": clean,
                     "co2_reference_energy_ev": co2, "adsorption_energy_ev": energy - clean - co2})
    with (run / "dft_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Final band: 5 frames between the endpoints, energies stored as run_neb writes them.
    band = []
    for k in range(5):
        image = endpoints["top_metal"].copy()
        image.positions = endpoints["top_metal"].positions + k / 4 * (
            endpoints["bridge"].positions - endpoints["top_metal"].positions)
        image.positions[-3:, 2] += 0.3 * (k not in (0, 4))  # internal images a little higher
        image.calc = SinglePointCalculator(image, energy=_energy(image), forces=image.get_positions() * 0)
        band.append(image)
    pair_dir = run / "neb" / MAT / PAIR
    pair_dir.mkdir(parents=True)
    write(pair_dir / rni.FINAL_IMAGES_TRAJ, band)
    (pair_dir / rni.NEB_SUMMARY_JSON).write_text(
        json.dumps({"ecut": 500.0, "kpts": [2, 3, 1], "dft_profile": "base"}), encoding="utf-8")
    return run


def _relax(run: Path, *extra: str, calls: list | None = None) -> int:
    def fake_build_calculator(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return EMT()

    with mock.patch.object(run_dft, "build_calculator", side_effect=fake_build_calculator):
        return rni.main(["--results-dir", str(run), "--pair", PAIR, "--image", "2", "--fmax", "0.05",
                         "--steps", "300", *extra])


def test_relaxes_the_image_and_writes_energies(run):
    calls = []
    assert _relax(run, calls=calls) == 0
    out = run / "neb" / MAT / PAIR / "image2_relaxed"
    assert {p.name for p in out.iterdir()} >= {"image2_relaxed.xyz", "relax.traj", "relax.log",
                                                 "image2_relaxed.json"}
    record = json.loads((out / "image2_relaxed.json").read_text())
    assert record["converged"] is True and record["final_fmax_ev_per_ang"] <= 0.05
    assert 0 < record["steps"] <= 300
    # Exactly the run's DFT settings: base profile, ecut and k-points of methods.dft.
    assert calls == [{"ecut": 500.0, "kpts": (2, 3, 1), "txt_file": out / "gpaw.txt", "mode": "pw",
                      "smearing": 0.05, "maxiter": 180, "mixer": (0.05, 5, 50), "scf_force_convergence": None}]
    assert record["calculator"]["profile"] == "base"
    # Relative energies and the implied adsorption energy use the dft_summary references.
    rows = {row["site"]: row for row in csv.DictReader((run / "dft_summary.csv").open())}
    energy = record["energy_ev"]
    assert record["e_rel_initial_ev"] == pytest.approx(energy - float(rows["top_metal"]["adsorbed_energy_ev"]))
    assert record["e_rel_final_ev"] == pytest.approx(energy - float(rows["bridge"]["adsorbed_energy_ev"]))
    assert record["implied_adsorption_energy_ev"] == pytest.approx(
        energy - float(rows["top_metal"]["clean_slab_energy_ev"]) - float(rows["top_metal"]["co2_reference_energy_ev"]))
    assert record["references_match_between_endpoints"] is True
    assert record["relaxation_energy_ev"] < 0  # the start image was raised above the band
    # FixAtoms of the endpoint DFT structure (bottom Cu layer).
    assert record["fixed_indices"] == [0, 1, 2, 3]
    assert record["fixed_indices_source"].endswith(f"{MAT}_top_metal_optimized.xyz")


def test_refuses_to_overwrite_and_endpoints(run):
    assert _relax(run) == 0
    with pytest.raises(SystemExit, match="--overwrite"):
        _relax(run)
    assert _relax(run, "--overwrite") == 0
    with mock.patch.object(run_dft, "build_calculator", side_effect=lambda **kw: EMT()):
        for image in ("0", "4"):
            with pytest.raises(SystemExit, match="internal image"):
                rni.main(["--results-dir", str(run), "--pair", PAIR, "--image", image])


def test_profile_and_setting_mismatch_are_errors(run):
    summary = run / "neb" / MAT / PAIR / rni.NEB_SUMMARY_JSON
    summary.write_text(json.dumps({"ecut": 400.0, "kpts": [2, 3, 1]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="ecut"):
        _relax(run)
    summary.write_text(json.dumps({"ecut": 500.0, "kpts": [2, 3, 1], "dft_profile": "robust"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="profile"):
        _relax(run)


def test_pair_names():
    assert rni.pair_sites("top_metal_to_bridge") == ("top_metal", "bridge")
    assert rni.pair_sites("bridge:top_oxygen") == ("bridge", "top_oxygen")
    with pytest.raises(SystemExit):
        rni.pair_sites("top_metal_to_carbonate")


def test_module_import_does_not_need_gpaw():
    code = ("import sys; import scripts.simulations.neb.relax_neb_image as m; "
            "m.parse_args(['--results-dir', '.', '--pair', 'a_to_b', '--image', '1']); "
            "assert 'gpaw' not in sys.modules, 'gpaw imported'")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
