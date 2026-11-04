"""export_pymol: PDB/CIF of the relaxed structures and the Fig. 1 PyMOL script (synthetic slabs)."""

from __future__ import annotations

import csv
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from scripts.visualization import export_pymol

MAT = "ZnO"
SITES = ("top_metal", "top_oxygen", "bridge")
LAYERS = (10.0, 10.9, 12.8, 13.7)  # two bilayers: the exposed one is 12.8 / 13.7 A


def _slab_with_co2(c_xy=(1.0, 1.0)) -> Atoms:
    symbols, positions = [], []
    for z in LAYERS:
        for x, y in ((0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)):
            symbols.append("Zn" if z in (10.0, 12.8) else "O")
            positions.append((x + (0.0 if z in (10.0, 12.8) else 1.5), y + 0.5, z))
    cx, cy = c_xy
    symbols += ["O", "C", "O"]
    positions += [(cx - 1.16, cy, 16.0), (cx, cy, 16.0), (cx + 1.16, cy, 16.0)]
    return Atoms(symbols, positions=positions, cell=[6.0, 6.0, 24.0], pbc=[True, True, False])


@pytest.fixture()
def run(tmp_path):
    run = tmp_path / "RUN"
    rows = []
    for i, site in enumerate(SITES):
        atoms = _slab_with_co2((1.0 + i, 0.3) if site != "bridge" else (5.9, 3.0))
        path = run / "dft" / MAT / site / f"{MAT}_{site}_optimized.xyz"
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, atoms, format="extxyz")
        rows.append({"material": MAT, "site": site, "dft_status": "failed" if site == "top_oxygen" else "ok"})
    with (run / "dft_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run


def _export(run: Path, out: Path, *extra: str) -> int:
    return export_pymol.main([
        "--input-dir", str(run / "dft" / MAT), "--summary", str(run / "dft_summary.csv"),
        "--output-dir", str(out), *extra,
    ])


def test_surface_threshold():
    assert export_pymol.surface_threshold(np.array(LAYERS)) == pytest.approx(0.5 * (12.8 + 10.9))
    assert export_pymol.surface_threshold(np.array([5.0, 5.3, 5.6])) == pytest.approx(4.6)  # no gap


def test_files_written_for_ok_sites(run, tmp_path):
    out = tmp_path / "pymol"
    assert _export(run, out, "--render", "none") == 0
    names = {p.name for p in out.iterdir()}
    assert names == {"top_metal.pdb", "top_metal.cif", "bridge.pdb", "bridge.cif",
                     "fig01_adsorption_structures.pml", "README.txt", "render_info.json"}  # top_oxygen failed
    info = json.loads((out / "render_info.json").read_text())
    assert info["renderer"] == "none" and info["requested"] == "none"
    assert info["panels"] == {} and info["combined"] is None
    assert info["panels_contact"] == {} and info["combined_contact"] is None
    assert set(info["contacts"]) == {"top_metal", "bridge"}
    pdb = (out / "bridge.pdb").read_text()
    assert "CRYST1" in pdb and " CO2 " in pdb and " SLB " in pdb
    atoms = read(out / "bridge.pdb", format="proteindatabank")
    source = read(run / "dft" / MAT / "bridge" / f"{MAT}_bridge_optimized.xyz")
    assert atoms.get_chemical_symbols() == source.get_chemical_symbols()
    # Lateral periodic shift only: z unchanged, carbon at the cell centre, CO2 intact.
    assert atoms.positions[:, 2] == pytest.approx(source.positions[:, 2], abs=1e-3)
    assert atoms.positions[-2, :2] == pytest.approx([3.0, 3.0], abs=1e-3)
    assert atoms.get_distance(-2, -3) == pytest.approx(1.16, abs=1e-3)
    assert atoms.get_distance(-2, -1) == pytest.approx(1.16, abs=1e-3)
    cif = read(out / "bridge.cif")
    assert len(cif) == len(source)
    readme = (out / "README.txt").read_text()
    assert "pymol fig01_adsorption_structures.pml" in readme


def test_pml_references_files_and_threshold(run, tmp_path):
    out = tmp_path / "pymol"
    _export(run, out, "--render", "none")
    pml = (out / "fig01_adsorption_structures.pml").read_text()
    assert "load top_metal.pdb, top_metal" in pml and "load bridge.pdb, bridge" in pml
    assert "top_oxygen" not in pml
    threshold = 0.5 * (12.8 + 10.9)
    assert f"z > {threshold:.3f} A" in pml
    ids = re.search(r"select top_metal_surface, top_metal and resn SLB and id ([0-9+]+)", pml).group(1)
    assert [int(i) for i in ids.split("+")] == list(range(9, 17))  # the 8 atoms of the exposed bilayer
    for setting in ("bg_color white", "set orthoscopic, 1", "show spheres, top_metal_shown",
                    "show sticks, top_metal_shown", "color fig01_zn, top_metal and elem Zn"):
        assert setting in pml
    assert pml.count("rotate x, -70.0,") == 2  # same orientation for every site
    for site in ("top_metal", "bridge"):
        assert f"# png fig01_{site}.png, dpi=600" in pml and "# ray 1600, 1600" in pml
        assert f"# png fig01_{site}_contact.png, dpi=600" in pml
        assert f"distance {site}_dist_contact, {site} and id " in pml
        assert f"set dash_gap, 0.15, {site}_dist_contact" in pml
    assert not re.search(r"^(ray|png)\b", pml, flags=re.M)  # ray/png lines stay commented


def test_selected_sites_only(run, tmp_path):
    out = tmp_path / "pymol"
    assert _export(run, out, "--sites", "bridge", "--render", "none") == 0
    assert {p.name for p in out.glob("*.pdb")} == {"bridge.pdb"}
    pml = (out / "fig01_adsorption_structures.pml").read_text()
    assert "load bridge.pdb" in pml and "top_metal" not in pml
    with pytest.raises(SystemExit, match="top_oxygen"):
        _export(run, tmp_path / "other", "--sites", "top_oxygen")


def test_missing_structure_is_an_error(run, tmp_path):
    (run / "dft" / MAT / "bridge" / f"{MAT}_bridge_optimized.xyz").unlink()
    with pytest.raises(SystemExit, match="ZnO_bridge_optimized.xyz"):
        _export(run, tmp_path / "pymol")


# ---------------------------------------------------------------- renderers


def _png_size(path: Path) -> tuple[int, int]:
    from matplotlib.image import imread

    image = imread(path)
    return image.shape[1], image.shape[0]


def test_ase_renderer_writes_panels_and_combined_figure(run, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_pymol, "pymol_available", lambda: False)
    out = tmp_path / "pymol"
    assert _export(run, out, "--dpi", "100") == 0  # auto -> ase without PyMOL
    assert "renderer: ase (requested auto)" in capsys.readouterr().out
    info = json.loads((out / "render_info.json").read_text())
    contacts = info.pop("contacts")
    assert info == {
        "requested": "auto", "renderer": "ase", "dpi": 100, "tilt_deg": 20.0,
        "panels": {"top_metal": "fig01_top_metal.png", "bridge": "fig01_bridge.png"},
        "combined": "fig01_adsorption_structures.png",
        "panels_contact": {"top_metal": "fig01_top_metal_contact.png", "bridge": "fig01_bridge_contact.png"},
        "combined_contact": "fig01_adsorption_structures_contact.png",
    }
    assert set(contacts) == {"top_metal", "bridge"}
    from matplotlib.image import imread

    for name in ("fig01_top_metal.png", "fig01_bridge.png", "fig01_adsorption_structures.png",
                 "fig01_top_metal_contact.png", "fig01_adsorption_structures_contact.png"):
        image = imread(out / name)
        assert image[..., :3].min() < 0.5  # something drawn
        assert image[0, 0, :3] == pytest.approx([1.0, 1.0, 1.0])  # white background
    # Same scale for every site: panels of the same size; combined = panels side by side.
    assert _png_size(out / "fig01_top_metal.png") == _png_size(out / "fig01_bridge.png")
    width, height = _png_size(out / "fig01_adsorption_structures.png")
    panel_w, panel_h = _png_size(out / "fig01_bridge.png")
    assert width == 700  # 7 in at 100 dpi; two panels of 3.5 in + 0.6 in for labels and legend
    assert height == pytest.approx(panel_h * 350 / panel_w + 60, abs=2)
    assert _png_size(out / "fig01_adsorption_structures_contact.png") == (width, height)
    # The two variants differ only in the annotation.
    assert not np.array_equal(imread(out / "fig01_bridge.png"), imread(out / "fig01_bridge_contact.png"))


def test_shortest_contact_uses_the_minimum_image():
    atoms = _slab_with_co2((1.0, 0.3))
    centred = export_pymol.centred(atoms)
    positions = export_pymol.pdb_positions(centred)
    contact = export_pymol.shortest_contact(centred, positions)
    slab = len(atoms) - 3
    from ase.geometry import get_distances

    _, d = get_distances(centred.positions[slab:], centred.positions[:slab], cell=centred.cell, pbc=centred.pbc)
    assert contact["distance"] == pytest.approx(d.min())
    assert contact["co2_index"] >= slab and contact["slab_index"] < slab
    assert np.linalg.norm(contact["endpoint"] - positions[contact["co2_index"]]) == pytest.approx(d.min())
    assert contact["pair"] in {"O–Zn", "O–O", "C–Zn", "C–O"}
    # Upright CO2 standing on one O over a surface O: the lower O makes the contact, not C.
    upright = atoms.copy()
    surface_o = upright.positions[12]  # exposed-layer O at z = 13.7
    upright.positions[-3] = surface_o + [0.0, 0.0, 2.0]
    upright.positions[-2] = surface_o + [0.0, 0.0, 3.16]
    upright.positions[-1] = surface_o + [0.0, 0.0, 4.32]
    contact = export_pymol.shortest_contact(upright, upright.get_positions())
    assert (contact["co2_index"], contact["slab_index"], contact["pair"]) == (len(upright) - 3, 12, "O–O")
    assert contact["distance"] == pytest.approx(2.0) and contact["in_cell"]


def test_view_rotation_matches_pml():
    rotation = export_pymol.view_rotation(20.0)
    normal = rotation @ np.array([0.0, 0.0, 1.0])
    # Surface normal mostly up on screen and tilted 20 degrees towards the viewer.
    assert normal == pytest.approx([0.0, np.sin(np.radians(70.0)), np.cos(np.radians(70.0))])
    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([1.0, 0.0, 0.0])


def test_bonds_use_covalent_radii():
    symbols = ["Zn", "O", "C", "O", "O"]
    positions = np.array([[0, 0, 0], [1.98, 0, 0], [0, 0, 5.0], [0, 0, 6.16], [0, 0, 8.0]], dtype=float)
    assert export_pymol.bonds(symbols, positions) == [(0, 1), (2, 3)]


def test_render_pymol_requires_pymol(run, tmp_path, monkeypatch):
    monkeypatch.setattr(export_pymol, "pymol_available", lambda: False)
    with pytest.raises(SystemExit, match="pymol-open-source"):
        _export(run, tmp_path / "pymol", "--render", "pymol")
    assert not (tmp_path / "pymol").exists()


class FakeCmd:
    """Records pymol.cmd calls; png writes a small white image."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "png":
                from matplotlib.image import imsave

                imsave(args[0], np.ones((40, 50, 3)))
        return call


def _fake_pymol(monkeypatch, with_pymol2: bool):
    cmd = FakeCmd()
    state = {}
    pymol = types.ModuleType("pymol")
    pymol.cmd = cmd
    pymol.finish_launching = lambda argv: state.setdefault("launched", argv)
    monkeypatch.setitem(sys.modules, "pymol", pymol)
    if with_pymol2:
        pymol2 = types.ModuleType("pymol2")

        class PyMOL:
            def __init__(self):
                self.cmd = cmd

            def start(self):
                state["started"] = True

            def stop(self):
                state["stopped"] = True

        pymol2.PyMOL = PyMOL
        monkeypatch.setitem(sys.modules, "pymol2", pymol2)
    else:
        monkeypatch.setitem(sys.modules, "pymol2", None)  # import pymol2 -> ImportError
    return cmd, state


@pytest.mark.parametrize("with_pymol2", [True, False])
def test_pymol_renderer_runs_the_pml_commands(run, tmp_path, monkeypatch, capsys, with_pymol2):
    cmd, state = _fake_pymol(monkeypatch, with_pymol2)
    out = tmp_path / "pymol"
    assert _export(run, out, "--dpi", "300") == 0  # auto -> pymol when importable
    assert "renderer: pymol (requested auto)" in capsys.readouterr().out
    if with_pymol2:
        assert state == {"started": True, "stopped": True}
    else:
        assert state == {"launched": ["pymol", "-qc"]}
    info = json.loads((out / "render_info.json").read_text())
    assert info["renderer"] == "pymol" and info["combined"] == "fig01_adsorption_structures.png"
    assert info["combined_contact"] == "fig01_adsorption_structures_contact.png"
    assert (out / "fig01_adsorption_structures.png").is_file()
    assert (out / "fig01_adsorption_structures_contact.png").is_file()

    calls = cmd.calls
    assert calls[0] == ("reinitialize", (), {})
    assert ("load", (str(out / "top_metal.pdb"), "top_metal"), {}) in calls
    assert ("rotate", ("x", -70.0, "bridge"), {"camera": 0, "origin": [0, 0, 0]}) in calls
    assert ("set_color", ("fig01_zn", [0.49, 0.5, 0.69]), {}) in calls
    # The scene commands are the non-comment lines of the .pml up to "deselect".
    pml = (out / "fig01_adsorption_structures.pml").read_text().splitlines()
    scene = [line for line in pml[:pml.index("deselect") + 1] if line and not line.startswith("#")]
    recorded = [export_pymol.pml_line(call) for call in calls[:len(scene)]]
    assert recorded == [line.replace("load top_metal.pdb", f"load {out / 'top_metal.pdb'}")
                        .replace("load bridge.pdb", f"load {out / 'bridge.pdb'}") for line in scene]
    # One PyMOL distance object per site for the contact (between the two atoms).
    contact_calls = [c for c in calls if c[0] == "distance" and c[1][0].endswith("_dist_contact")]
    assert [c[1][0] for c in contact_calls] == ["top_metal_dist_contact", "bridge_dist_contact"]
    for call in contact_calls:
        site = call[1][0].removesuffix("_dist_contact")
        ids = info["contacts"][site]
        assert call[1][1:] == (f"{site} and id {ids['co2_atom_id']}", f"{site} and id {ids['slab_atom_id']}")
    # Then two blocks per site (C-O labels, contact): the commented render blocks of the .pml, executed.
    rest = calls[len(scene):]
    assert [c[0] for c in rest] == ["disable", "enable", "enable", "reset", "zoom", "ray", "png"] * 4
    assert rest[1] == ("enable", ("top_metal",), {}) and rest[4] == ("zoom", ("top_metal_shown", 1.5), {})
    assert rest[2] == ("enable", ("top_metal_dist_co*",), {})
    assert rest[9] == ("enable", ("top_metal_dist_contact",), {})
    assert rest[5] == ("ray", (1600, 1600), {})
    assert rest[6] == ("png", (str(out / "fig01_top_metal.png"),), {"dpi": 300})
    assert rest[13] == ("png", (str(out / "fig01_top_metal_contact.png"),), {"dpi": 300})
    assert rest[20] == ("png", (str(out / "fig01_bridge.png"),), {"dpi": 300})
    assert rest[27] == ("png", (str(out / "fig01_bridge_contact.png"),), {"dpi": 300})
    for call in rest[:14]:
        assert f"# {export_pymol.pml_line(call)}".replace(str(out) + "/", "") in pml
