"""Export the relaxed ZnO-CO2 structures, a PyMOL script and the rendered Fig. 1 (3D views).

For every site with ``dft_status`` ok in ``--summary`` it reads
``--input-dir/<site>/<MAT>_<site>_optimized.xyz`` and writes to ``--output-dir``:

* ``<site>.pdb`` (residue ``SLB`` = slab, ``CO2`` = adsorbate; read by the script) and
  ``<site>.cif``: the relaxed structure translated laterally (a lattice-periodic shift,
  slab wrapped into the cell) so that the CO2 carbon sits at the cell centre; the
  coordinates are otherwise the saved DFT ones;
* ``fig01_adsorption_structures.pml``: loads the PDB files and sets one publication view
  (sticks + spheres, element colours, white background, orthographic, the same tilted
  side view for every site); only the exposed surface layer (slab atoms above a z
  threshold computed from each structure and written into the script) and CO2 are shown;
  commented ``ray``/``png`` lines save ``fig01_<site>.png``;
* ``fig01_<site>.png`` and ``fig01_adsorption_structures.png`` (panels side by side,
  labelled (a), (b), (c)), drawn by the renderer chosen with ``--render``:

  - ``pymol``: PyMOL (``pymol-open-source`` as a Python package) run headless, executing
    the same commands as the ``.pml`` (one command list drives both);
  - ``ase``: a ball-and-stick renderer that needs only ASE + matplotlib (same atoms, same
    orientation, element colours, sticks, C-O lengths);
  - ``auto`` (default): ``pymol`` when ``import pymol`` works, else ``ase``;
  - ``none``: no PNG (structures + script only);
* ``render_info.json``: the renderer used and the PNG names;
* ``README.txt``: the files and how to re-render in PyMOL.

The surface threshold is the midpoint of the first gap larger than ``--layer-gap`` (A)
below the topmost slab atom; without such a gap it is 1 A below the topmost slab atom.
The CO2 molecule is the last three atoms (O, C, O), as written by the DFT stage.
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from ase.io import read, write

SITE_ORDER = ("top_metal", "top_oxygen", "bridge")
SITE_TITLES = {"top_metal": "Top metal", "top_oxygen": "Top oxygen", "bridge": "Bridge"}
PML_NAME = "fig01_adsorption_structures.pml"
COMBINED_NAME = "fig01_adsorption_structures.png"
RENDER_INFO_NAME = "render_info.json"
README_NAME = "README.txt"
RENDERERS = ("auto", "pymol", "ase", "none")
DEFAULT_LAYER_GAP_ANG = 1.2
DEFAULT_TILT_DEG = 20.0
DEFAULT_DPI = 600
FALLBACK_DEPTH_ANG = 1.0
RAY_SIZE_PX = (1600, 1600)
ZOOM_BUFFER_ANG = 1.5
# RGB of the element colours (Zn grey-violet, O red, C dark grey), as in the draft Fig. 1.
ELEMENT_COLORS = {
    "Zn": (0.49, 0.50, 0.69),
    "O": (0.80, 0.22, 0.24),
    "C": (0.25, 0.26, 0.29),
}
STICK_GREY = 0.70  # PyMOL grey70
STICK_RADIUS_ANG = 0.12
SPHERE_SCALE = 0.38  # ase renderer: sphere radius = SPHERE_SCALE * covalent radius
BOND_TOLERANCE = 1.15  # bond when d < BOND_TOLERANCE * (sum of covalent radii)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="RUN/dft/<MAT> with <site>/*_optimized.xyz.")
    parser.add_argument("--summary", type=Path, required=True, help="RUN/dft_summary.csv.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--sites", nargs="+", default=None,
                        help="Sites to export (default: every site with dft_status ok).")
    parser.add_argument("--layer-gap", type=float, default=DEFAULT_LAYER_GAP_ANG,
                        help="Smallest z gap (A) separating the exposed layer from the one below.")
    parser.add_argument("--tilt-deg", type=float, default=DEFAULT_TILT_DEG,
                        help="Tilt of the side view towards the surface normal (degrees).")
    parser.add_argument("--render", choices=RENDERERS, default="auto",
                        help="PNG renderer: pymol (headless PyMOL), ase (ASE + matplotlib), "
                             "auto (pymol when importable, else ase) or none (files + .pml only).")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Resolution of the PNG files.")
    return parser.parse_args(argv)


def ok_sites(summary: Path, material: str) -> list[str]:
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("material") == material]
    present = [row["site"] for row in rows if row.get("dft_status") == "ok"]
    return [s for s in SITE_ORDER if s in present] + sorted(s for s in present if s not in SITE_ORDER)


def check_co2(atoms: Atoms, site: str) -> None:
    if len(atoms) < 4 or atoms[-3:].get_chemical_symbols() != ["O", "C", "O"]:
        raise SystemExit(f"[export-pymol] {site}: expected a slab followed by O, C, O")


def centred(atoms: Atoms) -> Atoms:
    """Lattice-periodic lateral shift putting the carbon at the cell centre; CO2 kept whole."""
    atoms = atoms.copy()
    pbc = atoms.get_pbc()
    if not pbc[:2].all():
        return atoms
    carbon = len(atoms) - 2
    frac_c = atoms.cell.scaled_positions(atoms.positions[[carbon]])[0]
    shift = atoms.cell.cartesian_positions(np.array([[0.5 - frac_c[0], 0.5 - frac_c[1], 0.0]]))[0]
    atoms.positions += shift
    slab = len(atoms) - 3
    scaled = atoms.get_scaled_positions(wrap=False)
    scaled[:slab, :2] %= 1.0
    atoms.set_scaled_positions(scaled)
    # The two CO2 oxygens: minimum image relative to the carbon.
    for oxygen in (slab, slab + 2):
        vector, _ = find_mic(atoms.positions[oxygen] - atoms.positions[carbon], atoms.cell, pbc)
        atoms.positions[oxygen] = atoms.positions[carbon] + vector
    return atoms


def pdb_positions(atoms: Atoms) -> np.ndarray:
    """Coordinates as ASE writes them to PDB (cell in standard form when periodic)."""
    positions = atoms.get_positions()
    if atoms.get_pbc().any():
        _, rotation = atoms.cell.standard_form()
        positions = positions @ rotation.T
    return positions


def surface_threshold(z_slab: np.ndarray, layer_gap: float = DEFAULT_LAYER_GAP_ANG) -> float:
    """z (A) separating the exposed surface layer from the layer below."""
    levels = np.sort(np.asarray(z_slab, dtype=float))[::-1]
    for upper, lower in zip(levels[:-1], levels[1:]):
        if upper - lower > layer_gap:
            return float(0.5 * (upper + lower))
    return float(levels[0] - FALLBACK_DEPTH_ANG)


def write_structure(atoms: Atoms, output_dir: Path, site: str) -> tuple[Path, Path]:
    atoms = atoms.copy()
    slab = len(atoms) - 3
    atoms.set_array("residuenames", np.array(["SLB"] * slab + ["CO2"] * 3, dtype=object))
    atoms.set_array("residuenumbers", np.array([1] * slab + [2] * 3))
    pdb, cif = output_dir / f"{site}.pdb", output_dir / f"{site}.cif"
    write(pdb, atoms, format="proteindatabank")
    for name in ("residuenames", "residuenumbers"):
        del atoms.arrays[name]
    write(cif, atoms, format="cif")
    return pdb, cif


# ---------------------------------------------------------------- PyMOL commands
#
# One command list is used both to write the .pml and to drive pymol.cmd: each item is
# ("#", comment), ("",) for a blank line, or (name, args, kwargs) for cmd.<name>(*args, **kwargs),
# written to the script as "name arg1, arg2, key=value".


def _cmd(name: str, *args, **kwargs) -> tuple:
    return (name, args, kwargs)


def _pml_value(value) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(round(value, 3))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_pml_value(v) for v in value) + "]"
    return str(value)


def pml_line(op: tuple) -> str:
    if op[0] == "#":
        return f"# {op[1]}"
    if op[0] == "":
        return ""
    name, args, kwargs = op
    parts = [_pml_value(a) for a in args] + [f"{k}={_pml_value(v)}" for k, v in kwargs.items()]
    return f"{name} {', '.join(parts)}" if parts else name


def scene_commands(entries: list[dict], tilt_deg: float) -> list[tuple]:
    """Global settings, then load / select / style / orient / label every site."""
    ops = [
        _cmd("reinitialize"),
        _cmd("bg_color", "white"),
        _cmd("set", "orthoscopic", 1),
        _cmd("set", "ray_opaque_background", 1),
        _cmd("set", "antialias", 2),
        _cmd("set", "ray_shadows", 0),
        _cmd("set", "depth_cue", 0),
        _cmd("set", "specular", 0.25),
        _cmd("set", "sphere_scale", 0.30),
        _cmd("set", "stick_radius", STICK_RADIUS_ANG),
        _cmd("set", "stick_color", "grey70"),
        _cmd("set", "label_size", 18),
        _cmd("set", "label_color", "black"),
        _cmd("set", "label_distance_digits", 3),
        _cmd("set", "dash_color", "grey50"),
        _cmd("set", "dash_gap", 0),
        _cmd("set", "dash_radius", 0.0),
        _cmd("set", "auto_zoom", 0),
    ]
    for symbol, rgb in ELEMENT_COLORS.items():
        ops.append(_cmd("set_color", f"fig01_{symbol.lower()}", [round(c, 2) for c in rgb]))
    ops.append(("",))
    for entry in entries:
        site = entry["site"]
        ids = "+".join(str(i) for i in entry["surface_ids"])
        ops += [
            ("#", f"---- {site} ({SITE_TITLES.get(site, site)}): exposed layer = slab atoms with z > "
                  f"{entry['threshold']:.3f} A"),
            _cmd("load", entry["pdb"], site),
            _cmd("select", f"{site}_co2", f"{site} and resn CO2"),
            ("#", f"ids of the slab atoms with z > {entry['threshold']:.3f} A in {entry['pdb']}"),
            _cmd("select", f"{site}_surface", f"{site} and resn SLB and id {ids}"),
            _cmd("select", f"{site}_shown", f"{site}_co2 or {site}_surface"),
            _cmd("hide", "everything", site),
            _cmd("show", "sticks", f"{site}_shown"),
            _cmd("show", "spheres", f"{site}_shown"),
            _cmd("color", "fig01_zn", f"{site} and elem Zn"),
            _cmd("color", "fig01_o", f"{site} and elem O"),
            _cmd("color", "fig01_c", f"{site} and elem C"),
            _cmd("set", "stick_color", "grey70", site),
            # Same orientation for every site: surface normal (z) up, viewed from the side
            # and tilted by the same angle towards the normal (rotation of the model about x,
            # before the distance objects are made so that they follow the atoms).
            _cmd("rotate", "x", float(round(-(90.0 - tilt_deg), 1)), site, camera=0, origin=[0, 0, 0]),
        ]
        for k, (o_id, length) in enumerate(zip(entry["o_ids"], entry["co_lengths"]), start=1):
            ops += [
                ("#", f"C-O length {length:.3f} A"),
                _cmd("distance", f"{site}_dist_co{k}", f"{site} and id {entry['c_id']}", f"{site} and id {o_id}"),
            ]
        ops += [
            _cmd("hide", "dashes", f"{site}_dist_co*"),
            _cmd("show", "labels", f"{site}_dist_co*"),
            ("",),
        ]
    ops.append(_cmd("deselect"))
    return ops


def view_commands(site: str) -> list[tuple]:
    """Show one site alone with the common view."""
    return [_cmd("disable"), _cmd("enable", site), _cmd("enable", f"{site}_dist_co*"),
            _cmd("reset"), _cmd("zoom", f"{site}_shown", ZOOM_BUFFER_ANG)]


def save_commands(png: str, dpi: int) -> list[tuple]:
    return [_cmd("ray", *RAY_SIZE_PX), _cmd("png", png, dpi=dpi)]


def pml_script(entries: list[dict], tilt_deg: float, dpi: int = DEFAULT_DPI) -> str:
    """PyMOL script: entries hold site, pdb name, threshold, surface ids, C/O ids, C-O lengths."""
    lines = [
        "# Fig. 1: 3D views of the relaxed ZnO-CO2 structures (written by scripts.visualization.export_pymol).",
        "# Run from this folder: pymol fig01_adsorption_structures.pml",
        "# Shown: CO2 and the exposed surface layer (slab atoms with z above the threshold of each site).",
        "",
        *(pml_line(op) for op in scene_commands(entries, tilt_deg)),
        "# One site at a time: same view for each.",
    ]
    for entry in entries:
        site = entry["site"]
        lines.append(f"# ---- render {site} (uncomment to save fig01_{site}.png)")
        lines += ["# " + pml_line(op) for op in view_commands(site) + save_commands(f"fig01_{site}.png", dpi)]
    first = entries[0]["site"]
    lines += ["", f"# Interactive start: {first}", *(pml_line(op) for op in view_commands(first)), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- renderers


def pymol_available() -> bool:
    try:
        import pymol  # noqa: F401
    except Exception:  # ImportError, or a broken installation
        return False
    return True


def _pymol_session():
    """(cmd, stop) of a headless PyMOL: pymol2.PyMOL() instance, else the -qc singleton."""
    try:
        import pymol2
    except ImportError:
        pymol2 = None
    if pymol2 is not None:
        instance = pymol2.PyMOL()
        instance.start()
        return instance.cmd, instance.stop
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd
    return cmd, (lambda: None)


def run_pymol(cmd, ops: list[tuple], output_dir: Path) -> None:
    """Execute command items on pymol.cmd; file names are taken relative to output_dir."""
    for op in ops:
        if op[0] in ("#", ""):
            continue
        name, args, kwargs = op
        if name in ("load", "png"):
            args = (str(output_dir / args[0]), *args[1:])
        getattr(cmd, name)(*args, **kwargs)


def render_pymol(entries: list[dict], output_dir: Path, tilt_deg: float, dpi: int) -> dict[str, Path]:
    cmd, stop = _pymol_session()
    panels = {}
    try:
        run_pymol(cmd, scene_commands(entries, tilt_deg), output_dir)
        for entry in entries:
            site = entry["site"]
            name = f"fig01_{site}.png"
            run_pymol(cmd, view_commands(site) + save_commands(name, dpi), output_dir)
            panels[site] = output_dir / name
    finally:
        stop()
    missing = [str(p) for p in panels.values() if not p.is_file()]
    if missing:
        raise SystemExit("[export-pymol] PyMOL wrote no " + ", ".join(missing))
    return panels


def view_rotation(tilt_deg: float) -> np.ndarray:
    """Rotation of the model about x by -(90 - tilt): as 'rotate x' in the .pml (camera looks along -z)."""
    angle = np.radians(-(90.0 - tilt_deg))
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def bonds(symbols: list[str], positions: np.ndarray) -> list[tuple[int, int]]:
    radii = np.array([covalent_radii[atomic_numbers[s]] for s in symbols])
    pairs = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            if np.linalg.norm(positions[i] - positions[j]) < BOND_TOLERANCE * (radii[i] + radii[j]):
                pairs.append((i, j))
    return pairs


def projected(entry: dict, tilt_deg: float) -> dict:
    """Shown atoms in view coordinates: x right, y up, z towards the viewer."""
    indices = entry["shown"]
    xyz = entry["positions"][indices] @ view_rotation(tilt_deg).T
    symbols = [entry["symbols"][i] for i in indices]
    radii = np.array([SPHERE_SCALE * covalent_radii[atomic_numbers[s]] for s in symbols])
    lo = (xyz[:, :2] - radii[:, None]).min(axis=0)
    hi = (xyz[:, :2] + radii[:, None]).max(axis=0)
    return {"xyz": xyz, "symbols": symbols, "radii": radii, "lo": lo, "hi": hi,
            "carbon": indices.index(entry["c_id"] - 1),
            "oxygens": [indices.index(o - 1) for o in entry["o_ids"]],
            "co_lengths": entry["co_lengths"]}


@lru_cache(maxsize=None)
def _sphere_image(rgb: tuple[float, float, float], size: int = 96) -> np.ndarray:
    """RGBA image of a shaded sphere (light from the upper left, soft highlight)."""
    u = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(u, -u)
    r2 = x ** 2 + y ** 2
    z = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    light = np.array([-0.45, 0.55, 0.70])
    light /= np.linalg.norm(light)
    diffuse = np.clip(x * light[0] + y * light[1] + z * light[2], 0.0, None)
    half = light + np.array([0.0, 0.0, 1.0])
    half /= np.linalg.norm(half)
    specular = np.clip(x * half[0] + y * half[1] + z * half[2], 0.0, None) ** 40
    base = np.array(rgb)
    colour = base * (0.42 + 0.62 * diffuse[..., None]) + 0.45 * specular[..., None]
    image = np.ones((size, size, 4))
    image[..., :3] = np.clip(colour, 0.0, 1.0)
    return image


def _draw_stick(ax, p, q, zorder):
    """Stick p -> q (2D) as a cylinder: nested strips, darker at the rim, light along the axis."""
    from matplotlib.patches import Polygon

    d = q[:2] - p[:2]
    length = np.linalg.norm(d)
    if length < 1e-6:
        return
    n = np.array([-d[1], d[0]]) / length
    if n @ np.array([-0.45, 0.55]) < 0:  # highlight on the lit side, the same for both halves
        n = -n
    # (half width as a fraction of the radius, offset of the strip centre, grey level)
    for fraction, offset, grey in ((1.0, 0.0, 0.50), (0.82, 0.08, 0.66), (0.55, 0.20, 0.80), (0.25, 0.30, 0.93)):
        centre = offset * STICK_RADIUS_ANG * n
        w = fraction * STICK_RADIUS_ANG * n
        corners = [p[:2] + centre - w, q[:2] + centre - w, q[:2] + centre + w, p[:2] + centre + w]
        ax.add_patch(Polygon(corners, closed=True, facecolor=(grey,) * 3, edgecolor="none",
                             antialiased=True, zorder=zorder))


def draw_structure(ax, view: dict, centre: np.ndarray, half: np.ndarray, label_size: float) -> None:
    from matplotlib import patheffects
    from matplotlib.patches import Circle

    xyz, radii, symbols = view["xyz"], view["radii"], view["symbols"]
    items = [(xyz[i, 2], "atom", i) for i in range(len(symbols))]
    for i, j in bonds(symbols, xyz):
        mid = 0.5 * (xyz[i] + xyz[j])
        items.append((0.5 * (xyz[i, 2] + mid[2]), "stick", (i, mid)))
        items.append((0.5 * (xyz[j, 2] + mid[2]), "stick", (j, mid)))
    for order, (_depth, kind, data) in enumerate(sorted(items, key=lambda item: item[0])):
        if kind == "stick":
            i, mid = data
            # The stick leaves the sphere at its surface (not at its centre).
            direction = (mid - xyz[i]) / np.linalg.norm(mid - xyz[i])
            # A small overlap at the midpoint hides the joint of the two halves.
            _draw_stick(ax, xyz[i] + 0.92 * radii[i] * direction, mid + 0.03 * direction, zorder=order)
            continue
        i = data
        x, y, r = xyz[i, 0], xyz[i, 1], radii[i]
        rgb = ELEMENT_COLORS.get(symbols[i], (0.6, 0.6, 0.6))
        clip = Circle((x, y), r, transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(clip)
        image = ax.imshow(_sphere_image(tuple(rgb)), extent=(x - r, x + r, y - r, y + r),
                          interpolation="bilinear", zorder=order, origin="upper")
        image.set_clip_path(clip)
        ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=tuple(0.55 * c for c in rgb),
                            linewidth=0.5, zorder=order))
    # C-O lengths beside each bond, on opposite sides of the molecular axis.
    carbon = xyz[view["carbon"], :2]
    o1, o2 = (xyz[o, :2] for o in view["oxygens"])
    axis = o2 - o1
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-6 else np.array([0.0, 1.0])
    normal = np.array([-axis[1], axis[0]])
    if normal[0] < 0:
        normal = -normal
    for sign, oxygen, length in zip((-1.0, 1.0), (o1, o2), view["co_lengths"]):
        anchor = 0.5 * (carbon + oxygen) + sign * 0.42 * normal
        ax.text(anchor[0], anchor[1], f"{length:.3f}", ha="left" if sign > 0 else "right", va="center",
                fontsize=label_size, color="black", zorder=len(items) + 1,
                path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")])
    ax.set_xlim(centre[0] - half[0], centre[0] + half[0])
    ax.set_ylim(centre[1] - half[1], centre[1] + half[1])
    ax.set_aspect("equal")
    ax.set_axis_off()


def render_ase(entries: list[dict], output_dir: Path, tilt_deg: float, dpi: int) -> dict[str, Path]:
    """Ball-and-stick panels with ASE + matplotlib; one common scale for every site."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    views = [projected(entry, tilt_deg) for entry in entries]
    margin = np.array([0.9, 0.5])  # room for the C-O labels
    half = np.max([0.5 * (v["hi"] - v["lo"]) for v in views], axis=0) + margin
    width_in = 2.4
    height_in = width_in * half[1] / half[0]
    panels = {}
    for entry, view in zip(entries, views):
        fig = Figure(figsize=(width_in, height_in))
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        draw_structure(ax, view, 0.5 * (view["lo"] + view["hi"]), half, label_size=8)
        path = output_dir / f"fig01_{entry['site']}.png"
        fig.savefig(path, dpi=dpi, facecolor="white")
        panels[entry["site"]] = path
    return panels


def compose_figure(panels: dict[str, Path], output: Path, dpi: int) -> Path:
    """Panels side by side, labelled (a), (b), ... below, with an element legend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.image import imread
    from matplotlib.lines import Line2D

    images = {site: imread(path) for site, path in panels.items()}
    aspect = max(img.shape[0] / img.shape[1] for img in images.values())
    n = len(images)
    width_in = 7.0
    panel_w = width_in / n
    label_h, legend_h = 0.30, 0.30
    height_in = panel_w * aspect + label_h + legend_h
    fig = Figure(figsize=(width_in, height_in))
    FigureCanvasAgg(fig)
    bottom = (label_h + legend_h) / height_in
    for k, (site, image) in enumerate(images.items()):
        ax = fig.add_axes((k / n, bottom, 1.0 / n, panel_w * aspect / height_in))
        ax.imshow(image, interpolation="lanczos")
        ax.set_axis_off()
        fig.text((k + 0.5) / n, (legend_h + 0.5 * label_h) / height_in,
                 f"({chr(ord('a') + k)}) {SITE_TITLES.get(site, site)}", ha="center", va="center", fontsize=10)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=rgb,
                      markeredgecolor=tuple(0.55 * c for c in rgb), label=symbol)
               for symbol, rgb in ELEMENT_COLORS.items()]
    handles.append(Line2D([], [], linestyle="none", label="C–O lengths in Å"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.5 * legend_h / height_in),
               ncol=len(handles), frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.6)
    fig.savefig(output, dpi=dpi, facecolor="white")
    return output


def choose_renderer(requested: str) -> str:
    if requested == "auto":
        return "pymol" if pymol_available() else "ase"
    if requested == "pymol" and not pymol_available():
        raise SystemExit("[export-pymol] --render pymol: 'import pymol' failed "
                         "(install pymol-open-source, or use --render auto/ase)")
    return requested


README_TEXT = """Fig. 1 (3D adsorption structures): files written by scripts.visualization.export_pymol.

Files
  <site>.pdb / <site>.cif   relaxed DFT structure of each site (lateral periodic shift only:
                            CO2 carbon at the cell centre); residue SLB = slab, CO2 = adsorbate
  {pml}   PyMOL script: loads the PDB files and sets one view
                            (sticks + spheres, element colours, white background, orthographic,
                            same tilted side view for every site, only the exposed surface
                            layer + CO2; the z threshold of each site is written in the script)
  fig01_<site>.png          rendered panel of each site
  {combined}   panels side by side, (a), (b), (c)
  {info}          renderer used (pymol = headless PyMOL running the commands of the
                            script; ase = ASE + matplotlib fallback with the same atoms and view)

The PNG files are rendered automatically (export_pymol --render auto: PyMOL when the
pymol-open-source Python package is importable, else the ASE/matplotlib fallback).
Installing pymol-open-source (conda-forge) is optional and only changes the renderer.

Re-render interactively in PyMOL
  cd <this folder>
  pymol {pml}
and uncomment the "render <site>" blocks at the end of the script
(disable/enable/reset/zoom/ray/png); each writes fig01_<site>.png.
"""


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit(f"[export-pymol] --dpi must be positive: {args.dpi}")
    if not args.summary.is_file():
        raise SystemExit(f"[export-pymol] DFT summary not found: {args.summary}")
    available = ok_sites(args.summary, args.material)
    if args.sites:
        unknown = [s for s in args.sites if s not in available]
        if unknown:
            raise SystemExit(f"[export-pymol] sites without dft_status ok in {args.summary}: {', '.join(unknown)}")
        sites = [s for s in available if s in args.sites]
    else:
        sites = available
    if not sites:
        raise SystemExit(f"[export-pymol] no {args.material} site with dft_status ok in {args.summary}")
    missing = [args.input_dir / s / f"{args.material}_{s}_optimized.xyz" for s in sites]
    missing = [str(p) for p in missing if not p.is_file()]
    if missing:
        raise SystemExit("[export-pymol] missing relaxed structure(s):\n  " + "\n  ".join(missing))
    renderer = choose_renderer(args.render)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for site in sites:
        atoms = read(args.input_dir / site / f"{args.material}_{site}_optimized.xyz")
        check_co2(atoms, site)
        atoms = centred(atoms)
        pdb, cif = write_structure(atoms, output_dir, site)
        slab = len(atoms) - 3
        positions = pdb_positions(atoms)
        z = positions[:, 2]
        threshold = surface_threshold(z[:slab], args.layer_gap)
        carbon = slab + 1
        surface = [i for i in range(slab) if z[i] > threshold]
        entries.append({
            "site": site, "pdb": pdb.name, "threshold": threshold,
            "surface_ids": [i + 1 for i in surface],
            "c_id": carbon + 1, "o_ids": [slab + 1, slab + 3],
            "co_lengths": [float(atoms.get_distance(carbon, o)) for o in (slab, slab + 2)],
            "positions": positions, "symbols": atoms.get_chemical_symbols(),
            "shown": [*surface, slab, slab + 1, slab + 2],
        })
        print(f"[export-pymol] {site}: wrote {pdb} and {cif} (surface z > {threshold:.3f} A, "
              f"{len(surface)} slab atoms shown)")
    pml = output_dir / PML_NAME
    pml.write_text(pml_script(entries, args.tilt_deg, args.dpi), encoding="utf-8")
    print(f"[export-pymol] wrote {pml}")

    panels: dict[str, Path] = {}
    combined = None
    if renderer == "pymol":
        panels = render_pymol(entries, output_dir, args.tilt_deg, args.dpi)
    elif renderer == "ase":
        panels = render_ase(entries, output_dir, args.tilt_deg, args.dpi)
    if panels:
        combined = compose_figure(panels, output_dir / COMBINED_NAME, args.dpi)
        print(f"[export-pymol] renderer: {renderer} (requested {args.render}); wrote "
              f"{', '.join(p.name for p in panels.values())} and {combined}")
    else:
        print("[export-pymol] renderer: none (no PNG written)")
    info = {
        "requested": args.render,
        "renderer": renderer,
        "dpi": args.dpi,
        "tilt_deg": args.tilt_deg,
        "panels": {site: path.name for site, path in panels.items()},
        "combined": combined.name if combined else None,
    }
    (output_dir / RENDER_INFO_NAME).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    readme = output_dir / README_NAME
    readme.write_text(README_TEXT.format(pml=PML_NAME, combined=COMBINED_NAME, info=RENDER_INFO_NAME),
                      encoding="utf-8")
    print(f"[export-pymol] wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
