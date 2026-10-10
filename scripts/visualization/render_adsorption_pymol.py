"""Ray-trace saved DFT geometries with PyMOL and assemble the three panels."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.visualization.render_adsorption_structures import COLORS, SITES, SITE_NAMES, top_layer_indices


def surface_model(data):
    """Display the saved exposed layer and CO2 without periodic replication."""
    from chempy import Atom, Bond
    from chempy.models import Indexed

    atoms = data.atoms
    origin = atoms.positions[-2].copy()
    origin[2] = atoms.positions[:-3, 2].max()
    model = Indexed()
    anchors = []
    for i in top_layer_indices(atoms):
        atom = Atom()
        atom.symbol = atoms[i].symbol
        atom.name = f"{atom.symbol}{len(model.atom) + 1}"
        atom.id = len(model.atom) + 1
        atom.coord = (atoms.positions[i] - origin).tolist()
        atom.resn = "ZNO"
        model.add_atom(atom)
        if i in data.site_atoms:
            anchors.append(len(model.atom))
    slab_count = len(model.atom)
    for i in range(len(atoms) - 3, len(atoms)):
        atom = Atom()
        atom.symbol = atoms[i].symbol
        atom.name = f"{atom.symbol}{len(model.atom) + 1}"
        atom.id = len(model.atom) + 1
        atom.coord = (atoms.positions[i] - origin).tolist()
        atom.resn = "CO2"
        model.add_atom(atom)
    for i in range(slab_count):
        for j in range(i + 1, slab_count):
            if {model.atom[i].symbol, model.atom[j].symbol} != {"Zn", "O"}:
                continue
            distance = np.linalg.norm(np.array(model.atom[i].coord) - model.atom[j].coord)
            if 1.4 <= distance <= 2.5:
                bond = Bond()
                bond.index = [i, j]
                bond.order = 1
                model.add_bond(bond)
    for oxygen in (slab_count, slab_count + 2):
        bond = Bond()
        bond.index = [slab_count + 1, oxygen]
        bond.order = 1
        model.add_bond(bond)
    return model, anchors, slab_count


def save_pymol(data_by_site, args, suffix):
    import pymol2

    models = {site: surface_model(data) for site, data in data_by_site.items()}
    points = np.array([atom.coord for model, _, _ in models.values() for atom in model.atom])
    low, high = points.min(axis=0) - 0.8, points.max(axis=0) + 0.8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_width, panel_height = 1500, 1500
    panels = []
    for site in SITES:
        model, anchors, slab_count = models[site]
        with pymol2.PyMOL() as session:
            cmd = session.cmd
            cmd.bg_color("white")
            cmd.set("orthoscopic", 1)
            cmd.set("antialias", 2)
            cmd.set("ray_shadows", 0)
            cmd.set("specular", 0.25)
            cmd.set("ambient", 0.45)
            cmd.set("depth_cue", 0)
            cmd.set("ray_opaque_background", 1)
            cmd.set("retain_order", 1)
            cmd.load_model(model, "structure")
            cmd.hide("everything", "all")
            for symbol, color in COLORS.items():
                cmd.set_color(f"atom_{symbol}", [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)])
                cmd.color(f"atom_{symbol}", f"elem {symbol}")
            cmd.show("spheres", "structure")
            cmd.set("sphere_scale", 0.22)
            cmd.show("sticks", "structure")
            cmd.set("stick_radius", 0.085)
            cmd.set("stick_color", "gray70")
            # Dark transparent shells identify the original starting-site atoms.
            for number, index in enumerate(anchors):
                cmd.create(f"site_{number}", f"structure and name {model.atom[index - 1].name}")
                cmd.color("gray20", f"site_{number}")
                cmd.hide("everything", f"site_{number}")
                cmd.show("spheres", f"site_{number}")
                cmd.set("sphere_scale", 0.30, f"site_{number}")
                cmd.set("sphere_transparency", 0.65, f"site_{number}")
            carbon = slab_count + 2
            for number, oxygen in enumerate((slab_count + 1, slab_count + 3)):
                name = f"CO_length_{number}"
                c_selection = f"structure and name {model.atom[carbon - 1].name}"
                o_selection = f"structure and name {model.atom[oxygen - 1].name}"
                expected = np.linalg.norm(
                    np.array(model.atom[carbon - 1].coord) - model.atom[oxygen - 1].coord
                )
                measured = cmd.get_distance(c_selection, o_selection)
                if not np.isclose(measured, expected, atol=1e-5, rtol=0):
                    raise ValueError(f"{site}: PyMOL distance does not match the saved C–O bond.")
                print(f"[PyMOL] {site} C–O {number + 1}: {measured:.6f} Å")
                cmd.distance(name, c_selection, o_selection)
                cmd.hide("dashes", name)
                cmd.set("label_position", [(-1 if number == 0 else 1) * 2.0, 0.2, 0], name)
            cmd.set("label_color", "black")
            cmd.set("label_size", 17)
            cmd.set("label_distance_digits", 3)
            cmd.set("label_outline_color", "white")
            for number, point in enumerate((low, high)):
                cmd.pseudoatom("bounds", pos=point.tolist(), name=f"B{number}")
            cmd.hide("everything", "bounds")
            # Change the view, not the coordinates, to expose the tilted molecule.
            cmd.rotate("z", 180 if site == "top_metal" else 0)
            cmd.rotate("x", -60)
            cmd.rotate("y", -22)
            cmd.zoom("bounds", buffer=0, complete=1)
            cmd.save(str(args.output_dir / f"zno_{site}_pymol{suffix}.pse"))
            target = args.output_dir / f"zno_{site}_pymol{suffix}.png"
            cmd.png(str(target), width=panel_width, height=panel_height, dpi=args.dpi, ray=1, quiet=1)
        panels.append(Image.open(target).convert("RGB"))

    # Apply one common crop so all three panels retain the same spatial scale.
    boxes = []
    for panel in panels:
        pixels = np.asarray(panel)
        rows, columns = np.where(np.any(pixels < 240, axis=2))
        boxes.append((columns.min(), rows.min(), columns.max() + 1, rows.max() + 1))
    crop = (max(0, min(box[0] for box in boxes) - 80),
            max(0, min(box[1] for box in boxes) - 80),
            min(panel_width, max(box[2] for box in boxes) + 80),
            min(panel_height, max(box[3] for box in boxes) + 80))
    panels = [panel.crop(crop) for panel in panels]
    panel_width, panel_height = panels[0].size

    # PyMOL renders the scenes; Pillow adds panel names and the shared legend.
    canvas = Image.new("RGB", (panel_width * 3, panel_height + 260), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 45) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 34) if font_path.exists() else ImageFont.load_default()
    for column, (site, panel) in enumerate(zip(SITES, panels)):
        canvas.paste(panel, (column * panel_width, 0))
        draw.text((column * panel_width + panel_width / 2, panel_height + 25),
                  f"({chr(97 + column)}) {SITE_NAMES[site]}", fill="black", font=font, anchor="mt")
    labels = [("Zn", COLORS["Zn"]), ("O", COLORS["O"]), ("C", COLORS["C"])]
    start = panel_width * 3 / 2 - 750
    for offset, (label, color) in enumerate(labels):
        x = start + offset * 240
        draw.ellipse((x, panel_height + 140, x + 32, panel_height + 172), fill=color)
        draw.text((x + 48, panel_height + 136), label, fill="black", font=small)
    draw.text((start + 720, panel_height + 136), "Dark halos: starting site     C–O lengths in Å",
              fill="black", font=small)
    target = args.output_dir / f"zno_adsorption_sites_pymol{suffix}.png"
    canvas.save(target, dpi=(args.dpi, args.dpi))
    print(f"[saved] {target}")
