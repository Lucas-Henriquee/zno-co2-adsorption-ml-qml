from __future__ import annotations
from scripts.common.timer import timer
import argparse
import json
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required. Install with: pip install numpy") from exc

from scripts.common.project_config import (
    CARBONATE_CO_BOND_ANGSTROM,
    CARBONATE_OCO_ANGLE_DEG,
    CO2_SITE_HEIGHTS,
    DFT_SITES,
    GEOMETRY_DIR,
    MATERIALS,
    ensure_directories,
)


def _require_ase() -> None:
    try:
        import ase  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ASE is required for geometry generation. Install with: pip install ase"
        ) from exc


def build_bulk(material: str):
    from ase.spacegroup import crystal

    if material == "ZnO":
        return crystal(
            symbols=["Zn", "O"],
            # Wurtzite: Zn and O on Wyckoff 2b (1/3, 2/3, z); (0, 0, z) stacks all atoms on one line.
            basis=[(1 / 3, 2 / 3, 0.0), (1 / 3, 2 / 3, 0.3825)],
            spacegroup=186,
            cellpar=[3.249, 3.249, 5.206, 90.0, 90.0, 120.0],
        )

    if material == "TiO2":
        return crystal(
            symbols=["Ti", "O"],
            basis=[(0.0, 0.0, 0.0), (0.305, 0.305, 0.0)],
            spacegroup=136,
            cellpar=[4.594, 4.594, 2.958, 90.0, 90.0, 90.0],
        )

    if material == "CeO2":
        return crystal(
            symbols=["Ce", "O"],
            basis=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
            spacegroup=225,
            cellpar=[5.411, 5.411, 5.411, 90.0, 90.0, 90.0],
        )

    raise ValueError(f"Unsupported material: {material}")


def build_slab(material: str):
    from ase.build import surface

    spec = MATERIALS[material]
    bulk = build_bulk(material)
    slab = surface(bulk, spec.miller_index, layers=spec.layers, vacuum=spec.vacuum)
    slab = slab.repeat(spec.repeat)
    slab.center(vacuum=spec.vacuum, axis=2)
    slab.wrap()
    return slab


def _top_atom_indices(slab, z_window: float = 0.3) -> list[int]:
    # Only the outermost plane: on ZnO(10-10) the next plane lies 0.94 A below the top.
    z_positions = slab.positions[:, 2]
    z_max = float(np.max(z_positions))
    return [idx for idx, z_val in enumerate(z_positions) if z_val > z_max - z_window]


def _nearest_metal_oxygen_pair(slab, metal_indices: list[int], oxygen_indices: list[int]) -> tuple[int, int]:
    best_pair = (metal_indices[0], oxygen_indices[0])
    best_distance = float("inf")
    for m_idx in metal_indices:
        for o_idx in oxygen_indices:
            distance = slab.get_distance(m_idx, o_idx, mic=True)
            if distance < best_distance:
                best_distance = distance
                best_pair = (m_idx, o_idx)
    return best_pair


def select_adsorption_sites(slab, metal_symbol: str) -> dict[str, np.ndarray]:
    top_indices = _top_atom_indices(slab)
    top_metals = [i for i in top_indices if slab[i].symbol == metal_symbol]
    top_oxygens = [i for i in top_indices if slab[i].symbol == "O"]

    if not top_metals:
        top_metals = [i for i, atom in enumerate(slab) if atom.symbol == metal_symbol]
        top_metals = [max(top_metals, key=lambda i: slab.positions[i, 2])]

    if not top_oxygens:
        top_oxygens = [i for i, atom in enumerate(slab) if atom.symbol == "O"]
        top_oxygens = [max(top_oxygens, key=lambda i: slab.positions[i, 2])]

    metal_idx = max(top_metals, key=lambda i: slab.positions[i, 2])
    oxygen_idx = max(top_oxygens, key=lambda i: slab.positions[i, 2])
    bridge_pair = _nearest_metal_oxygen_pair(slab, top_metals, top_oxygens)
    z_surface = float(np.max(slab.positions[:, 2]))

    metal_site = slab.positions[metal_idx].copy()
    oxygen_site = slab.positions[oxygen_idx].copy()
    bridge_vector = slab.get_distance(bridge_pair[0], bridge_pair[1], mic=True, vector=True)
    bridge_site = slab.positions[bridge_pair[0]] + 0.5 * bridge_vector

    metal_site[2] = z_surface
    oxygen_site[2] = z_surface
    bridge_site[2] = z_surface

    return {
        "top_metal": metal_site,
        "top_oxygen": oxygen_site,
        "bridge": bridge_site,
        "carbonate": oxygen_site.copy(),
        "tridentate": oxygen_site.copy(),
    }


def create_co2(site_name: str):
    from ase import Atoms

    if site_name == "carbonate":
        # Bent O-C-O in the xz plane (along the 6.50 A cell axis), C pointing down to the
        # surface O and both O atoms above it, as in a monodentate carbonate.
        half_angle = np.radians(CARBONATE_OCO_ANGLE_DEG / 2.0)
        dx = CARBONATE_CO_BOND_ANGSTROM * np.sin(half_angle)
        dz = CARBONATE_CO_BOND_ANGSTROM * np.cos(half_angle)
        return Atoms("OCO", positions=[(-dx, 0.0, dz), (0.0, 0.0, 0.0), (dx, 0.0, dz)])

    if site_name == "tridentate":
        # Same bent O-C-O in the yz plane, along [0001], so each O points to one of the two Zn
        # that neighbour the surface O along the Zn-O dimer rows.
        half_angle = np.radians(CARBONATE_OCO_ANGLE_DEG / 2.0)
        dy = CARBONATE_CO_BOND_ANGSTROM * np.sin(half_angle)
        dz = CARBONATE_CO_BOND_ANGSTROM * np.cos(half_angle)
        return Atoms("OCO", positions=[(0.0, -dy, dz), (0.0, 0.0, 0.0), (0.0, dy, dz)])

    co2 = Atoms("OCO", positions=[(-1.16, 0.0, 0.0), (0.0, 0.0, 0.0), (1.16, 0.0, 0.0)])
    if site_name == "top_oxygen":
        co2.rotate(35.0, "y", center=(0.0, 0.0, 0.0))
    elif site_name == "bridge":
        co2.rotate(90.0, "y", center=(0.0, 0.0, 0.0))
    return co2


def add_co2_to_site(slab, site_name: str, site_position: np.ndarray):
    combined = slab.copy()
    co2 = create_co2(site_name)
    target_position = site_position.copy()
    target_position[2] += CO2_SITE_HEIGHTS[site_name]
    carbon_index = 1
    translation = target_position - co2.positions[carbon_index]
    co2.translate(translation)
    combined.extend(co2)
    return combined


def _as_float_list(vector: np.ndarray) -> list[float]:
    return [float(value) for value in vector.tolist()]


def save_material_geometries(material: str) -> None:
    from ase.io import write

    spec = MATERIALS[material]
    slab = build_slab(material)
    sites = select_adsorption_sites(slab, spec.metal_symbol)

    material_dir = GEOMETRY_DIR / material
    material_dir.mkdir(parents=True, exist_ok=True)

    clean_traj_path = material_dir / "clean_slab.traj"
    clean_cif_path = material_dir / "clean_slab.cif"
    write(clean_traj_path, slab)
    write(clean_cif_path, slab)

    metadata = {
        "material": material,
        "miller_index": list(spec.miller_index),
        "site_positions_angstrom": {},
        "generated_structures": {
            "clean_slab": str(clean_traj_path),
            "clean_slab_cif": str(clean_cif_path),
        },
    }

    for site_name in DFT_SITES:
        case_atoms = add_co2_to_site(slab, site_name, sites[site_name])
        case_traj_path = material_dir / f"co2_{site_name}.traj"
        case_cif_path = material_dir / f"co2_{site_name}.cif"
        write(case_traj_path, case_atoms)
        write(case_cif_path, case_atoms)
        metadata["site_positions_angstrom"][site_name] = _as_float_list(sites[site_name])
        metadata["generated_structures"][site_name] = str(case_traj_path)
        metadata["generated_structures"][f"{site_name}_cif"] = str(case_cif_path)

    with (material_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build oxide slabs with three CO2 adsorption sites.")
    parser.add_argument(
        "--materials",
        nargs="*",
        default=sorted(MATERIALS.keys()),
        help="Subset of materials to generate (default: all).",
    )
    return parser.parse_args()


def validate_materials(materials: Iterable[str]) -> list[str]:
    unknown = [material for material in materials if material not in MATERIALS]
    if unknown:
        raise SystemExit(f"Unknown materials: {', '.join(unknown)}")
    return list(materials)


def main() -> None:
    _require_ase()
    ensure_directories()
    args = parse_args()
    selected_materials = validate_materials(args.materials)

    for material in selected_materials:
        with timer(f"Geometry total → {material}"):
            save_material_geometries(material)
        print(f"[geometry] generated structures for {material}")


if __name__ == "__main__":
    main()
