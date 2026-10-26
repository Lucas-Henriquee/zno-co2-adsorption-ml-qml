from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    metal_symbol: str
    structure: str
    miller_index: tuple[int, int, int]
    layers: int = 2
    repeat: tuple[int, int, int] = (1, 1, 1)
    vacuum: float = 10.0


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    if value:
        return Path(value).expanduser().resolve()
    return default


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = _configured_path("QML_ZNO_DATA_DIR", ROOT_DIR / "data")
GEOMETRY_DIR = _configured_path("QML_ZNO_GEOMETRY_DIR", DATA_DIR / "geometries")
RESULTS_DIR = _configured_path("QML_ZNO_RESULTS_DIR", ROOT_DIR / "results")
DFT_RESULTS_DIR = RESULTS_DIR / "dft"
TDDFT_RESULTS_DIR = RESULTS_DIR / "tddft"

MATERIALS: dict[str, MaterialSpec] = {
    "ZnO": MaterialSpec(
        name="ZnO",
        metal_symbol="Zn",
        structure="wurtzite",
        miller_index=(1, 0, 0),
        # 2x1 surface cell (6.50 x 5.21 A): CO2 stays >= 4.18 A from its periodic
        # copy for the three sites; a CO2 lying along y would need 2x2.
        repeat=(2, 1, 1),
    ),
    "TiO2": MaterialSpec(
        name="TiO2",
        metal_symbol="Ti",
        structure="rutile",
        miller_index=(1, 1, 0),
    ),
    "CeO2": MaterialSpec(
        name="CeO2",
        metal_symbol="Ce",
        structure="fluorite",
        miller_index=(1, 1, 1),
    ),
}

ADSORPTION_SITES: tuple[str, ...] = ("top_metal", "top_oxygen", "bridge")

# DFT-only exploratory starts; not run by default and not used by TDDFT, CO2RR, MD or NEB.
# "carbonate": bent CO2 with C above a surface O, a carbonate-like start that the three
# linear starts above (2.2-3.7 A away) cannot reach by local relaxation.
# "tridentate": the same bent CO2 turned along [0001] (cell y axis), where the surface O has
# its two Zn neighbours, as in the tridentate carbonate reported for ZnO(10-10)
# (Wang et al., Angew. Chem. Int. Ed. 46, 5624, 2007). The carbonate start lies along x,
# where those Zn are missing.
EXPLORATORY_SITES: tuple[str, ...] = ("carbonate", "tridentate")
DFT_SITES: tuple[str, ...] = ADSORPTION_SITES + EXPLORATORY_SITES

CO2_SITE_HEIGHTS = {
    "top_metal": 2.20,  # Reference placement retained from the original workflow.
    "top_oxygen": 3.17,  # Updated from the earlier 2.40 Angstrom placement.
    "bridge": 3.67,  # Updated from the earlier 2.80 Angstrom placement.
    "carbonate": 1.70,  # C above the surface O; between a C-O bond (~1.35 A) and contact.
    "tridentate": 1.55,  # Closer to a C-O bond; stays above the 1.5 A start-geometry check.
}

# Bent CO2 for the carbonate start (O-C-O angle and C-O length of an activated CO2).
CARBONATE_OCO_ANGLE_DEG = 130.0
CARBONATE_CO_BOND_ANGSTROM = 1.25

# Engineering-friendly screening targets for photocatalytic CO2 splitting.
TARGET_ADSORPTION_ENERGY_EV = -0.70
TARGET_BAND_GAP_EV = 2.40
TARGET_ONSET_EV = 2.20


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        GEOMETRY_DIR,
        RESULTS_DIR,
        DFT_RESULTS_DIR,
        TDDFT_RESULTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
