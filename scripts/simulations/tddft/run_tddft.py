"""Linear-response TDDFT spectrum of the adsorbed CO2 molecule, isolated in a box.

What this script computes
-------------------------
For every requested (material, site) case it takes the DFT-optimised adsorbed
structure of the same run, extracts the CO2 molecule (by construction the last
three atoms, O-C-O), places it alone in a cubic non-periodic box and computes
its LrTDDFT (Casida) excitation spectrum with GPAW in fd mode at Gamma.

The result is therefore the optical spectrum of the CO2 molecule in its
adsorbed (distorted) geometry, *without* the ZnO slab. It is NOT the optical
response of the ZnO-CO2 system and must not be compared with a photocatalyst
absorption-onset target (e.g. 2.2 eV): the first excitations of CO2 lie at
~8-10 eV. The slab .gpw (PW mode, several k-points) cannot be reused because
GPAW LrTDDFT requires fd or lcao mode and a single (Gamma) k-point.

Casida basis: Kohn-Sham transitions i -> j with j up to the last converged band
(restrict jend = converged bands - 1; the unconverged buffer bands are excluded)
and with KS transition energy within --energy-window.

Descriptors (all over every computed transition; --max-transitions only limits the
rows written to transitions.csv): first excitation (energy and f), first bright
excitation (lowest transition with f >= --osc-threshold; not an absorption onset),
peak (max f) and the TRK sum, which is the total oscillator strength.

KS ionization threshold: -epsilon_HOMO of the isolated CO2 (non-periodic box, vacuum
level ~0). With PBE, excitations above it lie in the Kohn-Sham continuum, which the
finite box discretises into artificial bound states (Casida et al., J. Chem. Phys.
108, 4439 (1998); Tozer and Handy, J. Chem. Phys. 109, 10180 (1998)); such
transitions are flagged (co2_isolated_first_bright_above_ks_threshold).

Units: GPAW returns excitation energies in Hartree; everything written by this
script is in eV. Oscillator strengths are the dimensionless isotropic totals
f_j = ex.get_oscillator_strength()[0].
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable
from scripts.common.timer import timer, print0

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required. Install with: pip install numpy") from exc

from scripts.common.project_config import ADSORPTION_SITES, DFT_RESULTS_DIR, MATERIALS, RESULTS_DIR, TDDFT_RESULTS_DIR, ensure_directories


TDDFT_MODE = "co2_isolated_adsorbed_geometry_fd_gamma"
GROUND_STATE_XC = "PBE"
BAND_BUFFER = 8
STATUS_OK = "ok"

RESULT_COLUMNS = (
    "co2_isolated_first_excitation_ev",
    "co2_isolated_first_excitation_f",
    "co2_isolated_first_bright_excitation_ev",
    "co2_isolated_peak_ev",
    "co2_isolated_peak_f",
    "co2_isolated_ks_ionization_threshold_ev",
    "co2_isolated_first_bright_above_ks_threshold",
    "co2_isolated_num_transitions",
    "co2_isolated_num_transitions_computed",
    "trk_sum",
    "trk_valence_electrons",
    "trk_fraction",
)

# Columns renamed since older tddft_summary.csv files; applied by the readers. The former
# co2_isolated_total_f (sum over the first max_transitions only) has no successor: trk_sum
# is the total.
LEGACY_COLUMN_ALIASES = {
    "co2_isolated_onset_ev": "co2_isolated_first_bright_excitation_ev",
}


def _require_gpaw() -> None:
    try:
        import gpaw  # noqa: F401
    except ImportError as exc:
        raise SystemExit("GPAW is required for TDDFT runs.") from exc


def is_master() -> bool:
    """True on MPI rank 0, or when GPAW (and hence MPI) is not available."""
    try:
        from gpaw.mpi import world
    except ImportError:
        return True
    return world.rank == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LrTDDFT excitation spectrum of the adsorbed CO2 molecule (DFT-optimised geometry), "
            "isolated in a box without the slab. Not the optical response of the ZnO-CO2 system."
        )
    )
    parser.add_argument("--materials", nargs="*", default=sorted(MATERIALS.keys()))
    parser.add_argument("--sites", nargs="*", default=list(ADSORPTION_SITES))
    parser.add_argument(
        "--osc-threshold",
        type=float,
        default=1.0e-3,
        help=(
            "Oscillator strength f threshold: the first bright excitation is the lowest transition "
            "with f >= threshold."
        ),
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=50,
        help=(
            "Number of lowest-energy transitions written to transitions.csv. The descriptors and "
            "the TRK sum always use every computed transition. <= 0 writes all."
        ),
    )
    parser.add_argument("--grid-spacing", type=float, default=0.25, help="Real-space grid spacing h in Angstrom.")
    parser.add_argument(
        "--box-size",
        type=float,
        default=18.0,
        help="Edge of the cubic non-periodic box holding the isolated CO2, in Angstrom.",
    )
    parser.add_argument(
        "--smearing-width",
        type=float,
        default=0.10,
        help="Fermi-Dirac smearing width of the ground state, in eV.",
    )
    parser.add_argument(
        "--nbands",
        type=int,
        default=68,
        help="Bands on the fixed density (CO2 has 8 occupied); raised to converged + 8 buffer bands.",
    )
    parser.add_argument(
        "--converged-bands",
        type=int,
        default=None,
        help="Bands converged on the fixed density (convergence={'bands': N}). Default: nbands.",
    )
    parser.add_argument(
        "--energy-window",
        type=float,
        default=30.0,
        help=(
            "Maximal Kohn-Sham transition energy in eV (restrict energy_range). The LrTDDFT basis is "
            "limited by this window and by jend = last converged band (buffer bands excluded)."
        ),
    )
    return parser.parse_args()


def ground_state_parameters(args: argparse.Namespace) -> dict[str, float | int | str]:
    nbands = int(args.nbands)
    converged = nbands if args.converged_bands is None else int(args.converged_bands)
    if nbands <= 8:
        raise SystemExit("--nbands must exceed the 8 occupied CO2 bands.")
    if not 8 < converged <= nbands:
        raise SystemExit("--converged-bands must be > 8 and <= --nbands.")
    # The top bands of the Davidson block converge poorly, so unconverged buffer bands are
    # always added above the converged ones (GPAW LrTDDFT tutorial: nbands=10, bands=8).
    nbands = max(nbands, converged + BAND_BUFFER)
    for name in ("grid_spacing", "box_size", "smearing_width", "energy_window"):
        if not float(getattr(args, name)) > 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    return {
        "gs_xc": GROUND_STATE_XC,
        "gs_grid_spacing_ang": float(args.grid_spacing),
        "gs_box_size_ang": float(args.box_size),
        "gs_smearing_width_ev": float(args.smearing_width),
        "gs_nbands": nbands,
        "gs_converged_bands": converged,
        "lr_energy_window_ev": float(args.energy_window),
        # 0-based index of the last band in the Casida basis (KSSRestrictor jend, inclusive).
        "lr_jend_band_index": converged - 1,
        "osc_threshold": float(args.osc_threshold),
        "max_transitions": int(args.max_transitions),
    }


def validate_inputs(materials: Iterable[str], sites: Iterable[str]) -> tuple[list[str], list[str]]:
    unknown_materials = [name for name in materials if name not in MATERIALS]
    if unknown_materials:
        raise SystemExit(f"Unknown materials: {', '.join(unknown_materials)}")

    unknown_sites = [name for name in sites if name not in ADSORPTION_SITES]
    if unknown_sites:
        raise SystemExit(f"Unknown sites: {', '.join(unknown_sites)}")

    return list(materials), list(sites)


def dft_case_paths(material: str, site: str) -> tuple[Path, Path]:
    gpw = DFT_RESULTS_DIR / material / site / f"{material}_{site}.gpw"
    xyz = DFT_RESULTS_DIR / material / site / f"{material}_{site}_optimized.xyz"
    return gpw, xyz


def load_dft_status(path: Path) -> dict[tuple[str, str], str] | None:
    """Map (material, site) to dft_status from dft_summary.csv; None if the file is missing."""
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (row.get("material", ""), row.get("site", "")): row.get("dft_status", "")
            for row in csv.DictReader(handle)
        }


def extract_transitions(lr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (energies in eV, total f, f x/y/z components) from a diagonalised LrTDDFT."""
    from ase.units import Hartree

    # ExcitationList.get_energies() is in Hartree.
    energies_ev = np.asarray(lr.get_energies(), dtype=float).reshape(-1) * Hartree
    strengths = []
    for ex in lr:
        # [f, fx, fy, fz]: index 0 is the isotropic total, the rest are components.
        osc = np.asarray(ex.get_oscillator_strength(), dtype=float).reshape(-1)
        if osc.shape != (4,):
            raise RuntimeError(f"Unexpected oscillator strength shape {osc.shape}; expected [f, fx, fy, fz].")
        strengths.append(osc)
    if len(energies_ev) == 0:
        raise RuntimeError("LrTDDFT produced no transitions.")
    if len(strengths) != len(energies_ev):
        raise RuntimeError(
            f"LrTDDFT returned {len(energies_ev)} energies but {len(strengths)} oscillator strengths."
        )
    osc_array = np.vstack(strengths)
    if not (np.all(np.isfinite(energies_ev)) and np.all(np.isfinite(osc_array))):
        raise RuntimeError("LrTDDFT returned non-finite energies or oscillator strengths.")
    return energies_ev, osc_array[:, 0], osc_array[:, 1:4]


def write_transitions_csv(path: Path, energies_ev: np.ndarray, f_total: np.ndarray, f_xyz: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "transition_index",
                "energy_ev",
                "oscillator_strength",
                "oscillator_strength_x",
                "oscillator_strength_y",
                "oscillator_strength_z",
            ]
        )
        for idx, (energy, f, fxyz) in enumerate(zip(energies_ev, f_total, f_xyz), start=1):
            writer.writerow([idx, float(energy), float(f), *(float(value) for value in fxyz)])


def _legacy_gpaw(*args, **kwargs):
    """GPAW calculator, requesting the legacy implementation when the installed GPAW knows it.

    gpaw.lrtddft.LrTDDFT works only with the legacy (gpaw.calculator) calculator. Newer GPAW
    releases accept ``legacy_gpaw=True`` to keep returning it once the new code becomes the
    default; GPAW 25.7 (the lab version) does not know the keyword and raises TypeError, in
    which case the call is repeated without it (25.7 always returns the legacy calculator).
    """
    from gpaw import GPAW

    try:
        return GPAW(*args, legacy_gpaw=True, **kwargs)
    except TypeError:
        # Any other TypeError is raised again by the plain call below.
        return GPAW(*args, **kwargs)


def _build_isolated_co2_ground_state(xyz_file: Path, out_dir: Path, params: dict):
    from ase.io import read
    from gpaw import FermiDirac

    full_atoms = read(xyz_file)
    if len(full_atoms) < 3:
        raise RuntimeError(f"Optimized structure has fewer than 3 atoms: {xyz_file}")

    # By construction from build_geometries.py the final 3 atoms are O, C, O.
    atoms = full_atoms[-3:].copy()
    if atoms.get_chemical_symbols() != ["O", "C", "O"]:
        raise RuntimeError(f"Last three atoms are {atoms.get_chemical_symbols()}, expected O, C, O: {xyz_file}")
    # Isolated molecule: cubic non-periodic box decoupled from the slab cell.
    box = float(params["gs_box_size_ang"])
    atoms.set_cell([box, box, box], scale_atoms=False)
    atoms.pbc = False
    atoms.center()
    extent = np.ptp(atoms.positions, axis=0).max()
    if (box - extent) / 2.0 < 4.0:
        raise RuntimeError(f"Box of {box} A leaves less than 4 A vacuum around the CO2.")

    fd_txt = out_dir / "co2_isolated_ground.txt"
    fd_gpw = out_dir / "co2_isolated_ground.gpw"

    # Self-consistent density first, converging only the occupied states (GPAW default).
    calc = _legacy_gpaw(
        mode="fd",
        h=float(params["gs_grid_spacing_ang"]),
        xc=GROUND_STATE_XC,
        occupations=FermiDirac(float(params["gs_smearing_width_ev"])),
        symmetry="off",
        txt=str(fd_txt),
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    # Unoccupied states enter the Casida basis, so they are converged non-self-consistently
    # on the fixed density. RMM-DIIS is unreliable for unoccupied states (GPAW docs): Davidson.
    calc = calc.fixed_density(
        nbands=int(params["gs_nbands"]),
        convergence={"bands": int(params["gs_converged_bands"])},
        eigensolver="dav",
        txt=str(out_dir / "co2_isolated_bands.txt"),
    )
    calc.write(str(fd_gpw), mode="all")
    # Re-read as the legacy calculator, which is what LrTDDFT needs.
    return _legacy_gpaw(str(fd_gpw), txt=None)


def _valence_electrons(calc) -> float:
    try:
        return float(calc.get_number_of_electrons())
    except Exception:
        return float("nan")


def ks_ionization_threshold(calc) -> float:
    """-epsilon_HOMO in eV of the isolated CO2 (non-periodic box, vacuum level ~0); NaN if unavailable."""
    try:
        homo = float(np.asarray(calc.get_homo_lumo(), dtype=float).reshape(-1)[0])
    except Exception:
        return float("nan")
    return -homo if math.isfinite(homo) else float("nan")


def run_lrtddft(
    xyz_file: Path, out_dir: Path, params: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Ground state + LrTDDFT for the isolated CO2.

    Returns (E_eV, f, f_xyz, valence electrons, KS ionization threshold -epsilon_HOMO in eV).
    """
    from gpaw.lrtddft import LrTDDFT

    if not xyz_file.exists():
        raise RuntimeError(f"TDDFT requires the optimized XYZ, but the file is missing: {xyz_file}")
    calc = _build_isolated_co2_ground_state(xyz_file=xyz_file, out_dir=out_dir, params=params)
    # txt is always this run's file (opened for writing); its content is never parsed.
    # KSSRestrictor defaults to jend = sys.maxsize: without jend the unconverged buffer bands
    # would enter the Casida basis.
    lr = LrTDDFT(
        calc,
        restrict={
            "energy_range": float(params["lr_energy_window_ev"]),
            "jend": int(params["lr_jend_band_index"]),
        },
        txt=str(out_dir / "lrtddft.txt"),
    )
    lr.diagonalize()
    energies_ev, f_total, f_xyz = extract_transitions(lr)
    return energies_ev, f_total, f_xyz, _valence_electrons(calc), ks_ionization_threshold(calc)


def empty_summary() -> dict[str, float]:
    return {key: float("nan") for key in RESULT_COLUMNS}


def summarize_transitions(
    energies_ev: np.ndarray,
    f_total: np.ndarray,
    osc_threshold: float,
    max_transitions: int,
    valence_electrons: float,
    ks_ionization_threshold_ev: float = float("nan"),
) -> dict[str, float]:
    """Spectral descriptors of the isolated CO2 over every computed transition (eV, f).

    max_transitions only sets co2_isolated_num_transitions, the rows written to transitions.csv.
    """
    energies_ev = np.asarray(energies_ev, dtype=float)
    f_total = np.asarray(f_total, dtype=float)
    if len(energies_ev) == 0 or len(energies_ev) != len(f_total):
        raise ValueError("Energies and oscillator strengths must be non-empty and of equal length.")

    order = np.argsort(energies_ev)
    energies_ev = energies_ev[order]
    f_total = f_total[order]

    # TRK sum rule: sum_j f_j -> number of valence electrons for a complete basis.
    trk_sum = float(np.sum(f_total))
    trk_fraction = trk_sum / valence_electrons if valence_electrons and math.isfinite(valence_electrons) else float("nan")

    bright = energies_ev[f_total >= osc_threshold]
    first_bright = float(bright[0]) if len(bright) > 0 else float("nan")
    peak_idx = int(np.argmax(f_total))
    threshold = float(ks_ionization_threshold_ev)
    if math.isfinite(first_bright) and math.isfinite(threshold):
        above_threshold = 1.0 if first_bright > threshold else 0.0
    else:
        above_threshold = float("nan")
    num_written = len(order) if max_transitions <= 0 else min(len(order), int(max_transitions))

    return {
        "co2_isolated_first_excitation_ev": float(energies_ev[0]),
        "co2_isolated_first_excitation_f": float(f_total[0]),
        "co2_isolated_first_bright_excitation_ev": first_bright,
        "co2_isolated_peak_ev": float(energies_ev[peak_idx]),
        "co2_isolated_peak_f": float(f_total[peak_idx]),
        "co2_isolated_ks_ionization_threshold_ev": threshold,
        "co2_isolated_first_bright_above_ks_threshold": above_threshold,
        "co2_isolated_num_transitions": float(num_written),
        "co2_isolated_num_transitions_computed": float(len(order)),
        "trk_sum": trk_sum,
        "trk_valence_electrons": float(valence_electrons),
        "trk_fraction": trk_fraction,
    }


def _case_row(material: str, site: str, status: str, params: dict, message: str = "", summary=None) -> dict:
    return {
        "material": material,
        "site": site,
        "tddft_status": status,
        "tddft_mode": TDDFT_MODE if status == STATUS_OK else "none",
        "error_message": message,
        **(summary if summary is not None else empty_summary()),
        **params,
    }


def run_tddft_workflow(args: argparse.Namespace) -> int:
    """Run all requested cases; returns the number of cases that did not finish with status ok."""
    ensure_directories()
    materials, sites = validate_inputs(args.materials, args.sites)
    params = ground_state_parameters(args)
    output_csv = RESULTS_DIR / "tddft_summary.csv"
    dft_summary_csv = RESULTS_DIR / "dft_summary.csv"
    dft_status = load_dft_status(dft_summary_csv)
    master = is_master()

    # Only cases requested in this invocation are written; rows of earlier runs are not carried over.
    rows: list[dict] = []
    for material in materials:
        for site in sites:
            _, xyz_path = dft_case_paths(material, site)
            status = None if dft_status is None else dft_status.get((material, site))
            if dft_status is None:
                message = f"Missing DFT summary: {dft_summary_csv}"
                rows.append(_case_row(material, site, "missing_dft", params, message))
                print0(f"[tddft] {material} {site} skipped: {message}")
                continue
            if status != STATUS_OK:
                message = f"DFT status is {status!r} (not 'ok') in {dft_summary_csv}"
                rows.append(_case_row(material, site, "skipped_dft_not_ok", params, message))
                print0(f"[tddft] {material} {site} skipped: {message}")
                continue
            if not xyz_path.exists():
                message = f"Missing optimized structure: {xyz_path}"
                rows.append(_case_row(material, site, "missing_dft", params, message))
                print0(f"[tddft] {material} {site} skipped: {message}")
                continue

            case_out = TDDFT_RESULTS_DIR / material / site
            case_out.mkdir(parents=True, exist_ok=True)
            transitions_csv = case_out / "transitions.csv"
            if master:
                # Never leave outputs of a previous run next to this run's results.
                for stale in (transitions_csv, case_out / "lrtddft.txt"):
                    stale.unlink(missing_ok=True)
            try:
                with timer(f"TDDFT (isolated CO2) -> {material} | {site}"):
                    energies_ev, f_total, f_xyz, valence, ks_threshold = run_lrtddft(
                        xyz_file=xyz_path,
                        out_dir=case_out,
                        params=params,
                    )
                    summary = summarize_transitions(
                        energies_ev=energies_ev,
                        f_total=f_total,
                        osc_threshold=args.osc_threshold,
                        max_transitions=args.max_transitions,
                        valence_electrons=valence,
                        ks_ionization_threshold_ev=ks_threshold,
                    )
                    if master:
                        keep = int(summary["co2_isolated_num_transitions"])
                        order = np.argsort(energies_ev)[:keep]
                        write_transitions_csv(transitions_csv, energies_ev[order], f_total[order], f_xyz[order])
                rows.append(_case_row(material, site, STATUS_OK, params, summary=summary))
                print0(
                    f"[tddft] {material} {site} isolated CO2: first={summary['co2_isolated_first_excitation_ev']:.4f} eV "
                    f"first bright(f>={args.osc_threshold:g})="
                    f"{summary['co2_isolated_first_bright_excitation_ev']:.4f} eV "
                    f"peak={summary['co2_isolated_peak_ev']:.4f} eV "
                    f"-eps_HOMO={summary['co2_isolated_ks_ionization_threshold_ev']:.4f} eV "
                    f"TRK={summary['trk_sum']:.3f}/{summary['trk_valence_electrons']:g}"
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                rows.append(_case_row(material, site, "failed", params, message))
                print0(f"[tddft] {material} {site} failed: {message}")

    if master:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print0(f"[tddft] wrote summary: {output_csv}")

    failures = [row for row in rows if row["tddft_status"] != STATUS_OK]
    if failures:
        names = ", ".join(f"{row['material']}/{row['site']}={row['tddft_status']}" for row in failures)
        print0(f"[tddft] {len(failures)} of {len(rows)} case(s) did not finish ok: {names}")
    return len(failures)


def main() -> None:
    _require_gpaw()
    args = parse_args()
    failures = run_tddft_workflow(args)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
