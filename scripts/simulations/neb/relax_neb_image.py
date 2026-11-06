"""Relax one image of a finished NEB band with exactly the DFT settings of the run.

A band image below both endpoints (e.g. image 3 of top_metal -> bridge) means that the
relaxed endpoint structures are not local minima of the flat physisorption surface. This
script takes that image as a start geometry and relaxes it (BFGS) to the nearest minimum,
so that its energy can be compared with the endpoints and the adsorption energy of the
run.

    python -m scripts.simulations.neb.relax_neb_image --results-dir RUN \\
        --pair top_metal_to_bridge --image 3 --fmax 0.02 --steps 300
    mpiexec -n 6 gpaw python -m scripts.simulations.neb.relax_neb_image ...   # MPI

Inputs (RUN = one simulation run):

* ``RUN/neb/<material>/<pair>/neb_final_images.traj``: final band of run_neb; frame
  ``--image`` (0 and the last frame are the endpoints and are refused).
* ``RUN/dft_summary.csv``: the DFT profile of both endpoints (must be the same), their
  ``adsorbed_energy_ev``, ``clean_slab_energy_ev`` and ``co2_reference_energy_ev``.
* ``RUN/config.yaml`` (or ``--config``): ``methods.dft`` ecut_ev, kpoints,
  fmax_ev_per_angstrom (default of ``--fmax``) and scf_force_convergence. The
  calculator is ``run_dft.build_calculator`` with the profile of the endpoints, as in
  run_neb; ecut, kpoints and profile are checked against ``neb_run_summary.json`` when
  present.
* Fixed atoms: the FixAtoms of the endpoint DFT structure
  (``RUN/dft/<material>/<site>/<material>_<site>_optimized.xyz``), else those stored in
  the image, else ``run_dft.apply_bottom_layer_constraint``.

Outputs in ``RUN/neb/<material>/<pair>/image<N>_relaxed/`` (written by MPI rank 0 only;
an existing folder is refused without ``--overwrite``):

* ``image<N>_relaxed.xyz`` (final geometry with energy and forces), ``relax.traj``,
  ``relax.log`` (BFGS) and ``gpaw.txt``;
* ``image<N>_relaxed.json``: energy, final fmax, steps, converged, the energy relative to
  each endpoint (dft_summary adsorbed energies and the band endpoint energies) and the
  implied adsorption energy E(image) - E(clean slab) - E(CO2) with the run's references.

GPAW is imported only when the calculation starts (inside ``main``), so the module can
be imported and tested without GPAW.
"""
from __future__ import annotations

import os

# Before GPAW (and NumPy's BLAS) is loaded: one OpenMP thread per MPI rank.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ADSORBATE_ATOMS = 3  # CO2 = last three atoms (O, C, O)
DEFAULT_STEPS = 300
DEFAULT_FMAX = 0.05  # used only when the run config has no methods.dft.fmax_ev_per_angstrom
FINAL_IMAGES_TRAJ = "neb_final_images.traj"  # run_neb.FINAL_IMAGES_TRAJ
NEB_SUMMARY_JSON = "neb_run_summary.json"  # run_neb.SUMMARY_JSON


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relax one NEB image with the DFT settings of the run.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Simulation run directory (RUN).")
    parser.add_argument("--pair", required=True,
                        help="NEB pair folder, e.g. top_metal_to_bridge (or top_metal:bridge).")
    parser.add_argument("--image", type=int, required=True, help="Index of the internal image in the final band.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--fmax", type=float, default=None,
                        help="BFGS force threshold (eV/A); default methods.dft.fmax_ev_per_angstrom.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Maximum BFGS steps.")
    parser.add_argument("--config", type=Path, default=None, help="Run config (default RUN/config.yaml).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing image<N>_relaxed/ folder.")
    return parser.parse_args(argv)


def pair_sites(pair: str) -> tuple[str, str]:
    """(initial, final) from 'a_to_b' or 'a:b' / 'a->b' (sites checked by run_neb.parse_pair)."""
    from scripts.simulations.neb import run_neb

    text = str(pair).strip()
    if "_to_" in text:
        initial, final = text.split("_to_", 1)
        return run_neb.parse_pair((initial, final))
    return run_neb.parse_pair(text)


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def dft_settings(config: dict, neb_summary: dict) -> dict:
    """ecut, kpts, fmax and scf_force_convergence of the run (config first, NEB summary as check)."""
    dft = (config.get("methods") or {}).get("dft") or {}
    ecut = dft.get("ecut_ev", neb_summary.get("ecut"))
    kpts = dft.get("kpoints", neb_summary.get("kpts"))
    if ecut is None or kpts is None:
        raise SystemExit("[relax-image] methods.dft.ecut_ev / kpoints not found in the run config "
                         "or in neb_run_summary.json")
    ecut, kpts = float(ecut), [int(value) for value in kpts]
    if neb_summary.get("ecut") is not None and not math.isclose(float(neb_summary["ecut"]), ecut):
        raise SystemExit(f"[relax-image] NEB ecut {neb_summary['ecut']} differs from the run config {ecut}")
    if neb_summary.get("kpts") is not None and [int(v) for v in neb_summary["kpts"]] != kpts:
        raise SystemExit(f"[relax-image] NEB kpts {neb_summary['kpts']} differ from the run config {kpts}")
    scf = dft.get("scf_force_convergence")
    return {
        "ecut": ecut,
        "kpts": kpts,
        "fmax": float(dft.get("fmax_ev_per_angstrom", DEFAULT_FMAX)),
        "scf_force_convergence": float(scf) if scf is not None else None,
    }


def endpoint_rows(results_dir: Path, material: str, initial: str, final: str) -> dict[str, dict]:
    from scripts.simulations.neb import run_neb

    rows = run_neb.read_dft_summary(results_dir)
    found = {}
    for site in (initial, final):
        row = rows.get((material, site))
        if row is None:
            raise SystemExit(f"[relax-image] no dft_summary.csv row for {material} | {site}")
        if row.get("dft_status") != "ok":
            raise SystemExit(f"[relax-image] DFT of {material} | {site} has status {row.get('dft_status')!r}")
        found[site] = row
    profiles = {site: row.get("dft_profile", "") for site, row in found.items()}
    if len(set(profiles.values())) != 1:
        raise SystemExit(f"[relax-image] endpoints use different DFT profiles: {profiles}")
    return found


def fixed_atoms(image, results_dir: Path, material: str, site: str) -> tuple[list[int], str]:
    """Fixed indices and their source: endpoint DFT structure, else the image, else bottom layers."""
    from scripts.simulations.dft import run_dft
    from scripts.simulations.neb import run_neb

    path = run_neb.dft_structure_path(results_dir, material, site)
    if path.is_file():
        from ase.io import read

        endpoint = read(path, parallel=False)
        indices = run_neb.fixed_indices(endpoint)
        if indices:
            return indices, str(path)
    indices = run_neb.fixed_indices(image)
    if indices:
        return indices, "neb image"
    probe = image.copy()
    run_dft.apply_bottom_layer_constraint(probe, adsorbate_atoms=ADSORBATE_ATOMS)
    return run_neb.fixed_indices(probe), "run_dft.apply_bottom_layer_constraint"


def stored_energy(atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except Exception:  # no calculator results stored in the frame
        return float("nan")


def relative_energies(energy: float, rows: dict[str, dict], initial: str, final: str,
                      band_initial: float, band_final: float) -> dict:
    """Energy of the relaxed image against both endpoints and the implied E_ads."""
    e_initial = safe_float(rows[initial].get("adsorbed_energy_ev"))
    e_final = safe_float(rows[final].get("adsorbed_energy_ev"))
    clean = safe_float(rows[initial].get("clean_slab_energy_ev"))
    co2 = safe_float(rows[initial].get("co2_reference_energy_ev"))
    references_match = all(
        math.isclose(safe_float(rows[final].get(key)), safe_float(rows[initial].get(key)), abs_tol=1e-6)
        for key in ("clean_slab_energy_ev", "co2_reference_energy_ev")
    )
    return {
        "initial_dft_energy_ev": e_initial,
        "final_dft_energy_ev": e_final,
        "e_rel_initial_ev": energy - e_initial,
        "e_rel_final_ev": energy - e_final,
        "initial_band_energy_ev": band_initial,
        "final_band_energy_ev": band_final,
        "e_rel_initial_band_ev": energy - band_initial,
        "e_rel_final_band_ev": energy - band_final,
        "clean_slab_energy_ev": clean,
        "co2_reference_energy_ev": co2,
        "references_match_between_endpoints": bool(references_match),
        "implied_adsorption_energy_ev": energy - clean - co2,
        "initial_adsorption_energy_ev": safe_float(rows[initial].get("adsorption_energy_ev")),
        "final_adsorption_energy_ev": safe_float(rows[final].get("adsorption_energy_ev")),
    }


def output_paths(pair_dir: Path, image: int) -> dict[str, Path]:
    folder = pair_dir / f"image{image}_relaxed"
    return {
        "dir": folder,
        "xyz": folder / f"image{image}_relaxed.xyz",
        "traj": folder / "relax.traj",
        "log": folder / "relax.log",
        "txt": folder / "gpaw.txt",
        "json": folder / f"image{image}_relaxed.json",
    }


def max_free_force(atoms) -> float:
    forces = atoms.get_forces()  # constrained components are zero
    return float(np.sqrt((forces ** 2).sum(axis=1)).max())


def relax_image(args: argparse.Namespace, world=None) -> dict:
    """Run the relaxation; returns the JSON record (written by rank 0)."""
    from ase.constraints import FixAtoms
    from ase.io import read, write
    from ase.optimize import BFGS

    from scripts.simulations.dft import run_dft
    from scripts.simulations.neb import run_neb

    world = world if world is not None else run_neb.get_world()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"[relax-image] results directory not found: {results_dir}")
    if args.steps < 1:
        raise SystemExit("[relax-image] --steps must be at least 1")
    initial, final = pair_sites(args.pair)
    pair_name = f"{initial}_to_{final}"
    pair_dir = results_dir / "neb" / args.material / pair_name
    traj_path = pair_dir / FINAL_IMAGES_TRAJ
    if not traj_path.is_file():
        raise SystemExit(f"[relax-image] final NEB band not found: {traj_path}")
    band = list(read(traj_path, index=":", parallel=False))
    if not 0 < args.image < len(band) - 1:
        raise SystemExit(f"[relax-image] --image must be an internal image, 1..{len(band) - 2} "
                         f"(band of {len(band)} frames in {traj_path})")
    paths = output_paths(pair_dir, args.image)
    if paths["dir"].exists() and any(paths["dir"].iterdir()) and not args.overwrite:
        raise SystemExit(f"[relax-image] {paths['dir']} exists; use --overwrite to replace it")

    config = run_neb.load_config(args.config, results_dir)
    summary_path = pair_dir / NEB_SUMMARY_JSON
    neb_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    settings = dft_settings(config, neb_summary)
    fmax = float(args.fmax) if args.fmax is not None else settings["fmax"]
    if fmax <= 0:
        raise SystemExit("[relax-image] --fmax must be positive")
    rows = endpoint_rows(results_dir, args.material, initial, final)
    profile = rows[initial].get("dft_profile", "")
    if neb_summary.get("dft_profile") and neb_summary["dft_profile"] != profile:
        raise SystemExit(f"[relax-image] NEB profile {neb_summary['dft_profile']!r} differs from the "
                         f"dft_summary profile {profile!r}")
    calculator = run_neb.calculator_settings(profile, settings["ecut"], settings["kpts"])
    calculator["scf_force_convergence"] = settings["scf_force_convergence"]

    start = band[args.image]
    band_energy = stored_energy(start)
    band_initial, band_final = stored_energy(band[0]), stored_energy(band[-1])
    fixed, fixed_source = fixed_atoms(start, results_dir, args.material, initial)
    atoms = start.copy()
    atoms.set_constraint(FixAtoms(indices=fixed))

    if world.rank == 0:
        paths["dir"].mkdir(parents=True, exist_ok=True)
        if args.overwrite and paths["json"].exists():
            paths["json"].unlink()  # never leave the result of an earlier run next to a new one
    world.barrier()

    atoms.calc = run_dft.build_calculator(
        ecut=calculator["ecut"],
        kpts=tuple(calculator["kpts"]),
        txt_file=paths["txt"],
        mode=calculator["mode"],
        smearing=calculator["smearing"],
        maxiter=calculator["maxiter"],
        mixer=tuple(calculator["mixer"]) if calculator["mixer"] is not None else None,
        scf_force_convergence=calculator["scf_force_convergence"],
    )
    # ASE optimizers write the log and trajectory on rank 0 only.
    optimizer = BFGS(atoms, logfile=str(paths["log"]), trajectory=str(paths["traj"]))
    converged = bool(optimizer.run(fmax=fmax, steps=args.steps))
    energy = float(atoms.get_potential_energy())
    final_fmax = max_free_force(atoms)
    displacement = run_neb.adsorbate_displacement(start, atoms)

    record = {
        "material": args.material,
        "pair": pair_name,
        "initial_site": initial,
        "final_site": final,
        "image_index": int(args.image),
        "band_frames": len(band),
        "source_traj": str(traj_path),
        "source_sha256": run_neb.sha256_file(traj_path),
        "dft_summary_sha256": run_neb.sha256_file(results_dir / "dft_summary.csv"),
        "config_path": config.get("_config_path"),
        "calculator": calculator,
        "fixed_indices": fixed,
        "fixed_indices_source": fixed_source,
        "optimizer": "BFGS",
        "fmax_target_ev_per_ang": fmax,
        "max_steps": int(args.steps),
        "steps": int(optimizer.nsteps),
        "converged": converged,
        "final_fmax_ev_per_ang": final_fmax,
        "energy_ev": energy,
        "band_image_energy_ev": band_energy,
        "relaxation_energy_ev": energy - band_energy,
        "max_adsorbate_displacement_ang": displacement,
        **relative_energies(energy, rows, initial, final, band_initial, band_final),
        "mpi_size": int(world.size),
        "command": " ".join(sys.argv),
    }
    if world.rank == 0:
        write(paths["xyz"], atoms, parallel=False)
        paths["json"].write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    world.barrier()
    return record


def main(argv=None) -> int:
    args = parse_args(argv)
    record = relax_image(args)
    from scripts.common.timer import print0

    print0(
        f"[relax-image] {record['pair']} image {record['image_index']}: "
        f"{'converged' if record['converged'] else 'NOT converged'} after {record['steps']} steps "
        f"(fmax {record['final_fmax_ev_per_ang']:.4f} eV/A); E = {record['energy_ev']:.6f} eV, "
        f"{record['e_rel_initial_ev']:+.4f} eV vs {record['initial_site']}, "
        f"{record['e_rel_final_ev']:+.4f} eV vs {record['final_site']}; "
        f"implied E_ads {record['implied_adsorption_energy_ev']:+.4f} eV"
    )
    return 0 if record["converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
