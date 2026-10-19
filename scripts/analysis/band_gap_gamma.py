"""Recompute Kohn-Sham band gaps of a finished DFT run on a Gamma-centred grid.

Reads the saved .gpw files, so the relaxations are not repeated. The relaxation
mesh is taken from the run's config.yaml unless --kpts is given.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from scripts.common.timer import print0
from scripts.simulations.dft.run_dft import compute_band_gap_gamma, compute_band_gap_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="results/simulations/<run-id> directory of a finished run.")
    parser.add_argument("--material", default="ZnO")
    parser.add_argument("--kpts", nargs=3, type=int, metavar=("KX", "KY", "KZ"),
                        help="Relaxation mesh (default: methods.dft.kpoints in config.yaml).")
    return parser.parse_args()


def gpw_files(run_dir: Path, material: str) -> list[tuple[str, Path]]:
    """Clean slab first, then every adsorbed case that saved a .gpw."""
    base = run_dir / "dft" / material
    cases = [("clean", base / "clean" / "clean_slab.gpw")]
    for path in sorted(base.glob(f"*/{material}_*.gpw")):
        cases.append((path.parent.name, path))
    return [(name, path) for name, path in cases if path.exists()]


def main() -> None:
    from gpaw import GPAW
    from gpaw.mpi import world

    args = parse_args()
    if args.kpts:
        kpts = tuple(args.kpts)
    else:
        config = yaml.safe_load((args.run_dir / "config.yaml").read_text())
        kpts = tuple(int(n) for n in config["methods"]["dft"]["kpoints"])

    rows = []
    for name, path in gpw_files(args.run_dir, args.material):
        calc = GPAW(str(path), txt=None)
        mesh_gap = compute_band_gap_safe(calc)
        gamma_gap = compute_band_gap_gamma(calc, kpts, path.with_suffix(".gap_gamma.txt"))
        rows.append({"case": name, "kpts": "x".join(map(str, kpts)),
                     "band_gap_mesh_ev": mesh_gap, "band_gap_gamma_ev": gamma_gap})
        print0(f"[gap] {name}: mesh {mesh_gap:.4f} eV | Gamma grid {gamma_gap:.4f} eV")

    if not rows:
        raise SystemExit(f"No .gpw files found under {args.run_dir / 'dft' / args.material}")
    if world.rank == 0:
        output = args.run_dir / "band_gap_gamma.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print0(f"[gap] wrote {output}")


if __name__ == "__main__":
    main()
