"""Superseded flat-layout pipeline (writes into the shared results/ tree, which held the
invalid first campaign). Kept for provenance; use scripts.workflows.run_simulations.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PIPELINE_MODULES = {
    "geometry": "scripts.simulations.dft.build_geometries",
    "dft": "scripts.simulations.dft.run_dft",
    "tddft": "scripts.simulations.tddft.run_tddft",
    "pathways": "scripts.simulations.co2rr.run_co2rr_pathways",
    "analysis": "scripts.analysis.analyze_results",
    "co2_analysis": "scripts.analysis.analysis_co2_reduction",
    "export": "scripts.datasets.export_qml_dataset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ZnO/TiO2/CeO2 DFT-TDDFT screening pipeline.")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-dft", action="store_true")
    parser.add_argument("--skip-tddft", action="store_true")
    parser.add_argument("--skip-pathways", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-co2-analysis", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def run_step(step_name: str) -> None:
    command = [sys.executable, "-m", PIPELINE_MODULES[step_name]]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    if not args.skip_geometry:
        run_step("geometry")
    if not args.skip_dft:
        run_step("dft")
    if not args.skip_tddft:
        run_step("tddft")
    if not args.skip_pathways:
        run_step("pathways")
    if not args.skip_analysis:
        run_step("analysis")
    if not args.skip_co2_analysis:
        run_step("co2_analysis")
    if not args.skip_export:
        run_step("export")


if __name__ == "__main__":
    main()
