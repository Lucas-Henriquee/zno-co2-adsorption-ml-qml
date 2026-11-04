from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml
from scripts.workflows.active_environment import active_environment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "analysis" / "zno_current.yaml"
METHOD_ORDER = (
    "screening",
    "co2rr_corrections",
    "co2_reduction",
    "neb",
    "md_trajectory",
    "structures",
    "paper_figures",
    "paper_tables",
    "model_analysis",
)
DEPENDENCIES = {
    "screening": (),
    "co2rr_corrections": (),
    "co2_reduction": ("co2rr_corrections",),
    "neb": (),
    "md_trajectory": (),
    "structures": (),
    "paper_figures": ("co2rr_corrections", "neb", "md_trajectory"),
    "paper_tables": ("co2rr_corrections", "neb", "md_trajectory"),
    "model_analysis": (),
}
# paper_figures needs these methods only for the figures that read their outputs.
FIGURE_DEPENDENCIES = {
    "fig05": "co2rr_corrections",
    "fig06": "md_trajectory",
    "fig07": "neb",
}
ALIASES = {
    "science": ["screening", "co2rr_corrections", "co2_reduction"],
    "paper": [
        "co2rr_corrections",
        "neb",
        "md_trajectory",
        "structures",
        "paper_figures",
        "paper_tables",
    ],
    "models": ["model_analysis"],
    "model-analysis": ["model_analysis"],
}
OUTPUT_SUBDIRECTORIES = {
    "screening": Path("science/screening"),
    "co2rr_corrections": Path("science/co2rr"),
    "co2_reduction": Path("science/co2_reduction"),
    "neb": Path("science/neb"),
    "md_trajectory": Path("science/md"),
    "structures": Path("science/structures"),
    "paper_figures": Path("paper"),
    "paper_tables": Path("paper"),
    "model_analysis": Path("models"),
}
PAPER_DIRECTORY = Path("paper")
PAPER_METHODS = ("paper_figures", "paper_tables")
REGISTRY_NAME = "figure_registry.csv"
REGISTRY_FIELDS = ("item", "file", "producer_module", "inputs", "sha256", "note")
SKIPPED_NOTE = "not written: skipped by the producer (see logs/{method}.log)"
PAPER_FIGURE_NAMES = {
    "fig03": "fig03_dft_descriptors",
    "fig04": "fig04_tddft_spectra",
    "fig05": "fig05_co2rr_pathway",
    "fig06": "fig06_md_distance",
    "fig07": "fig07_neb_profiles",
}
DIAGNOSTIC_NOTE = "diagnostic draft: geometry warnings (allow_geometry_warnings), not publication-ready"
CO2RR_VARIANTS = ("a", "b", "c")
STRUCTURE_SITES = ("top_metal", "top_oxygen", "bridge")
CORRECTED_SUMMARY = "co2rr_corrected_summary.csv"
MD_SUMMARY = "md_trajectory_analysis_summary.csv"
# Paper items: producer module and the analysis method whose run inputs they use.
PAPER_ITEMS = {
    "fig01_adsorption_structures": ("scripts.visualization.export_pymol", "structures"),
    "fig03_dft_descriptors": ("scripts.visualization.paper_figures", "paper_figures"),
    "fig04_tddft_spectra": ("scripts.visualization.paper_figures", "paper_figures"),
    "fig05_co2rr_pathway": ("scripts.visualization.paper_figures", "paper_figures"),
    "fig06_md_distance": ("scripts.visualization.paper_figures", "paper_figures"),
    "fig07_neb_profiles": ("scripts.visualization.paper_figures", "paper_figures"),
    "table02_site_summary": ("scripts.analysis.paper_tables", "paper_tables"),
}
# Folders of the science methods copied into A/paper: (method, source folder below the
# method directory, paper folder, file that must be present). fig03-fig07 are drawn by
# paper_figures itself; fig01 is rendered by export_pymol (PyMOL when importable, else the
# ASE/matplotlib fallback) in two variants (C-O lengths; shortest CO2-surface contact),
# which are copied to A/paper/ and not into A/paper/fig01_pymol/ (one registry row each).
PYMOL_SUBDIRECTORY = "pymol"
FIG01_PYMOL_DIRECTORY = "fig01_pymol"
FIG01_PML = "fig01_adsorption_structures.pml"
FIG01_PNG = "fig01_adsorption_structures.png"
FIG01_CONTACT_PNG = "fig01_adsorption_structures_contact.png"
FIG01_PNGS = (FIG01_PNG, FIG01_CONTACT_PNG)
FIG01_RENDER_INFO = "render_info.json"
FIG01_NOTE = f"made in PyMOL from {FIG01_PYMOL_DIRECTORY}/{FIG01_PML}"
FIG01_RENDER_NOTES = {
    "pymol": f"rendered by export_pymol with PyMOL (headless, the commands of {FIG01_PYMOL_DIRECTORY}/{FIG01_PML})",
    "ase": (f"rendered by export_pymol with the ASE/matplotlib fallback (PyMOL not importable; same atoms "
            f"and view as {FIG01_PYMOL_DIRECTORY}/{FIG01_PML})"),
}
STRUCTURE_RENDERERS = ("auto", "pymol", "ase", "none")
PAPER_COPIES = (
    ("structures", PYMOL_SUBDIRECTORY, FIG01_PYMOL_DIRECTORY, FIG01_PML),
)
PREVIEW_DIAGNOSTIC = "zno_adsorption_sites_combined_diagnostic.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated scientific, model, table, and figure analyses."
    )
    parser.add_argument(
        "methods",
        nargs="*",
        help=(
            f"{', '.join(METHOD_ORDER)}; aliases: {', '.join(ALIASES)}; or all. "
            "Dependencies are added automatically."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scientific-results", type=Path)
    parser.add_argument("--model-run", type=Path)
    parser.add_argument("--run-id", help="Safe directory name for this analysis run.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without creating files.",
    )
    parser.add_argument("--list-methods", action="store_true")
    return parser.parse_args(argv)


def fail(message: str) -> None:
    raise SystemExit(f"[error] {message}")


def resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def display_path(path: Path) -> str:
    """Path relative to the project root when below it, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    if not resolved.is_file():
        fail(f"Configuration file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        fail(f"Configuration must contain a YAML mapping: {resolved}")
    if payload.get("schema_version") != 1:
        fail("Unsupported or missing schema_version; expected 1.")
    payload["_config_path"] = str(resolved)
    return payload


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        fail(f"Configuration field '{key}' must be a mapping.")
    return value


def method_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    methods = require_mapping(config, "methods")
    value = methods.get(method)
    if not isinstance(value, dict):
        fail(f"Configuration field 'methods.{method}' must be a mapping.")
    return value


def material_name(config: dict[str, Any]) -> str:
    value = str(require_mapping(config, "input").get("material", "ZnO"))
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        fail(f"Configuration field 'input.material' is not a material name: {value}")
    return value


def co2rr_variant(config: dict[str, Any]) -> str:
    value = str(method_config(config, "co2rr_corrections").get("variant", "c"))
    if value not in CO2RR_VARIANTS:
        fail(f"methods.co2rr_corrections.variant must be one of {', '.join(CO2RR_VARIANTS)}: {value}")
    return value


def positive_int(settings: dict[str, Any], key: str, default: int, method: str) -> int:
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"methods.{method}.{key} must be a positive integer: {value!r}")
    return value


def positive_float(settings: dict[str, Any], key: str, default: float, method: str) -> float:
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        fail(f"methods.{method}.{key} must be a positive number: {value!r}")
    return float(value)


def format_number(value: float) -> str:
    """Two decimals when exact (0.1 -> '0.10'), else the shortest repr."""
    return f"{value:.2f}" if round(value, 2) == value else repr(value)


def normalize_methods(raw: list[str]) -> list[str]:
    if not raw:
        fail("Select at least one analysis method or use 'all'.")
    requested: list[str] = []
    for name in raw:
        normalized = name.lower()
        if normalized == "all":
            if len(raw) != 1:
                fail("Use 'all' by itself, without additional method names.")
            return list(METHOD_ORDER)
        if normalized in ALIASES:
            requested.extend(ALIASES[normalized])
        elif normalized in METHOD_ORDER:
            requested.append(normalized)
        else:
            fail(f"Unknown analysis method: {normalized}")
    return list(dict.fromkeys(requested))


def method_dependencies(method: str, config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """DEPENDENCIES; with a config, paper_figures needs only what the selected figures read."""
    if method == "paper_figures" and config is not None:
        figures = paper_figure_selection(config)
        return tuple(dict.fromkeys(FIGURE_DEPENDENCIES[f] for f in figures if f in FIGURE_DEPENDENCIES))
    return DEPENDENCIES[method]


def resolve_dependencies(selected: list[str], config: dict[str, Any] | None = None) -> list[str]:
    required = set(selected)

    def add(method: str) -> None:
        for dependency in method_dependencies(method, config):
            if dependency not in required:
                required.add(dependency)
                add(dependency)

    for method in selected:
        add(method)
    return [method for method in METHOD_ORDER if method in required]


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        fail("Run ID may only contain letters, numbers, periods, underscores, and hyphens.")
    return run_id


def default_run_id(output_root: Path | None = None) -> str:
    from scripts.common.run_names import next_run_name
    return next_run_name(output_root or PROJECT_ROOT / "results/analyses", "analysis")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ["ase", "numpy", "pandas", "matplotlib", "pyyaml"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_state() -> dict[str, Any]:
    """Commit and dirty flag of the checkout; None values outside a Git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "dirty": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"commit": commit or None, "dirty": bool(status.strip())}


def simulation_run_manifest(scientific_results: Path) -> dict[str, Any]:
    path = scientific_results / "run_manifest.json"
    if not path.is_file():
        return {"path": display_path(path), "sha256": None, "present": False}
    return {"path": display_path(path), "sha256": file_sha256(path), "present": True}


def discover_benchmark_summary(model_run: Path) -> Path:
    matches = sorted(model_run.rglob("benchmark_summary.csv"))
    if not matches:
        fail(f"No benchmark_summary.csv found below: {model_run}")
    if len(matches) > 1:
        fail(f"Multiple benchmark summaries found below {model_run}; select a single model run.")
    return matches[0]


def require_matches(root: Path, pattern: str, description: str) -> list[Path]:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if not matches:
        fail(f"No {description} found: {root / pattern}")
    return matches


def desorption_threshold(config: dict[str, Any]) -> str:
    """methods.md_trajectory.desorption_threshold_ang, passed to md_trajectory_analysis."""
    settings = method_config(config, "md_trajectory")
    return format_number(positive_float(settings, "desorption_threshold_ang", 4.5, "md_trajectory"))


def paper_figure_selection(config: dict[str, Any]) -> list[str]:
    """methods.paper_figures.figures: null (absent) = every figure; else a non-empty subset."""
    figures = method_config(config, "paper_figures").get("figures")
    if figures is None:
        return list(PAPER_FIGURE_NAMES)
    if (not isinstance(figures, list) or not figures
            or not all(str(f) in PAPER_FIGURE_NAMES for f in figures)):
        fail(f"methods.paper_figures.figures must be null or a non-empty subset of "
             f"{list(PAPER_FIGURE_NAMES)}: {figures!r}")
    return list(dict.fromkeys(str(f) for f in figures))


def md_seed(config: dict[str, Any]) -> str:
    """methods.paper_figures.md_seed: the MD seed drawn in fig06 (default 1234).

    A non-negative integer without leading zeros ("0123" would never match seed_123).
    """
    value = method_config(config, "paper_figures").get("md_seed", 1234)
    if isinstance(value, bool) or not re.fullmatch(r"0|[1-9][0-9]*", str(value)):
        fail(f"methods.paper_figures.md_seed must be a non-negative integer without leading zeros: {value!r}")
    return str(value)


def expected_paper_files(config: dict[str, Any], resolved: list[str]) -> dict[str, str]:
    """Files paper_figures / paper_tables should write -> producing method."""
    expected: dict[str, str] = {}
    if "paper_figures" in resolved:
        for figure in paper_figure_selection(config):
            stem = PAPER_FIGURE_NAMES[figure]
            for suffix in (".png", ".pdf", "_data.csv"):
                expected[stem + suffix] = "paper_figures"
    if "paper_tables" in resolved:
        for ext in ("csv", "md", "tex"):
            expected[f"table02_site_summary.{ext}"] = "paper_tables"
    return expected


def structure_renderer(config: dict[str, Any]) -> str:
    value = str(method_config(config, "structures").get("render", "auto"))
    if value not in STRUCTURE_RENDERERS:
        fail(f"methods.structures.render must be one of {', '.join(STRUCTURE_RENDERERS)}: {value}")
    return value


def allow_geometry_warnings(config: dict[str, Any]) -> bool:
    return bool(method_config(config, "structures").get("allow_geometry_warnings", False))


def structure_geometry_dir(config: dict[str, Any], scientific_results: Path) -> Path:
    """Starting geometry of the structures figure: configured, else the run's, else data/."""
    material = material_name(config)
    configured = method_config(config, "structures").get("geometry_dir")
    if configured:
        return resolve_project_path(Path(str(configured))) / material
    run_geometry = scientific_results / "data" / "geometries" / material
    if (run_geometry / "metadata.json").is_file() and (run_geometry / "clean_slab.traj").is_file():
        return run_geometry
    return PROJECT_ROOT / "data" / "geometries" / material


def collect_inputs(
    method: str,
    config: dict[str, Any],
    scientific_results: Path,
    model_run: Path,
) -> list[Path]:
    """Input files of one method below the simulation or model run.

    Required inputs must exist (else an error); optional ones are recorded when present.
    Files written by earlier methods of the same analysis run are not inputs.
    """
    paths: list[Path] = []
    optional: list[Path] = []
    run = scientific_results
    material = material_name(config) if method != "model_analysis" else ""
    if method in {"screening", "co2_reduction"}:
        paths.extend([run / "dft_summary.csv", run / "tddft_summary.csv"])
    if method in {"paper_figures", "paper_tables"}:
        # Producers skip what is missing ([skip]); only the DFT summary is always needed.
        paths.append(run / "dft_summary.csv")
        if method == "paper_figures" and "fig04" in paper_figure_selection(config):
            paths.append(run / "tddft_summary.csv")
            transitions = sorted(run.glob(f"tddft/{material}/*/transitions.csv"))
            if not transitions:
                fail(f"No TDDFT transition files found below (needed by fig04): {run / 'tddft' / material}")
            paths.extend(transitions)
        else:
            optional.append(run / "tddft_summary.csv")
    if method == "co2_reduction":
        pathway = run / "co2rr_pathway_summary.csv"
        if bool(method_config(config, method).get("require_pathway_results", False)):
            paths.append(pathway)
        elif pathway.is_file():
            paths.append(pathway)
    if method == "co2rr_corrections":
        paths.append(run / "co2rr_pathway_summary.csv")
        if (run / "pathways").is_dir():
            paths.extend(require_matches(run / "pathways", "**/energy.json", "CO2RR energy.json files"))
        else:
            fail(f"CO2RR pathway directory does not exist: {run / 'pathways'}")
        optional.append(run / "config.yaml")
    if method == "neb":
        base = run / "neb" / material
        paths.extend(require_matches(base, "*/neb_run_summary.json", "NEB summaries (neb_run_summary.json)"))
        optional.extend(sorted(base.glob("*/neb_final_images.traj")))
        optional.append(run / "config.yaml")
    if method == "md_trajectory":
        paths.extend(require_matches(run / "md" / material, "*/seed_*/md_*.traj", "MD trajectories"))
        optional.append(run / "config.yaml")
    if method == "structures":
        geometry = structure_geometry_dir(config, run)
        paths.append(run / "dft_summary.csv")
        paths.extend(
            run / "dft" / material / site / f"{material}_{site}_optimized.xyz" for site in STRUCTURE_SITES
        )
        paths.extend([geometry / "metadata.json", geometry / "clean_slab.traj"])
    if method == "co2_reduction":
        # Relaxed gas-phase CO2 of the run: baseline of the C-O stretch.
        optional.extend(sorted(run.glob("dft/_reference/*/co2_ref.*_optimized.cif")))
        optional.extend(sorted(run.glob("pathways/_references/*/co2/co2_optimized.xyz")))
    if method == "model_analysis":
        paths.append(discover_benchmark_summary(model_run))
        confusion = sorted(model_run.rglob("confusion_matrix.csv"))
        if not confusion:
            fail(f"No confusion matrices found below: {model_run}")
        paths.extend(confusion)

    missing = [path for path in paths if not path.is_file()]
    if missing:
        fail(f"Missing inputs of '{method}': " + ", ".join(str(path) for path in missing))
    paths.extend(path for path in optional if path.is_file())
    return list(dict.fromkeys(paths))


def build_commands(
    method: str,
    config: dict[str, Any],
    scientific_results: Path,
    model_run: Path,
    run_dir: Path,
) -> list[list[str]]:
    python = sys.executable
    output = run_dir / OUTPUT_SUBDIRECTORIES[method]
    settings = {} if method == "screening" else method_config(config, method)
    if method == "screening":
        return [[
            python,
            "-m",
            "scripts.analysis.analyze_results",
            "--results-dir",
            str(scientific_results),
            "--output-dir",
            str(output),
        ]]
    if method == "co2rr_corrections":
        return [
            [
                python, "-m", "scripts.analysis.co2rr_corrections",
                "--results-dir", str(scientific_results),
                "--output-dir", str(output),
            ],
            [
                python, "-m", "scripts.visualization.plot_co2rr_pathway",
                "--summary", str(scientific_results / "co2rr_pathway_summary.csv"),
                "--corrected", str(output / CORRECTED_SUMMARY),
                "--variant", co2rr_variant(config),
                "--material", material_name(config),
                "--output-dir", str(output),
            ],
        ]
    if method == "co2_reduction":
        return [[
            python,
            "-m",
            "scripts.analysis.analysis_co2_reduction",
            "--results-dir",
            str(scientific_results),
            "--output-dir",
            str(output),
            "--co2rr-variant",
            co2rr_variant(config),
            "--corrected-summary",
            str(run_dir / OUTPUT_SUBDIRECTORIES["co2rr_corrections"] / CORRECTED_SUMMARY),
        ]]
    if method == "neb":
        base = scientific_results / "neb" / material_name(config)
        resolution = settings.get("barrier_resolution_ev")
        if resolution is not None:
            if isinstance(resolution, bool) or not isinstance(resolution, (int, float)) or resolution < 0:
                fail(f"methods.neb.barrier_resolution_ev must be null or a non-negative number: {resolution!r}")
            resolution_args = ["--barrier-resolution", str(float(resolution))]
        else:
            resolution_args = []  # the producers read methods.neb of the run's config.yaml
        plot = [
            python, "-m", "scripts.visualization.plot_neb",
            "--base", str(base), "--output-dir", str(output), *resolution_args,
        ]
        if bool(settings.get("smooth_band", True)):
            plot.append("--smooth-band")
        if "dpi" in settings:
            plot.extend(["--dpi", str(positive_int(settings, "dpi", 600, method))])
        return [
            [
                python, "-m", "scripts.analysis.neb_energy_table",
                "--base", str(base), "--output-dir", str(output), *resolution_args,
            ],
            plot,
        ]
    if method == "md_trajectory":
        return [[
            python, "-m", "scripts.analysis.md_trajectory_analysis",
            "--results-dir", str(scientific_results),
            "--material", material_name(config),
            "--output-dir", str(output),
            "--desorption-threshold", desorption_threshold(config),
        ]]
    if method == "structures":
        material = material_name(config)
        geometry = structure_geometry_dir(config, scientific_results)
        return [[
            # Fig. 1: relaxed structures (PDB/CIF), PyMOL script and the rendered panels.
            python, "-m", "scripts.visualization.export_pymol",
            "--input-dir", str(scientific_results / "dft" / material),
            "--summary", str(scientific_results / "dft_summary.csv"),
            "--material", material,
            "--output-dir", str(output / PYMOL_SUBDIRECTORY),
            "--render", structure_renderer(config),
            "--dpi", str(positive_int(settings, "dpi", 600, method)),
        ], [
            # Matplotlib preview of the same structures (not copied into A/paper).
            python, "-m", "scripts.visualization.render_adsorption_structures",
            "--input-dir", str(scientific_results / "dft" / material),
            "--summary", str(scientific_results / "dft_summary.csv"),
            "--site-metadata", str(geometry / "metadata.json"),
            "--initial-slab", str(geometry / "clean_slab.traj"),
            "--output-dir", str(output),
            "--dpi", str(positive_int(settings, "dpi", 600, method)),
            *(["--allow-geometry-warnings"] if allow_geometry_warnings(config) else []),
        ]]
    if method == "paper_figures":
        command = [
            python, "-m", "scripts.visualization.paper_figures",
            "--results-dir", str(scientific_results),
            "--analysis-dir", str(run_dir),
            "--output-dir", str(output),
            "--material", material_name(config),
            "--dpi", str(positive_int(settings, "dpi", 600, method)),
            "--tddft-sigma-ev", format_number(positive_float(settings, "tddft_sigma_ev", 0.10, method)),
            "--md-seed", md_seed(config),
            "--co2rr-variant", co2rr_variant(config),
        ]
        if settings.get("figures") is not None:
            command.extend(["--figures", *paper_figure_selection(config)])
        return [command]
    if method == "paper_tables":
        return [[
            python, "-m", "scripts.analysis.paper_tables",
            "--results-dir", str(scientific_results),
            "--analysis-dir", str(run_dir),
            "--output-dir", str(output),
            "--material", material_name(config),
            "--co2rr-variant", co2rr_variant(config),
        ]]
    benchmark_summary = discover_benchmark_summary(model_run)
    model_tables = output / "tables"
    model_figures = output / "figures"
    return [
        [
            python,
            "-m",
            "scripts.analysis.analyze_model_results",
            "--benchmark-summary",
            str(benchmark_summary),
            "--output-dir",
            str(model_tables),
        ],
        [
            python,
            "-m",
            "scripts.visualization.plot_current_model_results",
            "--comparison",
            str(model_tables / "model_comparison.csv"),
            "--model-run-dir",
            str(model_run),
            "--output-dir",
            str(model_figures),
            "--dpi",
            str(int(settings.get("dpi", 300))),
        ],
    ]


def inventory(paths: list[Path]) -> list[dict[str, Any]]:
    """Path (relative to the project root when below it), size and sha256 of each file."""
    return [
        {"path": display_path(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(paths)
    ]


def snapshot(directory: Path) -> dict[Path, tuple[int, int]]:
    if not directory.is_dir():
        return {}
    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in directory.rglob("*") if path.is_file()
    }


def new_or_changed(directory: Path, before: dict[Path, tuple[int, int]]) -> list[Path]:
    """Files written below a (possibly shared) method directory since `before`."""
    after = snapshot(directory)
    return sorted(path for path, state in after.items() if before.get(path) != state)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def run_command(command: list[str], log_path: Path) -> None:
    print(f"[command] {shlex.join(command)}", flush=True)
    environment = os.environ.copy()
    cache_root = log_path.parent / ".cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    environment["XDG_CACHE_HOME"] = str(cache_root)
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in [str(PROJECT_ROOT), existing_python_path] if item
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[{utc_now()}] {shlex.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


# ------------------------------------------------------------------ paper collection


def paper_collection_enabled(resolved: list[str]) -> bool:
    return any(method in resolved for method in PAPER_METHODS)


def paper_copy_plan(resolved: list[str], run_dir: Path) -> list[tuple[str, Path, str, str]]:
    """(method, source folder, paper folder, required file) of the copies made after all methods."""
    if not paper_collection_enabled(resolved):
        return []
    return [
        (method, run_dir / OUTPUT_SUBDIRECTORIES[method] / source, target, required)
        for method, source, target, required in PAPER_COPIES
        if method in resolved
    ]


def copy_paper_outputs(
    resolved: list[str], run_dir: Path, allow_diagnostic: bool = False
) -> list[dict[str, str]]:
    """Copy the fig01 folder (structures, .pml, panel PNGs) into A/paper/fig01_pymol/.

    When export_pymol rendered the figure (render_info.json names a combined PNG), both
    combined variants (FIG01_PNGS) must exist and go to A/paper/ only.

    A structures preview rendered with geometry warnings (``*_diagnostic.png``) is a
    failure unless allow_diagnostic (the registry then flags fig01 as a diagnostic draft).
    """
    paper_dir = run_dir / PAPER_DIRECTORY
    paper_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for method, source_dir, target_dir, required in paper_copy_plan(resolved, run_dir):
        if method == "structures":
            diagnostic = run_dir / OUTPUT_SUBDIRECTORIES[method] / PREVIEW_DIAGNOSTIC
            if diagnostic.is_file() and not allow_diagnostic:
                raise RuntimeError(
                    f"Paper collection: structures preview has geometry warnings ({diagnostic}); "
                    "set methods.structures.allow_geometry_warnings for a diagnostic draft"
                )
        if not (source_dir / required).is_file():
            raise RuntimeError(f"Paper collection: '{method}' wrote no {source_dir / required}")
        promoted = FIG01_PNGS if method == "structures" else ()
        (paper_dir / target_dir).mkdir(parents=True, exist_ok=True)
        for source in sorted(path for path in source_dir.iterdir() if path.is_file()):
            if source.name in promoted:
                continue
            target = f"{target_dir}/{source.name}"
            shutil.copy2(source, paper_dir / target)
            copied.append({"method": method, "source": str(source), "file": target})
        if method == "structures" and fig01_render_info(source_dir).get("combined"):
            for name in FIG01_PNGS:
                source = source_dir / name
                if not source.is_file():
                    raise RuntimeError(f"Paper collection: export_pymol rendered fig01 but wrote no {source}")
                shutil.copy2(source, paper_dir / name)
                copied.append({"method": method, "source": str(source), "file": name})
    return copied


def fig01_render_info(directory: Path) -> dict[str, Any]:
    """render_info.json of export_pymol ({} when absent or unreadable)."""
    path = directory / FIG01_RENDER_INFO
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def structures_preview_is_diagnostic(run_dir: Path) -> bool:
    return (run_dir / OUTPUT_SUBDIRECTORIES["structures"] / PREVIEW_DIAGNOSTIC).is_file()


def paper_item(name: str) -> str:
    if name.startswith(FIG01_PYMOL_DIRECTORY + "/"):
        return "fig01_adsorption_structures"
    for item in PAPER_ITEMS:
        if name == item or name.startswith(item + "_") or name.startswith(item + "."):
            return item
    return Path(name).stem


def paper_item_inputs(
    item: str,
    run_dir: Path,
    scientific_results: Path,
    material: str,
    inputs_by_method: dict[str, list[Path]],
) -> list[Path]:
    """Files an item was drawn from: run inputs and intermediate analysis tables."""
    run = scientific_results
    science = {method: run_dir / OUTPUT_SUBDIRECTORIES[method] for method in OUTPUT_SUBDIRECTORIES}
    if item == "fig01_adsorption_structures":
        paths = [run / "dft_summary.csv",
                 *(run / "dft" / material / site / f"{material}_{site}_optimized.xyz" for site in STRUCTURE_SITES)]
    elif item == "fig03_dft_descriptors":
        paths = [run / "dft_summary.csv"]
    elif item == "fig04_tddft_spectra":
        paths = [run / "tddft_summary.csv", *sorted(run.glob(f"tddft/{material}/*/transitions.csv"))]
    elif item == "fig05_co2rr_pathway":
        paths = [science["co2rr_corrections"] / CORRECTED_SUMMARY]
    elif item == "fig06_md_distance":
        paths = [
            science["md_trajectory"] / MD_SUMMARY,
            *sorted(run.glob(f"md/{material}/*/md_frame_analysis.csv")),
        ]
    elif item == "fig07_neb_profiles":
        paths = [science["neb"] / "neb_image_energies.csv", science["neb"] / "neb_barriers.csv"]
    elif item == "table02_site_summary":
        paths = [
            run / "dft_summary.csv",
            science["co2rr_corrections"] / CORRECTED_SUMMARY,
            science["neb"] / "neb_barriers.csv",
            science["md_trajectory"] / MD_SUMMARY,
            run / "tddft_summary.csv",
        ]
    elif item in PAPER_ITEMS:
        paths = list(inputs_by_method.get(PAPER_ITEMS[item][1], []))
    else:
        paths = []
    return [path for path in dict.fromkeys(paths) if path.is_file()]


def write_figure_registry(
    run_dir: Path,
    scientific_results: Path,
    material: str,
    inputs_by_method: dict[str, list[Path]],
    copied: list[dict[str, str]],
    expected: dict[str, str] | None = None,
) -> Path:
    """A/paper/figure_registry.csv: one row per file in A/paper.

    Expected files of paper_figures / paper_tables that are missing (a figure the
    producer skipped) get a row with an empty sha256 and a note. The two rendered fig01
    PNGs are noted with the renderer (render_info.json); when export_pymol rendered
    nothing (render: none), each gets a row with an empty sha256 noting that it is made
    in PyMOL from the copied script.
    """
    paper_dir = run_dir / PAPER_DIRECTORY
    sources = {entry["file"]: Path(entry["source"]) for entry in copied}
    registry = paper_dir / REGISTRY_NAME
    rows = []
    for path in sorted(p for p in paper_dir.rglob("*") if p.is_file() and p != registry):
        name = str(path.relative_to(paper_dir))
        item = paper_item(name)
        producer = PAPER_ITEMS.get(item, ("unknown", ""))[0]
        inputs = paper_item_inputs(item, run_dir, scientific_results, material, inputs_by_method)
        if name in sources:
            inputs = [sources[name], *inputs]
        rows.append({
            "item": item,
            "file": name,
            "producer_module": producer,
            "inputs": ";".join(display_path(p) for p in inputs),
            "sha256": file_sha256(path),
            "note": DIAGNOSTIC_NOTE if "_diagnostic" in name else "",
        })
        if name in FIG01_PNGS:
            renderer = str(fig01_render_info(paper_dir / FIG01_PYMOL_DIRECTORY).get("renderer", ""))
            note = FIG01_RENDER_NOTES.get(renderer, f"rendered by export_pymol (renderer: {renderer or 'unknown'})")
            if structures_preview_is_diagnostic(run_dir):
                note += "; " + DIAGNOSTIC_NOTE
            rows[-1]["note"] = note
    pml = paper_dir / FIG01_PYMOL_DIRECTORY / FIG01_PML
    if pml.is_file():
        note = FIG01_NOTE
        if structures_preview_is_diagnostic(run_dir):
            note += "; " + DIAGNOSTIC_NOTE
        for name in FIG01_PNGS:
            if (paper_dir / name).is_file():
                continue
            rows.append({
                "item": "fig01_adsorption_structures",
                "file": name,
                "producer_module": "PyMOL",
                "inputs": display_path(pml),
                "sha256": "",
                "note": note,
            })
    present = {row["file"] for row in rows}
    for name, method in sorted((expected or {}).items()):
        if name not in present:
            item = paper_item(name)
            rows.append({
                "item": item,
                "file": name,
                "producer_module": PAPER_ITEMS.get(item, ("unknown", ""))[0],
                "inputs": "",
                "sha256": "",
                "note": SKIPPED_NOTE.format(method=method),
            })
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return registry


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_methods:
        print("\n".join([*METHOD_ORDER, *ALIASES, "all"]))
        return

    config = load_config(args.config)
    selected = normalize_methods(args.methods)
    resolved = resolve_dependencies(selected, config)
    input_config = require_mapping(config, "input")
    scientific_results = resolve_project_path(
        args.scientific_results or Path(str(input_config.get("scientific_results", "results")))
    )
    model_run = resolve_project_path(
        args.model_run or Path(str(input_config.get("model_run", "results/models")))
    )
    if any(method != "model_analysis" for method in resolved) and not scientific_results.is_dir():
        fail(f"Scientific results directory does not exist: {scientific_results}")
    if "model_analysis" in resolved and not model_run.is_dir():
        fail(f"Model run directory does not exist: {model_run}")

    output_config = require_mapping(config, "output")
    output_root = resolve_project_path(Path(str(output_config.get("root", "results/analyses"))))
    protected_root = (PROJECT_ROOT / "results").resolve()
    if not output_root.is_relative_to(protected_root):
        fail(f"Analysis output must stay below the project results directory: {protected_root}")
    run_id = validate_run_id(args.run_id or default_run_id(output_root))
    run_dir = (output_root / run_id).resolve()

    inputs_by_method = {
        method: collect_inputs(method, config, scientific_results, model_run)
        for method in resolved
    }
    commands_by_method = {
        method: build_commands(method, config, scientific_results, model_run, run_dir)
        for method in resolved
    }
    copy_plan = paper_copy_plan(resolved, run_dir)

    print(f"[plan] config: {config['_config_path']}")
    print(f"[plan] run ID: {run_id}")
    print(f"[plan] selected methods: {', '.join(selected)}")
    print(f"[plan] resolved methods: {', '.join(resolved)}")
    print(f"[plan] scientific results: {scientific_results}")
    if "model_analysis" in resolved:
        print(f"[plan] model run: {model_run}")
    print(f"[plan] output: {run_dir}")
    for method in resolved:
        for command in commands_by_method[method]:
            print(f"[plan:{method}] {shlex.join(command)}")
        print(f"[plan:{method}] input files: {len(inputs_by_method[method])}")
    if paper_collection_enabled(resolved):
        for _method, source, target, _required in copy_plan:
            print(f"[plan:paper] copy {source}/ -> {run_dir / PAPER_DIRECTORY / target}/")
        for method, source, _target, _required in copy_plan:
            if method == "structures" and structure_renderer(config) != "none":
                for name in FIG01_PNGS:
                    print(f"[plan:paper] copy {source / name} -> {run_dir / PAPER_DIRECTORY / name} "
                          f"(export_pymol --render {structure_renderer(config)})")
        print(f"[plan:paper] registry: {run_dir / PAPER_DIRECTORY / REGISTRY_NAME}")

    if args.dry_run:
        print("[dry-run] Validation completed; no analysis files were created.")
        return
    if run_dir.exists():
        fail(f"Analysis run directory already exists; choose another --run-id: {run_dir}")

    (run_dir / "logs").mkdir(parents=True)
    config_path = Path(config["_config_path"])
    shutil.copy2(config_path, run_dir / "config.yaml")
    all_inputs = list(dict.fromkeys(path for paths in inputs_by_method.values() for path in paths))
    execution = require_mapping(config, "execution")
    environment_name = active_environment(str(execution.get("environment", "qml-zno-modeling")))
    manifest_path = run_dir / "run_manifest.json"
    uses_simulation_run = any(method != "model_analysis" for method in resolved)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        # inputs, outputs, input_files and files: relative to the project root when below
        # it (portable between machines); commands keep the absolute argv that was run.
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path),
        "config_snapshot": str(run_dir / "config.yaml"),
        "config_sha256": file_sha256(config_path),
        "environment": environment_name,
        "versions": runtime_versions(),
        "git": git_state(),
        "scientific_results": str(scientific_results),
        "simulation_run_manifest": (
            simulation_run_manifest(scientific_results) if uses_simulation_run else None
        ),
        "model_run": str(model_run),
        "selected_methods": selected,
        "resolved_methods": resolved,
        "inputs": inventory(all_inputs),
        "methods": {},
    }
    write_manifest(manifest_path, manifest)

    try:
        for method in resolved:
            record: dict[str, Any] = {
                "status": "running",
                "started_at": utc_now(),
                "commands": commands_by_method[method],
                "input_files": [display_path(path) for path in inputs_by_method[method]],
            }
            manifest["methods"][method] = record
            write_manifest(manifest_path, manifest)
            method_root = run_dir / OUTPUT_SUBDIRECTORIES[method]
            before = snapshot(method_root)
            try:
                for command in commands_by_method[method]:
                    run_command(command, run_dir / "logs" / f"{method}.log")
                outputs = new_or_changed(method_root, before)
                if not outputs:
                    raise RuntimeError(f"Method '{method}' produced no files.")
                record["outputs"] = inventory(outputs)
                if method == "md_trajectory":
                    # Per-frame tables stay in the simulation run (the dataset builder reads them).
                    frames = sorted(scientific_results.glob(
                        f"md/{material_name(config)}/*/md_frame_analysis.csv"
                    ))
                    record["run_outputs"] = inventory(frames)
            except BaseException as exc:
                record["status"] = "failed"
                record["failed_at"] = utc_now()
                record["error"] = f"{type(exc).__name__}: {exc}"
                write_manifest(manifest_path, manifest)
                raise
            record["status"] = "completed"
            record["completed_at"] = utc_now()
            write_manifest(manifest_path, manifest)

        if paper_collection_enabled(resolved):
            paper: dict[str, Any] = {"status": "running", "started_at": utc_now()}
            manifest["paper"] = paper
            write_manifest(manifest_path, manifest)
            try:
                copied = copy_paper_outputs(
                    resolved, run_dir,
                    "structures" in resolved and allow_geometry_warnings(config),
                )
                registry = write_figure_registry(
                    run_dir, scientific_results, material_name(config), inputs_by_method, copied,
                    expected_paper_files(config, resolved),
                )
            except BaseException as exc:
                paper["status"] = "failed"
                paper["error"] = f"{type(exc).__name__}: {exc}"
                write_manifest(manifest_path, manifest)
                raise
            paper.update({
                "status": "completed",
                "completed_at": utc_now(),
                "copied": copied,
                "registry": {"path": display_path(registry), "sha256": file_sha256(registry)},
                "files": inventory(sorted(p for p in registry.parent.rglob("*") if p.is_file())),
                "missing": sorted(
                    name for name in expected_paper_files(config, resolved)
                    if not (registry.parent / name).is_file()
                ),
            })
            write_manifest(manifest_path, manifest)
            print(f"[paper] {registry}")

        manifest["status"] = "completed"
        write_manifest(manifest_path, manifest)
        print(f"[done] Analysis workflow completed: {run_dir}")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
