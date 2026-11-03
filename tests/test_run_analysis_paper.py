"""run_analysis: methods, dependencies, commands, dry run, paper collection and manifest.

Command execution is mocked (fake producers write placeholder files); the last test
checks the built commands against the real producers' command-line options.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.workflows import run_analysis as ra

ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = ROOT / "configs" / "analysis" / "zno_paper.yaml"
CURRENT_CONFIG = ROOT / "configs" / "analysis" / "zno_current.yaml"
SITES = ("top_metal", "top_oxygen", "bridge")
PY = sys.executable


def touch(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """Temporary project root (outputs stay out of the real results/) and one fake run."""
    root = tmp_path / "project"
    run = root / "results" / "simulations" / "fake_run"
    for name in ("dft_summary.csv", "tddft_summary.csv", "co2rr_pathway_summary.csv", "config.yaml"):
        touch(run / name, f"{name}\n")
    touch(run / "run_manifest.json", '{"run_id": "fake_run"}\n')
    touch(run / "pathways/ZnO/top_metal/che/cooh/energy.json", "{}\n")
    touch(run / "tddft/ZnO/top_metal/transitions.csv")
    touch(run / "neb/ZnO/top_metal_to_bridge/neb_run_summary.json", "{}\n")
    touch(run / "neb/ZnO/top_metal_to_bridge/neb_final_images.traj")
    touch(run / "md/ZnO/top_metal/seed_1/md_production.traj")
    for site in SITES:
        touch(run / f"dft/ZnO/{site}/ZnO_{site}_optimized.xyz")
    touch(run / "data/geometries/ZnO/metadata.json", "{}\n")
    touch(run / "data/geometries/ZnO/clean_slab.traj")
    monkeypatch.setattr(ra, "PROJECT_ROOT", root)
    return root, run


def option(command: list[str], name: str) -> Path:
    return Path(command[command.index(name) + 1])


def option_value(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def fake_run_command(command: list[str], log_path: Path) -> None:
    """Write placeholder outputs where each producer would."""
    module = command[2]
    out = option(command, "--output-dir")
    render = option_value(command, "--render") if module == "scripts.visualization.export_pymol" else ""
    files = {
        "scripts.analysis.analyze_results": ["ranked_candidates.csv"],
        "scripts.analysis.co2rr_corrections": ["co2rr_corrected_summary.csv", "co2rr_corrections_note.md"],
        "scripts.visualization.plot_co2rr_pathway": ["co2rr_pathway.png", "co2rr_pathway_profile_table.csv"],
        "scripts.analysis.analysis_co2_reduction": ["co2_reduction_table.csv"],
        "scripts.visualization.render_publication_figures": ["fig01.png"],
        "scripts.analysis.neb_energy_table": ["neb_image_energies.csv", "neb_barriers.csv"],
        "scripts.visualization.plot_neb": [
            "individual_profiles/neb_profile_top_metal_to_bridge.png", "migration_barrier_comparison.png",
        ],
        "scripts.analysis.md_trajectory_analysis": [
            "md_trajectory_analysis_summary.csv", "md/ZnO/md_nearest_site_occupancy.png",
        ],
        "scripts.visualization.render_adsorption_structures": ["zno_adsorption_sites_combined.png"],
        "scripts.visualization.export_pymol": [
            *[f"{site}.{ext}" for site in SITES for ext in ("pdb", "cif")],
            "fig01_adsorption_structures.pml", "README.txt",
            *([] if render == "none" else
              [*[f"fig01_{site}.png" for site in SITES], "fig01_adsorption_structures.png"]),
        ],
        "scripts.visualization.paper_figures": [
            f"{stem}{suffix}"
            for stem in ("fig03_dft_descriptors", "fig04_tddft_spectra", "fig05_co2rr_pathway",
                         "fig06_md_distance", "fig07_neb_profiles")
            for suffix in (".png", ".pdf", "_data.csv")
        ],
        "scripts.analysis.paper_tables": [f"table02_site_summary.{ext}" for ext in ("csv", "md", "tex")],
    }[module]
    for name in files:
        touch(out / name, f"{module} {name}\n")
    if module == "scripts.visualization.export_pymol":
        rendered = render != "none"
        touch(out / "render_info.json", json.dumps({
            "requested": render,
            "renderer": "ase" if rendered else "none",
            "combined": "fig01_adsorption_structures.png" if rendered else None,
        }))
    if module == "scripts.analysis.md_trajectory_analysis":
        touch(option(command, "--results-dir") / "md/ZnO/top_metal/md_frame_analysis.csv")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n")


def plan(project, config: Path = PAPER_CONFIG, *methods: str):
    root, run = project
    cfg = ra.load_config(config)
    resolved = ra.resolve_dependencies(ra.normalize_methods(list(methods) or ["paper"]))
    run_dir = root / "results/analyses/A"
    commands = {m: ra.build_commands(m, cfg, run, root / "results/models", run_dir) for m in resolved}
    return cfg, resolved, run_dir, commands


# ---------------------------------------------------------------- methods


def test_dependencies():
    assert ra.resolve_dependencies(["paper_tables"]) == [
        "co2rr_corrections", "neb", "md_trajectory", "paper_tables",
    ]
    assert ra.resolve_dependencies(["paper_figures"]) == [
        "co2rr_corrections", "neb", "md_trajectory", "paper_figures",
    ]
    assert ra.resolve_dependencies(["co2_reduction"]) == ["co2rr_corrections", "co2_reduction"]
    assert ra.resolve_dependencies(["science_figures"]) == [
        "co2rr_corrections", "co2_reduction", "science_figures",
    ]
    assert set(ra.DEPENDENCIES) == set(ra.METHOD_ORDER) == set(ra.OUTPUT_SUBDIRECTORIES)


def test_aliases():
    assert ra.resolve_dependencies(ra.normalize_methods(["science"])) == [
        "screening", "co2rr_corrections", "co2_reduction", "science_figures",
    ]
    assert ra.resolve_dependencies(ra.normalize_methods(["paper"])) == [
        "co2rr_corrections", "co2_reduction", "science_figures", "neb", "md_trajectory",
        "structures", "paper_figures", "paper_tables",
    ]
    assert ra.normalize_methods(["models"]) == ["model_analysis"]
    assert ra.normalize_methods(["all"]) == list(ra.METHOD_ORDER)
    with pytest.raises(SystemExit, match="Unknown analysis method"):
        ra.normalize_methods(["papers"])


def test_output_subdirectories():
    expected = {
        "co2rr_corrections": "science/co2rr", "co2_reduction": "science/co2_reduction",
        "neb": "science/neb", "md_trajectory": "science/md", "structures": "science/structures",
        "paper_figures": "paper", "paper_tables": "paper",
    }
    assert {m: str(ra.OUTPUT_SUBDIRECTORIES[m]) for m in expected} == expected


# ---------------------------------------------------------------- commands


def test_commands_exact_argv(project):
    root, run = project
    _cfg, resolved, a, commands = plan(project)
    neb = str(run / "neb/ZnO")
    assert commands["co2rr_corrections"] == [
        [PY, "-m", "scripts.analysis.co2rr_corrections",
         "--results-dir", str(run), "--output-dir", str(a / "science/co2rr")],
        [PY, "-m", "scripts.visualization.plot_co2rr_pathway",
         "--summary", str(run / "co2rr_pathway_summary.csv"),
         "--corrected", str(a / "science/co2rr/co2rr_corrected_summary.csv"),
         "--variant", "c", "--material", "ZnO", "--output-dir", str(a / "science/co2rr")],
    ]
    assert commands["co2_reduction"] == [[
        PY, "-m", "scripts.analysis.analysis_co2_reduction",
        "--results-dir", str(run), "--output-dir", str(a / "science/co2_reduction"),
        "--co2rr-variant", "c",
        "--corrected-summary", str(a / "science/co2rr/co2rr_corrected_summary.csv"),
    ]]
    assert commands["science_figures"] == [[
        PY, "-m", "scripts.visualization.render_publication_figures",
        "--analysis-dir", str(a / "science/co2_reduction"), "--tddft-dir", str(run / "tddft"),
        "--output-dir", str(a / "science/figures"), "--dpi", "600",
    ]]
    assert commands["neb"] == [
        [PY, "-m", "scripts.analysis.neb_energy_table", "--base", neb, "--output-dir", str(a / "science/neb")],
        [PY, "-m", "scripts.visualization.plot_neb", "--base", neb, "--output-dir", str(a / "science/neb"),
         "--smooth-band", "--dpi", "600"],
    ]
    assert commands["md_trajectory"] == [[
        PY, "-m", "scripts.analysis.md_trajectory_analysis",
        "--results-dir", str(run), "--material", "ZnO", "--output-dir", str(a / "science/md"),
        "--desorption-threshold", "4.50",
    ]]
    geometry = run / "data/geometries/ZnO"
    assert commands["structures"] == [[
        PY, "-m", "scripts.visualization.export_pymol",
        "--input-dir", str(run / "dft/ZnO"), "--summary", str(run / "dft_summary.csv"),
        "--material", "ZnO", "--output-dir", str(a / "science/structures/pymol"),
        "--render", "auto", "--dpi", "600",
    ], [
        PY, "-m", "scripts.visualization.render_adsorption_structures",
        "--input-dir", str(run / "dft/ZnO"), "--summary", str(run / "dft_summary.csv"),
        "--site-metadata", str(geometry / "metadata.json"),
        "--initial-slab", str(geometry / "clean_slab.traj"),
        "--output-dir", str(a / "science/structures"), "--dpi", "600",
    ]]
    assert commands["paper_figures"] == [[
        PY, "-m", "scripts.visualization.paper_figures",
        "--results-dir", str(run), "--analysis-dir", str(a), "--output-dir", str(a / "paper"),
        "--material", "ZnO", "--dpi", "600", "--tddft-sigma-ev", "0.10",
        "--desorption-threshold", "4.50", "--md-seed", "1234", "--co2rr-variant", "c",
    ]]
    assert commands["paper_tables"] == [[
        PY, "-m", "scripts.analysis.paper_tables",
        "--results-dir", str(run), "--analysis-dir", str(a), "--output-dir", str(a / "paper"),
        "--material", "ZnO", "--co2rr-variant", "c",
    ]]


def test_command_options_from_config(project):
    root, run = project
    cfg = ra.load_config(PAPER_CONFIG)
    cfg["methods"]["neb"] = {"barrier_resolution_ev": 0.05, "smooth_band": False}
    cfg["methods"]["paper_figures"]["figures"] = ["fig04"]
    cfg["methods"]["paper_figures"]["tddft_sigma_ev"] = 0.125
    a = root / "results/analyses/A"
    neb = ra.build_commands("neb", cfg, run, root, a)
    assert neb[0][-2:] == ["--barrier-resolution", "0.05"]
    assert neb[1][-2:] == ["--barrier-resolution", "0.05"]
    figures = ra.build_commands("paper_figures", cfg, run, root, a)[0]
    assert figures[-2:] == ["--figures", "fig04"]
    cfg["methods"]["md_trajectory"]["desorption_threshold_ang"] = 5.0
    md = ra.build_commands("md_trajectory", cfg, run, root, a)[0]
    figures = ra.build_commands("paper_figures", cfg, run, root, a)[0]
    assert option(md, "--desorption-threshold") == option(figures, "--desorption-threshold") == Path("5.00")
    cfg["methods"]["paper_figures"]["figures"] = ["fig02"]
    with pytest.raises(SystemExit, match="paper_figures.figures"):
        ra.build_commands("paper_figures", cfg, run, root, a)
    assert option(figures, "--tddft-sigma-ev") == Path("0.125")
    cfg["methods"]["paper_figures"]["figures"] = None
    cfg["methods"]["paper_figures"]["md_seed"] = 2345
    cfg["methods"]["co2rr_corrections"]["variant"] = "a"
    figures = ra.build_commands("paper_figures", cfg, run, root, a)[0]
    assert option(figures, "--md-seed") == Path("2345")
    assert option(figures, "--co2rr-variant") == Path("a")
    cfg["methods"]["paper_figures"]["md_seed"] = "seed1"
    with pytest.raises(SystemExit, match="md_seed"):
        ra.build_commands("paper_figures", cfg, run, root, a)
    cfg["methods"]["co2rr_corrections"]["variant"] = "d"
    with pytest.raises(SystemExit, match="variant"):
        ra.build_commands("paper_tables", cfg, run, root, a)


def test_structures_fall_back_to_repository_geometry(project):
    root, run = project
    for name in ("metadata.json", "clean_slab.traj"):
        (run / "data/geometries/ZnO" / name).unlink()
    cfg = ra.load_config(PAPER_CONFIG)
    command = ra.build_commands("structures", cfg, run, root, root / "results/analyses/A")[1]
    assert option(command, "--site-metadata") == root / "data/geometries/ZnO/metadata.json"
    with pytest.raises(SystemExit, match="Missing inputs of 'structures'"):
        ra.collect_inputs("structures", cfg, run, root)


def test_science_with_current_config_uses_variant_c(project):
    _cfg, resolved, a, commands = plan(project, CURRENT_CONFIG, "science")
    assert "co2rr_corrections" in resolved
    command = commands["co2_reduction"][0]
    assert option(command, "--co2rr-variant") == Path("c")
    assert option(command, "--corrected-summary") == a / "science/co2rr/co2rr_corrected_summary.csv"


# ---------------------------------------------------------------- inputs


def test_inputs_required_and_optional(project):
    root, run = project
    cfg = ra.load_config(PAPER_CONFIG)
    neb = ra.collect_inputs("neb", cfg, run, root)
    assert run / "neb/ZnO/top_metal_to_bridge/neb_run_summary.json" in neb
    assert run / "neb/ZnO/top_metal_to_bridge/neb_final_images.traj" in neb
    assert run / "config.yaml" in neb
    (run / "config.yaml").unlink()
    assert run / "config.yaml" not in ra.collect_inputs("neb", cfg, run, root)
    co2rr = ra.collect_inputs("co2rr_corrections", cfg, run, root)
    assert run / "pathways/ZnO/top_metal/che/cooh/energy.json" in co2rr


@pytest.mark.parametrize(
    "method, remove, message",
    [
        ("neb", "neb/ZnO/top_metal_to_bridge/neb_run_summary.json", "NEB summaries"),
        ("md_trajectory", "md/ZnO/top_metal/seed_1/md_production.traj", "MD trajectories"),
        ("co2rr_corrections", "co2rr_pathway_summary.csv", "Missing inputs of 'co2rr_corrections'"),
        ("structures", "dft/ZnO/bridge/ZnO_bridge_optimized.xyz", "ZnO_bridge_optimized.xyz"),
        ("paper_figures", "tddft/ZnO/top_metal/transitions.csv", "TDDFT transition"),
        ("paper_tables", "tddft_summary.csv", "Missing inputs of 'paper_tables'"),
    ],
)
def test_missing_inputs_fail(project, method, remove, message):
    root, run = project
    (run / remove).unlink()
    with pytest.raises(SystemExit, match=re.escape(message)):
        ra.collect_inputs(method, ra.load_config(PAPER_CONFIG), run, root)


def test_dry_run_creates_nothing(project, capsys):
    root, run = project
    ra.main(["paper", "--config", str(PAPER_CONFIG), "--scientific-results", str(run),
             "--run-id", "dry", "--dry-run"])
    out = capsys.readouterr().out
    assert not (root / "results/analyses").exists()
    assert "[plan] resolved methods: co2rr_corrections, co2_reduction, science_figures, neb, " \
           "md_trajectory, structures, paper_figures, paper_tables" in out
    assert "[plan:paper_tables] " in out and "scripts.analysis.paper_tables" in out
    assert "science/structures/pymol/ -> " in out and "paper/fig01_pymol/" in out
    assert "science/structures/pymol/fig01_adsorption_structures.png -> " in out
    assert "paper/fig01_adsorption_structures.png (export_pymol --render auto)" in out
    assert "figure_registry.csv" in out
    assert "fig05_co2rr_pathway" not in out and "fig07_neb_profiles" not in out  # drawn, not copied
    assert "[dry-run]" in out


def test_dry_run_reports_missing_input(project):
    root, run = project
    (run / "neb/ZnO/top_metal_to_bridge/neb_run_summary.json").unlink()
    with pytest.raises(SystemExit, match="NEB summaries"):
        ra.main(["paper", "--config", str(PAPER_CONFIG), "--scientific-results", str(run),
                 "--run-id", "dry", "--dry-run"])


# ---------------------------------------------------------------- execution (mocked)


def run_paper(project, monkeypatch, git=None):
    root, run = project
    monkeypatch.setattr(ra, "run_command", fake_run_command)
    monkeypatch.setattr(ra, "active_environment", lambda default: "test-env")
    if git is not None:
        monkeypatch.setattr(ra, "git_state", lambda: git)
    ra.main(["paper", "--config", str(PAPER_CONFIG), "--scientific-results", str(run), "--run-id", "P1"])
    a = root / "results/analyses/P1"
    return a, json.loads((a / "run_manifest.json").read_text())


def test_paper_collection_registry_and_manifest(project, monkeypatch):
    root, run = project
    a, manifest = run_paper(project, monkeypatch, git={"commit": "abc123", "dirty": True})
    paper = a / "paper"
    pymol = {f"fig01_pymol/{site}.{ext}" for site in SITES for ext in ("pdb", "cif")} | {
        f"fig01_pymol/fig01_{site}.png" for site in SITES} | {
        f"fig01_pymol/{name}" for name in (
            "fig01_adsorption_structures.pml", "fig01_adsorption_structures.png", "README.txt",
            "render_info.json")}
    assert {str(p.relative_to(paper)) for p in (paper / "fig01_pymol").iterdir()} == pymol
    assert (paper / "fig01_adsorption_structures.png").read_bytes() == \
        (paper / "fig01_pymol/fig01_adsorption_structures.png").read_bytes()
    for name in ("fig05_co2rr_pathway.png", "fig05_co2rr_pathway_data.csv",
                 "fig06_md_distance.pdf", "fig07_neb_profiles.png", "fig07_neb_profiles_data.csv",
                 "fig03_dft_descriptors.pdf", "table02_site_summary.tex"):
        assert (paper / name).is_file(), name
    assert (paper / "fig05_co2rr_pathway.png").read_text() == \
        "scripts.visualization.paper_figures fig05_co2rr_pathway.png\n"
    assert not list(paper.glob("fig07_neb_profiles_top_metal*")) and not list(paper.glob("*_barriers*"))

    with (paper / "figure_registry.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["file"]: row for row in csv.DictReader(handle)}
    assert list(next(iter(rows.values()))) == ["item", "file", "producer_module", "inputs", "sha256", "note"]
    fig01 = rows.pop("fig01_adsorption_structures.png")
    assert fig01["item"] == "fig01_adsorption_structures"
    assert fig01["sha256"] == sha(paper / "fig01_adsorption_structures.png")
    assert fig01["producer_module"] == "scripts.visualization.export_pymol"
    assert fig01["note"] == ra.FIG01_RENDER_NOTES["ase"]
    assert "ASE/matplotlib fallback" in fig01["note"]
    assert fig01["inputs"].split(";")[0] == \
        "results/analyses/P1/science/structures/pymol/fig01_adsorption_structures.png"
    assert all(row["note"] == "" for row in rows.values())
    assert "figure_registry.csv" not in rows
    assert set(rows) == {str(p.relative_to(paper)) for p in paper.rglob("*") if p.is_file()} - {
        "figure_registry.csv", "fig01_adsorption_structures.png"}
    assert {row["item"] for row in rows.values()} == set(ra.PAPER_ITEMS)
    for name, row in rows.items():
        assert row["sha256"] == sha(paper / name)
    for name in pymol:
        assert rows[name]["item"] == "fig01_adsorption_structures"
        assert rows[name]["producer_module"] == "scripts.visualization.export_pymol"
        assert "results/simulations/fake_run/dft/ZnO/bridge/ZnO_bridge_optimized.xyz" in \
            rows[name]["inputs"].split(";")
    assert rows["fig04_tddft_spectra_data.csv"]["producer_module"] == "scripts.visualization.paper_figures"
    for name in ("fig05_co2rr_pathway.png", "fig06_md_distance_data.csv", "fig07_neb_profiles_data.csv"):
        assert rows[name]["producer_module"] == "scripts.visualization.paper_figures"
    assert rows["table02_site_summary.md"]["producer_module"] == "scripts.analysis.paper_tables"
    assert "results/simulations/fake_run/tddft/ZnO/top_metal/transitions.csv" in \
        rows["fig04_tddft_spectra.png"]["inputs"].split(";")
    assert rows["fig05_co2rr_pathway.png"]["inputs"] == \
        "results/analyses/P1/science/co2rr/co2rr_corrected_summary.csv"
    assert "results/simulations/fake_run/md/ZnO/top_metal/md_frame_analysis.csv" in \
        rows["fig06_md_distance.png"]["inputs"]
    assert rows["fig07_neb_profiles.pdf"]["inputs"].split(";") == [
        "results/analyses/P1/science/neb/neb_image_energies.csv", "results/analyses/P1/science/neb/neb_barriers.csv",
    ]
    table_inputs = rows["table02_site_summary.csv"]["inputs"].split(";")
    assert "results/analyses/P1/science/neb/neb_barriers.csv" in table_inputs

    assert manifest["status"] == "completed"
    assert manifest["git"] == {"commit": "abc123", "dirty": True}
    assert manifest["environment"] == "test-env"
    assert manifest["simulation_run_manifest"] == {
        "path": str(run / "run_manifest.json"), "sha256": sha(run / "run_manifest.json"), "present": True,
    }
    inputs = {entry["path"]: entry["sha256"] for entry in manifest["inputs"]}
    for path in (run / "dft_summary.csv", run / "neb/ZnO/top_metal_to_bridge/neb_final_images.traj",
                 run / "md/ZnO/top_metal/seed_1/md_production.traj", run / "data/geometries/ZnO/clean_slab.traj"):
        assert inputs[str(path)] == sha(path)
    assert manifest["paper"]["status"] == "completed"
    assert manifest["paper"]["registry"]["sha256"] == sha(paper / "figure_registry.csv")
    assert sorted(entry["file"] for entry in manifest["paper"]["copied"]) == sorted(
        pymol | {"fig01_adsorption_structures.png"})
    # paper_figures and paper_tables share A/paper; each lists only its own outputs.
    figure_outputs = {Path(e["path"]).name for e in manifest["methods"]["paper_figures"]["outputs"]}
    table_outputs = {Path(e["path"]).name for e in manifest["methods"]["paper_tables"]["outputs"]}
    assert figure_outputs == {
        f"{stem}{suffix}" for stem in ra.PAPER_FIGURE_NAMES.values() for suffix in (".png", ".pdf", "_data.csv")
    }
    assert table_outputs == {"table02_site_summary.csv", "table02_site_summary.md", "table02_site_summary.tex"}
    assert [Path(e["path"]).name for e in manifest["methods"]["md_trajectory"]["run_outputs"]] == \
        ["md_frame_analysis.csv"]
    assert (a / "config.yaml").read_bytes() == PAPER_CONFIG.read_bytes()
    with pytest.raises(SystemExit, match="already exists"):
        ra.main(["paper", "--config", str(PAPER_CONFIG), "--scientific-results", str(run), "--run-id", "P1"])


def test_missing_copy_source_fails_the_run(project, monkeypatch):
    root, run = project

    def without_structures(command, log_path):
        if command[2] != "scripts.visualization.render_adsorption_structures":
            fake_run_command(command, log_path)
        else:
            touch(option(command, "--output-dir") / "zno_adsorption_sites_combined_diagnostic.png")

    monkeypatch.setattr(ra, "run_command", without_structures)
    monkeypatch.setattr(ra, "active_environment", lambda default: "test-env")
    with pytest.raises(RuntimeError, match="geometry warnings"):
        ra.main(["paper", "--config", str(PAPER_CONFIG), "--scientific-results", str(run), "--run-id", "P2"])
    manifest = json.loads((root / "results/analyses/P2/run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["paper"]["status"] == "failed"

    config = root / "diagnostic.yaml"
    config.write_text(PAPER_CONFIG.read_text().replace(
        "allow_geometry_warnings: false", "allow_geometry_warnings: true"))
    ra.main(["paper", "--config", str(config), "--scientific-results", str(run), "--run-id", "P3"])
    paper = root / "results/analyses/P3/paper"
    assert (paper / "fig01_pymol/fig01_adsorption_structures.pml").is_file()
    assert (paper / "fig01_adsorption_structures.png").is_file()
    with (paper / "figure_registry.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["file"]: row for row in csv.DictReader(handle)}
    assert rows["fig01_adsorption_structures.png"]["note"] == \
        ra.FIG01_RENDER_NOTES["ase"] + "; " + ra.DIAGNOSTIC_NOTE
    structures = json.loads((paper.parent / "run_manifest.json").read_text())["methods"]["structures"]
    assert structures["commands"][1][-1] == "--allow-geometry-warnings"


def test_fig01_without_rendering(project, monkeypatch):
    """render: none -> PyMOL files only; the fig01 PNG row is a placeholder (made in PyMOL)."""
    root, run = project
    monkeypatch.setattr(ra, "run_command", fake_run_command)
    monkeypatch.setattr(ra, "active_environment", lambda default: "test-env")
    config = root / "norender.yaml"
    config.write_text(PAPER_CONFIG.read_text().replace("render: auto", "render: none"))
    ra.main(["paper", "--config", str(config), "--scientific-results", str(run), "--run-id", "P6"])
    a = root / "results/analyses/P6"
    paper = a / "paper"
    assert not list(paper.glob("fig01*.png")) and not list((paper / "fig01_pymol").glob("*.png"))
    with (paper / "figure_registry.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["file"]: row for row in csv.DictReader(handle)}
    fig01 = rows["fig01_adsorption_structures.png"]
    assert fig01["sha256"] == "" and fig01["producer_module"] == "PyMOL"
    assert fig01["note"] == ra.FIG01_NOTE
    assert fig01["inputs"] == "results/analyses/P6/paper/fig01_pymol/fig01_adsorption_structures.pml"
    command = json.loads((a / "run_manifest.json").read_text())["methods"]["structures"]["commands"][0]
    assert command[command.index("--render") + 1] == "none"


def test_invalid_renderer(project):
    root, run = project
    cfg = ra.load_config(PAPER_CONFIG)
    cfg["methods"]["structures"]["render"] = "povray"
    with pytest.raises(SystemExit, match="methods.structures.render"):
        ra.build_commands("structures", cfg, run, root / "results/models", root / "results/analyses/A")


def test_skipped_figure_is_registered(project, monkeypatch):
    root, run = project

    def skip_fig04(command, log_path):
        fake_run_command(command, log_path)
        if command[2] == "scripts.visualization.paper_figures":
            for path in option(command, "--output-dir").glob("fig04_*"):
                path.unlink()

    monkeypatch.setattr(ra, "run_command", skip_fig04)
    monkeypatch.setattr(ra, "active_environment", lambda default: "test-env")
    ra.main(["paper", "--config", str(PAPER_CONFIG), "--scientific-results", str(run), "--run-id", "P5"])
    a = root / "results/analyses/P5"
    with (a / "paper/figure_registry.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["file"]: row for row in csv.DictReader(handle)}
    for name in ("fig04_tddft_spectra.png", "fig04_tddft_spectra.pdf", "fig04_tddft_spectra_data.csv"):
        assert rows[name]["sha256"] == "" and "skipped" in rows[name]["note"]
        assert rows[name]["item"] == "fig04_tddft_spectra"
    assert rows["fig03_dft_descriptors.png"]["sha256"]
    manifest = json.loads((a / "run_manifest.json").read_text())
    assert manifest["paper"]["missing"] == [
        "fig04_tddft_spectra.pdf", "fig04_tddft_spectra.png", "fig04_tddft_spectra_data.csv",
    ]


def test_missing_copy_source_without_warnings(project, monkeypatch):
    root, run = project

    def no_structures(command, log_path):
        if command[2] != "scripts.visualization.export_pymol":
            fake_run_command(command, log_path)
        else:
            touch(option(command, "--output-dir") / "top_metal.pdb")

    monkeypatch.setattr(ra, "run_command", no_structures)
    monkeypatch.setattr(ra, "active_environment", lambda default: "test-env")
    with pytest.raises(RuntimeError, match="fig01_adsorption_structures.pml"):
        ra.main(["paper", "--config", str(PAPER_CONFIG), "--scientific-results", str(run), "--run-id", "P4"])


def test_screening_only_run_has_no_paper_step(project, monkeypatch):
    root, run = project
    monkeypatch.setattr(ra, "run_command", fake_run_command)
    monkeypatch.setattr(ra, "active_environment", lambda default: "test-env")
    ra.main(["screening", "--config", str(CURRENT_CONFIG), "--scientific-results", str(run), "--run-id", "S1"])
    a = root / "results/analyses/S1"
    manifest = json.loads((a / "run_manifest.json").read_text())
    assert manifest["resolved_methods"] == ["screening"]
    assert "paper" not in manifest and not (a / "paper").exists()


def test_git_state_outside_git(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    state = ra.git_state()
    assert state["commit"] is None and state["dirty"] is None and "error" in state


def test_git_state_in_this_checkout():
    state = ra.git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", state["commit"])
    assert isinstance(state["dirty"], bool)


def test_simulation_manifest_missing(tmp_path):
    assert ra.simulation_run_manifest(tmp_path) == {
        "path": str(tmp_path / "run_manifest.json"), "sha256": None, "present": False,
    }


# ---------------------------------------------------------------- real producers


PRODUCERS = [
    "scripts/analysis/co2rr_corrections.py", "scripts/visualization/plot_co2rr_pathway.py",
    "scripts/analysis/analysis_co2_reduction.py", "scripts/analysis/neb_energy_table.py",
    "scripts/visualization/plot_neb.py", "scripts/analysis/md_trajectory_analysis.py",
    "scripts/visualization/render_adsorption_structures.py", "scripts/visualization/export_pymol.py",
    "scripts/visualization/paper_figures.py",
    "scripts/analysis/paper_tables.py", "scripts/visualization/render_publication_figures.py",
]


@pytest.mark.skipif(
    not all((ROOT / path).is_file() for path in PRODUCERS), reason="paper producers not present yet"
)
def test_paper_commands_match_producer_options(project):
    """Every option of the built paper commands is accepted by the producer's --help."""
    _cfg, _resolved, _a, commands = plan(project)
    for method, method_commands in commands.items():
        for command in method_commands:
            result = subprocess.run(
                [PY, "-m", command[2], "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(ROOT)},
            )
            assert result.returncode == 0, f"{command[2]} --help failed:\n{result.stderr}"
            accepted = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", result.stdout))
            for item in command[3:]:
                if item.startswith("--"):
                    assert item in accepted, f"{method}: {command[2]} does not accept {item}"
