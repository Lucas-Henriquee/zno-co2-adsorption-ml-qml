"""Read-only checks for organization and preservation of original experiments."""

import ast
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import yaml

from scripts.workflows import run_simulations as simulations

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "4a2ff4a"
# build_geometries.py is deliberately excluded: its ZnO geometry was corrected
# (Wyckoff 2b basis, top-plane bridge pair, 2x2 cell); see tests/test_geometry.py.
SOURCES = {
    "run_dft.py": "scripts/simulations/dft/run_dft.py",
    "run_tddft.py": "scripts/simulations/tddft/run_tddft.py",
    "run_md.py": "scripts/simulations/md/run_md.py",
    "run_neb.py": "scripts/simulations/neb/run_neb.py",
    "run_co2rr_pathways.py": "scripts/simulations/co2rr/run_co2rr_pathways.py",
    "classical_svm_baseline.py": "scripts/models/ml/classical_svm_baseline.py",
    "qsvm_baseline.py": "scripts/models/qml/qsvm_baseline.py",
    "qsvm_ibm_hardware.py": "scripts/models/qml_ibm/qsvm_ibm_hardware.py",
    "run_benchmarks.py": "scripts/models/benchmarks/run_benchmarks.py",
}


def relocated_source(name, source):
    source = source.replace("from utils.timer import", "from scripts.common.timer import")
    source = source.replace("from project_config import", "from scripts.common.project_config import")
    source = source.replace("from run_dft import", "from scripts.simulations.dft.run_dft import")
    if name == "run_benchmarks.py":
        source = source.replace("script_dir = Path(__file__).resolve().parent",
                                "script_dir = Path(__file__).resolve().parent.parent")
        for filename, folder in [("classical_svm_baseline.py", "ml"),
                                 ("qsvm_baseline.py", "qml"), ("qsvm_ibm_hardware.py", "qml_ibm")]:
            source = source.replace(f'script_dir / "{filename}"', f'script_dir / "{folder}" / "{filename}"')
    return source


def scientific_ast(name, source):
    """Check scientific code independently of output-name construction."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "make_run_dir":
            node.body = [ast.Pass()]
        if name == "run_benchmarks.py" and isinstance(node, ast.FunctionDef) and node.name == "main":
            body = []
            for statement in node.body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                    if isinstance(target, ast.Name) and target.id == "index":
                        continue
                    if isinstance(target, ast.Name) and target.id == "suite_dir":
                        statement.value = ast.Constant("output_directory")
                if isinstance(statement, ast.While) and "args.output_root" in ast.unparse(statement.test):
                    continue
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    call = statement.value
                    if isinstance(call.func, ast.Attribute) and ast.unparse(call.func) == "suite_dir.mkdir":
                        call.keywords = [kw for kw in call.keywords if kw.arg != "exist_ok"]
                body.append(statement)
            node.body = body
    return ast.dump(tree)


class OrganizationTests(unittest.TestCase):
    def test_original_scientific_sources_preserved(self):
        for original, current in SOURCES.items():
            with self.subTest(source=original):
                # This path addresses the historical Git tree, not the filesystem.
                baseline = subprocess.check_output(
                    ["git", "show", f"{BASELINE}:qml-co2-splitting-mo/{original}"], cwd=ROOT, text=True)
                expected = scientific_ast(original, relocated_source(original, baseline))
                actual = scientific_ast(original, (ROOT / current).read_text())
                self.assertEqual(actual, expected)

    def test_python_syntax(self):
        for path in (ROOT / "scripts").rglob("*.py"):
            with self.subTest(path=path):
                ast.parse(path.read_text(), filename=str(path))

    def test_shell_syntax(self):
        for path in [*ROOT.glob("*.sh"), *(ROOT / "scripts").rglob("*.sh"), *(ROOT / "notebooks").glob("*.sh")]:
            with self.subTest(path=path):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_notebook_syntax(self):
        notebook = ROOT / "notebooks/zno-co2-lucas.ipynb"
        for cell in json.loads(notebook.read_text())["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_notebook_resolves_root_from_both_locations(self):
        notebook = json.loads((ROOT / "notebooks/zno-co2-lucas.ipynb").read_text())
        initialization = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        previous = Path.cwd()
        try:
            for directory in (ROOT, ROOT / "notebooks"):
                with self.subTest(directory=directory), redirect_stdout(io.StringIO()):
                    os.chdir(directory)
                    namespace = {}
                    exec("".join(initialization["source"]), namespace)
                    self.assertEqual(namespace["LUCAS_DIR"], ROOT)
                    self.assertEqual(namespace["NOTEBOOK_DIR"], ROOT / "notebooks")
        finally:
            os.chdir(previous)

    def test_flattened_repository_paths(self):
        self.assertFalse((ROOT / "undergrads").exists())
        self.assertFalse((ROOT / "qml-co2-splitting-mo").exists())
        for module_name in ("run_simulations", "run_datasets", "run_models", "run_analysis"):
            source = ROOT / "scripts/workflows" / (module_name + ".py")
            tree = ast.parse(source.read_text())
            assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == "PROJECT_ROOT"
                                      for target in node.targets))
            expression = compile(ast.Expression(assignment.value), str(source), "eval")
            self.assertEqual(eval(expression, {"Path": Path, "__file__": str(source)}), ROOT)
        from scripts.common import project_config
        self.assertEqual(project_config.ROOT_DIR, ROOT)

    def test_launcher_arguments_match_restored_scripts(self):
        config = simulations.load_config(ROOT / "configs/simulations/zno_production.yaml")
        for method in simulations.METHOD_ORDER:
            module = importlib.import_module(simulations.MODULES[method])
            for command in simulations.build_commands(method, config, ["ZnO"], ["top_metal", "bridge"], 1):
                with self.subTest(method=method), patch.object(sys, "argv", [command[2], *command[3:]]):
                    module.parse_args()

    def test_yaml_preserves_notebook_numeric_parameters(self):
        notebook = json.loads((ROOT / "notebooks/zno-co2-lucas.ipynb").read_text())
        values = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    try:
                        values[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        config = yaml.safe_load((ROOT / "configs/simulations/zno_production.yaml").read_text())
        mapping = {
            "dft": {"max_steps": "DFT_STEPS", "ecut_ev": "DFT_ECUT", "kpoints": "DFT_KPTS", "fmax_ev_per_angstrom": "DFT_FMAX"},
            "co2rr": {"max_steps": "PATHWAY_STEPS", "fmax_ev_per_angstrom": "PATHWAY_FMAX"},
            "tddft": {"max_transitions": "TDDFT_MAX_TRANSITIONS", "oscillator_strength_threshold": "TDDFT_OSC_THRESHOLD"},
            "md": {"steps": "MD_STEPS", "ecut_ev": "MD_ECUT", "kpoints": "MD_KPTS", "log_interval": "MD_LOG_INTERVAL", "timestep_fs": "MD_TIMESTEP_FS", "temperature_kelvin": "MD_TEMPERATURE_K"},
            "neb": {"max_steps": "NEB_STEPS", "ecut_ev": "NEB_ECUT", "kpoints": "NEB_KPTS", "internal_images": "NEB_IMAGES", "fmax_ev_per_angstrom": "NEB_FMAX", "interpolation": "NEB_INTERPOLATION", "climb": "NEB_CLIMB", "parallel_images": "NEB_PARALLEL_IMAGES", "optimizer": "NEB_OPTIMIZER"},
        }
        for method, fields in mapping.items():
            for field, variable in fields.items():
                expected = values[variable]
                if isinstance(expected, tuple):
                    expected = list(expected)
                self.assertEqual(config["methods"][method][field], expected, variable)
        self.assertEqual(config["execution"]["mpi"]["processes"], values["MPI_CORES"])
        self.assertNotIn("random_seed", config["methods"]["md"])
        self.assertEqual(config["execution"]["mpi"]["extra_args"], ["--mca", "btl", "self,vader"])


if __name__ == "__main__":
    unittest.main()
