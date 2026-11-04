"""Static and dry-run checks of the notebooks in notebooks/.

The notebooks are a secondary interface to the Python workflow: they must stay valid
nbformat 4 JSON without stored outputs, do nothing while every RUN_* flag is False,
and only call modules (and CLI options) that exist under scripts/.
"""
import ast
import json
import os
import re
import subprocess
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "simulation": ROOT / "notebooks" / "zno-co2-lucas.ipynb",
    "analysis": ROOT / "notebooks" / "zno-co2-analysis.ipynb",
}
FLAG = re.compile(r"^(RUN_[A-Z0-9_]+)\s*=\s*(True|False)\b", re.MULTILINE)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_sources(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def string_value(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def invoked_modules(sources: list[str]) -> dict[str, set[str]]:
    """Modules run with `-m` (in a literal list or through run_post) and their options."""
    modules: dict[str, set[str]] = {}
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            items: list[ast.AST] = []
            if isinstance(node, ast.List):
                items = node.elts
                strings = [string_value(item) for item in items]
                for index, value in enumerate(strings[:-1]):
                    if value == "-m" and strings[index + 1]:
                        options = {s for s in strings[index + 2:] if s and s.startswith("--")}
                        modules.setdefault(strings[index + 1], set()).update(options)
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_post":
                strings = [string_value(arg) for arg in node.args]
                if strings and strings[0]:
                    options = {s for s in strings[1:] if s and s.startswith("--")}
                    modules.setdefault(strings[0], set()).update(options)
    return modules


@lru_cache(maxsize=None)
def help_text(module: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if result.returncode != 0:
        raise AssertionError(f"{module} --help failed:\n{result.stderr}")
    return result.stdout


class NotebookFormatTest(unittest.TestCase):
    def test_valid_nbformat4_without_outputs(self):
        for name, path in NOTEBOOKS.items():
            with self.subTest(notebook=name):
                notebook = load(path)
                self.assertEqual(notebook["nbformat"], 4)
                self.assertIn("kernelspec", notebook["metadata"])
                ids = [cell.get("id") for cell in notebook["cells"]]
                self.assertTrue(all(ids), "every cell needs an id")
                self.assertEqual(len(ids), len(set(ids)), "cell ids must be unique")
                for cell in notebook["cells"]:
                    self.assertIn(cell["cell_type"], {"code", "markdown"})
                    self.assertIsInstance(cell["source"], (list, str))
                    if cell["cell_type"] == "code":
                        self.assertEqual(cell["outputs"], [], f"stored outputs in cell {cell['id']}")
                        self.assertIsNone(cell["execution_count"])

    def test_code_cells_parse(self):
        for name, path in NOTEBOOKS.items():
            for index, source in enumerate(code_sources(load(path))):
                with self.subTest(notebook=name, cell=index):
                    ast.parse(source)

    def test_run_flags_default_false(self):
        for name, path in NOTEBOOKS.items():
            with self.subTest(notebook=name):
                flags = FLAG.findall("\n".join(code_sources(load(path))))
                self.assertTrue(flags, "no RUN_* flags found")
                self.assertEqual({value for _, value in flags}, {"False"}, flags)


class NotebookModulesTest(unittest.TestCase):
    def test_invoked_modules_exist(self):
        for name, path in NOTEBOOKS.items():
            modules = invoked_modules(code_sources(load(path)))
            self.assertTrue(modules, f"{name}: no `-m <module>` invocation found")
            for module in modules:
                with self.subTest(notebook=name, module=module):
                    self.assertTrue(module.startswith("scripts."), module)
                    self.assertTrue((ROOT / (module.replace(".", "/") + ".py")).is_file(), module)

    def test_analysis_options_are_accepted(self):
        modules = invoked_modules(code_sources(load(NOTEBOOKS["analysis"])))
        for module, options in modules.items():
            text = help_text(module)
            accepted = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", text))
            for option in options:
                with self.subTest(module=module, option=option):
                    self.assertIn(option, accepted)


class AnalysisNotebookDryRunTest(unittest.TestCase):
    def test_all_flags_false_runs_nothing(self):
        notebook = load(NOTEBOOKS["analysis"])
        analyses = ROOT / "results" / "analyses"
        before = sorted(analyses.iterdir()) if analyses.is_dir() else []
        for directory in (ROOT, ROOT / "notebooks"):
            with self.subTest(cwd=directory):
                previous = Path.cwd()
                namespace: dict = {}
                try:
                    os.chdir(directory)
                    with mock.patch("subprocess.run") as run, mock.patch("subprocess.Popen") as popen, \
                            mock.patch("builtins.print"):
                        for source in code_sources(notebook):
                            exec(compile(source, "<notebook>", "exec"), namespace)
                finally:
                    os.chdir(previous)
                run.assert_not_called()
                popen.assert_not_called()
                self.assertEqual(namespace["LUCAS_DIR"], ROOT)
                self.assertEqual(namespace["NOTEBOOK_DIR"], ROOT / "notebooks")
                self.assertEqual(namespace["RUN_DIR"], ROOT / "results" / "simulations" / namespace["RUN_ID"])
                self.assertEqual(namespace["POST_ENV"]["QML_ZNO_RESULTS_DIR"], str(namespace["RUN_DIR"]))
        self.assertEqual(sorted(analyses.iterdir()) if analyses.is_dir() else [], before)


class AnalysisNotebookWorkflowTest(unittest.TestCase):
    """The analysis cells run the launcher's command, and only md_frame_analysis.csv lands in RUN."""

    def notebook_commands(self) -> tuple[list[list[str]], dict]:
        """Every command the cells would run with all RUN_* flags True (nothing executed)."""
        sources = code_sources(load(NOTEBOOKS["analysis"]))
        namespace: dict = {}
        commands: list[list[str]] = []
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with mock.patch("builtins.print"):
                for index, source in enumerate(sources):
                    if index > 1:  # after the helpers and the configuration cell
                        namespace["run_cmd"] = lambda args, env=None: commands.append(list(args))
                    if index == 1:
                        source = FLAG.sub(lambda m: f"{m.group(1)} = True", source)
                    exec(compile(source, "<notebook>", "exec"), namespace)
        finally:
            os.chdir(previous)
        return commands, namespace

    def test_cells_call_run_analysis_like_the_launcher(self):
        commands, namespace = self.notebook_commands()
        run_dir = str(namespace["RUN_DIR"])
        config = str(ROOT / "configs" / "analysis" / "zno_current.yaml")
        self.assertEqual(commands[0][:3], [sys.executable, "-m", "scripts.workflows.run_status"])
        analyses = commands[1:]
        self.assertEqual([command[3] for command in analyses], ["co2_reduction", "neb", "md_trajectory", "paper"])
        ids = set()
        for command in analyses:
            with self.subTest(method=command[3]):
                self.assertEqual(command[:3], [sys.executable, "-m", "scripts.workflows.run_analysis"])
                self.assertEqual(command[4:9], ["--config", config, "--scientific-results", run_dir, "--run-id"])
                self.assertEqual(len(command), 10)
                ids.add(command[9])
        self.assertEqual(len(ids), len(analyses), "each section needs its own analysis run ID")
        code = "\n".join(code_sources(load(NOTEBOOKS["analysis"])))
        for stale in ("render_publication_figures", "science_figures", "zno_paper.yaml",
                      "scripts.analysis.", "scripts.visualization."):
            self.assertNotIn(stale, code)

    def test_notebook_commands_leave_the_simulation_run_unchanged(self):
        """Run the notebook argv through run_analysis.main (fake producers) on a fake run."""
        import tempfile

        import importlib.util

        from scripts.workflows import run_analysis

        # The fake producers of the run_analysis tests (loaded by path: tests/ is not a package).
        spec = importlib.util.spec_from_file_location(
            "run_analysis_paper_tests", Path(__file__).resolve().parent / "test_run_analysis_paper.py")
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        fake_run_command, touch = helpers.fake_run_command, helpers.touch

        commands, _namespace = self.notebook_commands()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "results" / "simulations" / "fake_run"
            for name in ("dft_summary.csv", "tddft_summary.csv", "co2rr_pathway_summary.csv", "config.yaml",
                         "run_manifest.json"):
                touch(run / name)
            touch(run / "pathways/ZnO/top_metal/che/cooh/energy.json", "{}\n")
            touch(run / "tddft/ZnO/top_metal/transitions.csv")
            touch(run / "neb/ZnO/top_metal_to_bridge/neb_run_summary.json", "{}\n")
            touch(run / "md/ZnO/top_metal/seed_1/md_production.traj")
            for site in ("top_metal", "top_oxygen", "bridge"):
                touch(run / f"dft/ZnO/{site}/ZnO_{site}_optimized.xyz")
            touch(run / "data/geometries/ZnO/metadata.json", "{}\n")
            touch(run / "data/geometries/ZnO/clean_slab.traj")
            before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
            with mock.patch.object(run_analysis, "PROJECT_ROOT", root), \
                    mock.patch.object(run_analysis, "run_command", fake_run_command), \
                    mock.patch.object(run_analysis, "active_environment", lambda default: "test-env"), \
                    mock.patch("builtins.print"):
                for command in commands[1:]:
                    argv = [*command[3:]]
                    argv[argv.index("--scientific-results") + 1] = str(run)
                    run_analysis.main(argv)
            after = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
            added = sorted(str(path.relative_to(run)) for path in set(after) - set(before))
            self.assertEqual(added, ["md/ZnO/top_metal/md_frame_analysis.csv"])
            self.assertEqual({path: after[path] for path in before}, before)
            analyses = sorted(path.name for path in (root / "results" / "analyses").iterdir())
            self.assertEqual(analyses, sorted(command[9] for command in commands[1:]))


if __name__ == "__main__":
    unittest.main()
