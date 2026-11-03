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


if __name__ == "__main__":
    unittest.main()
