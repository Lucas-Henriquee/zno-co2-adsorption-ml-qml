"""Environment files cover what the code imports, and parse.

Every third-party module imported by scripts/ (except scripts/legacy), tests/ and the
notebooks (except the optional PyMOL renderer) must be provided by at least one environment YAML; the pip requirements
must cover the GPAW-free subset; every shell script must pass ``bash -n``.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ROOT / "environments"

# Import name -> distribution name, where they differ.
IMPORT_TO_PACKAGE = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "qiskit_aer": "qiskit-aer",
    "qiskit_machine_learning": "qiskit-machine-learning",
    "qiskit_ibm_runtime": "qiskit-ibm-runtime",
    "dftd3": "dftd3-python",
}
# Imported directly but installed as a dependency of a listed package.
TRANSITIVE = {
    "urllib3": "qiskit-ibm-runtime",
    "packaging": "matplotlib",  # used by this test only
}
# Needs GPAW, which pip cannot provide with MPI: only the simulation environment.
# Simulation environment only: gpaw, and dftd3 (simple-dftd3, imported by run_dft
# only for methods.dft.dispersion: d3bj).
GPAW_ONLY = {"gpaw", "dftd3"}
# Optional, not in any environment: export_pymol renders fig01 with PyMOL
# (pymol-open-source) when importable, else with its ASE/matplotlib fallback.
OPTIONAL = {"pymol", "pymol2"}
FIRST_PARTY = {"scripts", "tests"}
STDLIB = set(sys.stdlib_module_names) | {"__future__"}
NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def imports_in_source(source: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return {name for name in found if name not in STDLIB and name not in FIRST_PARTY and name not in OPTIONAL}


def python_files(*directories: str) -> list[Path]:
    files = []
    for directory in directories:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if "legacy" not in path.relative_to(ROOT).parts:
                files.append(path)
    return files


def imports_in(*directories: str) -> dict[str, set[str]]:
    """module -> files importing it."""
    result: dict[str, set[str]] = {}
    for path in python_files(*directories):
        for name in imports_in_source(path.read_text(encoding="utf-8")):
            result.setdefault(name, set()).add(str(path.relative_to(ROOT)))
    return result


def notebook_imports() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            # Drop IPython magics and shell escapes, which are not Python.
            source = "\n".join(
                line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
            )
            for name in imports_in_source(source):
                result.setdefault(name, set()).add(str(path.relative_to(ROOT)))
    return result


def package_for(module: str) -> str:
    if module in TRANSITIVE:
        return normalize(TRANSITIVE[module])
    return normalize(IMPORT_TO_PACKAGE.get(module, module))


def load_environment(name: str) -> dict:
    with (ENVIRONMENTS / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def environment_packages(name: str) -> dict[str, str]:
    """package -> spec ("conda" entries and the pip section)."""
    packages = {}
    for entry in load_environment(name)["dependencies"]:
        entries = entry["pip"] if isinstance(entry, dict) else [entry]
        for item in entries:
            match = NAME_PATTERN.match(str(item))
            assert match, f"{name}: cannot parse dependency {item!r}"
            packages[normalize(match.group(1))] = str(item)
    return packages


def requirement_lines(path: Path, seen: set[Path] | None = None) -> list[tuple[Path, str]]:
    """(file, requirement) pairs of a requirements file, following -r includes."""
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise AssertionError(f"recursive -r include: {path}")
    seen.add(path)
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            lines.extend(requirement_lines(path.parent / line.split(None, 1)[1], seen))
            continue
        lines.append((path, line))
    return lines


def requirement_packages(filename: str) -> dict[str, list[str]]:
    from packaging.requirements import Requirement

    packages: dict[str, list[str]] = {}
    for _, line in requirement_lines(ROOT / filename):
        requirement = Requirement(line)
        packages.setdefault(normalize(requirement.name), []).append(str(requirement.specifier))
    return packages


ENVIRONMENT_FILES = ("simulation.yaml", "modeling.yaml")
REQUIREMENT_FILES = ("requirements.txt", "requirements-modeling.txt", "requirements-dev.txt")


class EnvironmentFilesTest(unittest.TestCase):
    def test_environment_files_parse(self):
        for name in ENVIRONMENT_FILES:
            with self.subTest(name=name):
                data = load_environment(name)
                self.assertTrue(data.get("name"))
                self.assertEqual(data.get("channels"), ["conda-forge", "nodefaults"])
                self.assertTrue(environment_packages(name))

    def test_requirement_files_parse(self):
        for name in REQUIREMENT_FILES:
            with self.subTest(name=name):
                self.assertTrue(requirement_packages(name))

    def test_every_import_is_in_an_environment(self):
        provided = set()
        for name in ENVIRONMENT_FILES:
            provided |= set(environment_packages(name))
        used = imports_in("scripts", "tests")
        for module, files in notebook_imports().items():
            used.setdefault(module, set()).update(files)
        missing = {
            module: sorted(files)[:3]
            for module, files in used.items()
            if package_for(module) not in provided
        }
        self.assertEqual(missing, {}, "imports without a package in environments/*.yaml")

    def test_simulation_environment_covers_simulations_and_analyses(self):
        provided = set(environment_packages("simulation.yaml"))
        used = imports_in("scripts/simulations", "scripts/analysis", "scripts/visualization",
                          "scripts/common", "scripts/workflows", "scripts/datasets")
        used.update(notebook_imports())
        missing = sorted(m for m in used if package_for(m) not in provided)
        self.assertEqual(missing, [])

    def test_modeling_environment_covers_everything_but_gpaw(self):
        provided = set(environment_packages("modeling.yaml"))
        used = set(imports_in("scripts")) | set(notebook_imports())
        missing = sorted(m for m in used - GPAW_ONLY if package_for(m) not in provided)
        self.assertEqual(missing, [])

    def test_pip_requirements_cover_the_gpaw_free_code(self):
        cases = {
            "requirements.txt": ("scripts/analysis", "scripts/visualization", "scripts/datasets",
                                 "scripts/common", "scripts/workflows"),
            "requirements-modeling.txt": ("scripts",),
        }
        for filename, directories in cases.items():
            with self.subTest(filename=filename):
                provided = set(requirement_packages(filename))
                used = set(imports_in(*directories)) - GPAW_ONLY
                missing = sorted(m for m in used if package_for(m) not in provided)
                self.assertEqual(missing, [])
        provided = set(requirement_packages("requirements-dev.txt"))
        # qiskit imports in tests are optional (skipped or guarded when absent).
        test_imports = set(imports_in("tests")) - GPAW_ONLY - {"qiskit", "qiskit_machine_learning"}
        self.assertEqual(sorted(m for m in test_imports if package_for(m) not in provided), [])

    def test_no_gpaw_in_pip_requirements(self):
        for name in REQUIREMENT_FILES:
            with self.subTest(name=name):
                self.assertNotIn("gpaw", requirement_packages(name))

    def test_pins_agree_between_conda_and_pip(self):
        from packaging.specifiers import SpecifierSet

        def conda_version(spec: str) -> str:
            # "ase=3.28.*" -> "3.28.0"; "qiskit==1.4.6" -> "1.4.6"
            version = re.split(r"==?", spec, maxsplit=1)[1].split("=")[0]
            return version.replace(".*", ".0")

        checks = (
            ("requirements.txt", ("modeling.yaml", "simulation.yaml")),
            ("requirements-modeling.txt", ("modeling.yaml",)),
        )
        for requirements, environments in checks:
            pip = requirement_packages(requirements)
            for environment in environments:
                packages = environment_packages(environment)
                for package, specifiers in pip.items():
                    if package not in packages or "=" not in packages[package]:
                        continue
                    with self.subTest(package=package, environment=environment, pip=requirements):
                        version = conda_version(packages[package])
                        for specifier in specifiers:
                            self.assertIn(version, SpecifierSet(specifier),
                                          f"{environment} {packages[package]} vs {requirements} {specifier}")

    def test_simulation_environment_uses_mpi_gpaw(self):
        packages = environment_packages("simulation.yaml")
        self.assertIn("mpi_openmpi", packages["gpaw"])
        self.assertIn("openmpi", packages)


class ShellScriptsTest(unittest.TestCase):
    def shell_scripts(self) -> list[Path]:
        return [*sorted(ROOT.glob("*.sh")), *sorted((ROOT / "scripts").rglob("*.sh")),
                *sorted((ROOT / "notebooks").glob("*.sh"))]

    def test_shell_scripts_parse(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not available")
        scripts = self.shell_scripts()
        self.assertTrue(scripts)
        for path in scripts:
            with self.subTest(script=str(path.relative_to(ROOT))):
                result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_environment_scripts_avoid_bash4_builtins(self):
        # macOS ships bash 3.2.
        for path in sorted((ROOT / "scripts" / "environment").glob("*.sh")):
            text = path.read_text(encoding="utf-8")
            for token in ("mapfile", "readarray", "declare -A", "local -A", "${!", ",,}", "^^}"):
                with self.subTest(script=path.name, token=token):
                    self.assertNotIn(token, text)

    def test_environment_scripts_show_help(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not available")
        for name in ("setup_environment.sh", "check_environment.sh"):
            with self.subTest(script=name):
                result = subprocess.run(
                    [bash, str(ROOT / "scripts" / "environment" / name), "--help"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--env", result.stdout)

    def test_launchers_show_help_without_conda(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not available")
        env = {**os.environ, "QML_ZNO_CONDA": "/nonexistent/conda"}
        for path in sorted(ROOT.glob("run_*.sh")):
            with self.subTest(launcher=path.name):
                result = subprocess.run([bash, str(path), "--help"], capture_output=True, text=True,
                                        env=env, cwd=ROOT, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"Usage: ./{path.name}", result.stdout)
                self.assertIn("python -m scripts.workflows.", result.stdout)

    def test_no_capture_output_for_conda_and_mamba1_only(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not available")
        with tempfile.TemporaryDirectory() as tmp:
            managers = {
                "conda": ("conda", "echo 'usage: conda run [--no-capture-output]'"),
                "mamba1": ("mamba", "echo '  --no-capture-output  Do not capture output'"),
                "mamba2": ("mamba", "echo 'Usage: mamba run [OPTIONS]'"),
                "micromamba": ("micromamba", "echo '--no-capture-output'"),
            }
            expected = {"conda": True, "mamba1": True, "mamba2": False, "micromamba": False}
            for label, (name, help_text) in managers.items():
                folder = Path(tmp) / label
                folder.mkdir()
                tool = folder / name
                tool.write_text(f"#!/usr/bin/env bash\n{help_text}\n", encoding="utf-8")
                tool.chmod(0o755)
                script = (f"source {ROOT / 'scripts/environment/conda_tools.sh'}; "
                          f"find_manager {tool}; set_run_prefix envs/x; echo \"${{RUN_PREFIX[*]}}\"")
                with self.subTest(manager=label):
                    result = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual("--no-capture-output" in result.stdout, expected[label], result.stdout)
                    self.assertTrue(result.stdout.strip().endswith("run --no-capture-output -p envs/x")
                                    if expected[label] else result.stdout.strip().endswith("run -p envs/x"))


if __name__ == "__main__":
    unittest.main()
