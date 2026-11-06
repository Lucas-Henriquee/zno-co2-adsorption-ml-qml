"""PBE-D3 option (explore/pbe-d3-sites): run_dft, run_neb settings, run_simulations and guards.

simple-dftd3 is replaced by a fake ``dftd3.ase`` module; no GPAW is needed.
"""

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones

from scripts.simulations.dft import run_dft
from scripts.simulations.md import run_md
from scripts.simulations.neb import run_neb
from scripts.workflows import run_simulations as simulations

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "configs/simulations/zno_d3_stage1.yaml"
PRODUCTION = ROOT / "configs/simulations/zno_production.yaml"


class FakeD3(Calculator):
    """Constant energy and a uniform force, so the sum is easy to check."""

    implemented_properties = ["energy", "forces"]
    created = []

    def __init__(self, method=None, damping=None, **kwargs):
        super().__init__(**kwargs)
        self.method, self.damping = method, damping
        FakeD3.created.append((method, damping))

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": -0.25, "forces": np.full((len(self.atoms), 3), 0.1)}


def fake_dftd3_modules():
    package = types.ModuleType("dftd3")
    module = types.ModuleType("dftd3.ase")
    module.DFTD3 = FakeD3
    package.ase = module
    return {"dftd3": package, "dftd3.ase": module}


def dimer():
    return Atoms("Ar2", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 3.8)], cell=[20, 20, 20])


class DispersionCalculatorTests(unittest.TestCase):
    def test_none_returns_the_dft_calculator_itself(self):
        calc = LennardJones()
        self.assertIs(run_dft.with_dispersion(calc, "none"), calc)
        self.assertIs(run_dft.with_dispersion(calc, ""), calc)  # blank column of older runs

    def test_d3bj_adds_energy_and_forces_with_pbe_bj_parameters(self):
        FakeD3.created.clear()
        with mock.patch.dict(sys.modules, fake_dftd3_modules()):
            atoms = dimer()
            dft = LennardJones()
            atoms.calc = run_dft.with_dispersion(dft, "d3bj")
            total = atoms.get_potential_energy()
            forces = atoms.get_forces()
            reference = dimer()
            reference.calc = LennardJones()
            self.assertAlmostEqual(total, reference.get_potential_energy() - 0.25)
            np.testing.assert_allclose(forces, reference.get_forces() + 0.1)
            self.assertAlmostEqual(run_dft.dispersion_energy(atoms, dft), -0.25)
        self.assertEqual(FakeD3.created, [("PBE", "d3bj")])

    def test_dispersion_energy_is_zero_without_dispersion(self):
        atoms = dimer()
        atoms.calc = LennardJones()
        self.assertEqual(run_dft.dispersion_energy(atoms, atoms.calc), 0.0)

    def test_missing_simple_dftd3_stops_with_the_install_hint(self):
        with mock.patch.dict(sys.modules, {"dftd3": None, "dftd3.ase": None}):
            with self.assertRaisesRegex(SystemExit, "conda install -c conda-forge dftd3-python"):
                run_dft.dispersion_calculator("d3bj")

    def test_unknown_dispersion_is_rejected(self):
        with self.assertRaises(ValueError):
            run_dft.dispersion_calculator("d4")


class SummaryCompatibilityTests(unittest.TestCase):
    ROW = {"ecut_ev": "500", "kpts": "2x3x1", "n_slab_atoms": "16"}

    def test_rows_of_older_runs_count_as_without_dispersion(self):
        self.assertTrue(run_dft.settings_match(dict(self.ROW), 500.0, "2x3x1", 16))
        self.assertFalse(run_dft.settings_match(dict(self.ROW), 500.0, "2x3x1", 16, "d3bj"))

    def test_d3_rows_are_not_merged_into_a_pbe_run(self):
        row = {**self.ROW, "dispersion": "d3bj"}
        self.assertTrue(run_dft.settings_match(row, 500.0, "2x3x1", 16, "d3bj"))
        self.assertFalse(run_dft.settings_match(row, 500.0, "2x3x1", 16, "none"))


class NebSettingsTests(unittest.TestCase):
    def test_calculator_settings_carry_the_dispersion(self):
        self.assertEqual(run_neb.calculator_settings("base", 500.0, (2, 3, 1))["dispersion"], "none")
        self.assertEqual(run_neb.calculator_settings("base", 500.0, (2, 3, 1), "d3bj")["dispersion"], "d3bj")


    def test_neb_refuses_endpoints_with_different_dispersion(self):
        rows = {
            ("ZnO", "top_metal"): {"dft_status": "ok", "dft_profile": "base", "dispersion": "d3bj"},
            ("ZnO", "bridge"): {"dft_status": "ok", "dft_profile": "base", "dispersion": ""},
        }
        record = {}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(run_neb.PairAborted) as raised:
                run_neb._run_pair(record, material="ZnO", initial_site="top_metal", final_site="bridge",
                                  args=None, results_dir=Path(tmp), dft_rows=rows, out_dir=Path(tmp), world=None)
        self.assertIn("different dispersion corrections", str(raised.exception))


class SimulationCommandTests(unittest.TestCase):
    def test_stage1_config_passes_d3bj_to_run_dft(self):
        config = simulations.load_config(STAGE1)
        command = simulations.build_commands("dft", config, ["ZnO"], ["bridge"], 6)[0]
        self.assertEqual(command[command.index("--dispersion") + 1], "d3bj")

    def test_production_config_stays_without_dispersion(self):
        config = simulations.load_config(PRODUCTION)
        command = simulations.build_commands("dft", config, ["ZnO"], ["bridge"], 6)[0]
        self.assertEqual(command[command.index("--dispersion") + 1], "none")

    def test_stage1_config_differs_from_production_only_in_name_and_dispersion(self):
        stage1, production = simulations.load_config(STAGE1), simulations.load_config(PRODUCTION)
        self.assertEqual(stage1["methods"]["dft"].pop("dispersion"), "d3bj")
        self.assertEqual(stage1["methods"], production["methods"])
        self.assertEqual(stage1["selection"], production["selection"])
        self.assertEqual(stage1["execution"], production["execution"])

    def test_unknown_dispersion_in_the_config_fails(self):
        config = simulations.load_config(STAGE1)
        config["methods"]["dft"]["dispersion"] = "d4"
        with self.assertRaisesRegex(SystemExit, "dispersion must be none or d3bj"):
            simulations.build_commands("dft", config, ["ZnO"], ["bridge"], 6)


class GuardTests(unittest.TestCase):
    def write_summary(self, root: Path, dispersion: str) -> None:
        with (root / "dft_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["material", "site", "dft_profile", "dispersion"])
            writer.writeheader()
            writer.writerow({"material": "ZnO", "site": "bridge", "dft_profile": "base", "dispersion": dispersion})

    def test_md_refuses_a_dispersion_corrected_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_summary(Path(tmp), "d3bj")
            with self.assertRaisesRegex(RuntimeError, "PBE without dispersion only"):
                run_md.read_dft_row(Path(tmp), "ZnO", "bridge")

    def test_md_accepts_rows_without_dispersion(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_summary(Path(tmp), "none")
            self.assertEqual(run_md.read_dft_row(Path(tmp), "ZnO", "bridge")["site"], "bridge")


if __name__ == "__main__":
    unittest.main()
