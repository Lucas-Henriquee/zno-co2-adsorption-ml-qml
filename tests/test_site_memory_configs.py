"""Short AIMD configurations of the site-memory analysis differ from production only in the MD."""

import unittest
from pathlib import Path

from scripts.workflows import run_simulations as simulations

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "configs/simulations/zno_production.yaml"
CASES = {
    "zno_md_100k.yaml": {"temperature_kelvin": 100.0, "seeds": [1234, 2345, 3456]},
    "zno_md_300k_extra.yaml": {"temperature_kelvin": 300.0, "seeds": [4567, 5678, 6789]},
}
SHORT_MD = {"steps": 500, "equilibration_steps": 500, "production_steps": 500}


class SiteMemoryConfigTests(unittest.TestCase):
    def test_only_temperature_steps_and_seeds_differ_from_production(self):
        production = simulations.load_config(PRODUCTION)
        for name, md_changes in CASES.items():
            with self.subTest(config=name):
                config = simulations.load_config(ROOT / "configs/simulations" / name)
                expected_md = {**production["methods"]["md"], **SHORT_MD, **md_changes}
                self.assertEqual(config["methods"]["md"], expected_md)
                other = {key: value for key, value in config["methods"].items() if key != "md"}
                self.assertEqual(other, {k: v for k, v in production["methods"].items() if k != "md"})
                self.assertEqual(config["selection"], production["selection"])
                self.assertEqual(config["execution"], production["execution"])

    def test_md_command_uses_the_configured_temperature(self):
        for name, md_changes in CASES.items():
            with self.subTest(config=name):
                config = simulations.load_config(ROOT / "configs/simulations" / name)
                command = simulations.build_commands("md", config, ["ZnO"], ["bridge"], 6)[0]
                self.assertEqual(float(command[command.index("--temperature") + 1]), md_changes["temperature_kelvin"])
                self.assertEqual(int(command[command.index("--steps") + 1]), 500)

    def test_new_300k_seeds_do_not_repeat_the_production_seeds(self):
        production = simulations.load_config(PRODUCTION)["methods"]["md"]["seeds"]
        self.assertFalse(set(CASES["zno_md_300k_extra.yaml"]["seeds"]) & set(production))


if __name__ == "__main__":
    unittest.main()
