"""run_simulations.py: exploratory sites, DFT command options and failure marking (no GPAW)."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from scripts.workflows import run_simulations as simulations

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "configs/simulations/zno_production.yaml"


class ExploratorySiteTests(unittest.TestCase):
    def test_carbonate_with_dft_only_is_accepted(self):
        simulations.validate_exploratory_sites(["top_metal", "carbonate"], ["geometry", "dft"])

    def test_carbonate_with_other_methods_fails_early(self):
        with self.assertRaisesRegex(SystemExit, "carbonate is a DFT-only exploratory site.*tddft, md"):
            simulations.validate_exploratory_sites(["carbonate"], ["geometry", "dft", "tddft", "md"])

    def test_main_rejects_carbonate_with_all_methods_before_planning(self):
        argv = ["run_simulations", "all", "--sites", "carbonate", "--dry-run"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(simulations, "build_commands") as build, \
                self.assertRaisesRegex(SystemExit, "DFT-only exploratory site"):
            simulations.main()
        build.assert_not_called()


class DftCommandTests(unittest.TestCase):
    def test_production_command_threads_new_dft_options(self):
        config = simulations.load_config(PRODUCTION)
        command = simulations.build_commands("dft", config, ["ZnO"], ["bridge"], 6)[0]
        self.assertEqual(command[command.index("-n") + 1], "6")
        self.assertEqual(command[command.index("--relax-extra-blocks") + 1], "1")
        self.assertNotIn("--scf-force-convergence", command)  # null in YAML: GPAW defaults

    def test_scf_force_convergence_is_passed_when_set(self):
        config = simulations.load_config(PRODUCTION)
        config["methods"]["dft"]["scf_force_convergence"] = 0.01
        command = simulations.build_commands("dft", config, ["ZnO"], ["bridge"], 1)[0]
        self.assertEqual(command[command.index("--scf-force-convergence") + 1], "0.01")


class MethodFailureTests(unittest.TestCase):
    """main() with each method replaced by a tiny Python command."""

    def run_main(self, dft_source):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        config = tmp / "config.yaml"
        shutil.copy(PRODUCTION, config)
        commands = {
            "geometry": [[sys.executable, "-c", "pass"]],
            "dft": [[sys.executable, "-c", textwrap.dedent(dft_source)]],
        }
        argv = ["run_simulations", "dft", "--config", str(config), "--run-id", "w1", "--mpi-processes", "1"]
        error = None
        with mock.patch.object(simulations, "PROJECT_ROOT", tmp), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(simulations, "build_commands",
                                  side_effect=lambda method, *args: commands[method]), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                simulations.main()
            except BaseException as exc:  # main re-raises the method failure
                error = exc
        manifest = json.loads((tmp / "results/simulations/w1/run_manifest.json").read_text())
        return error, manifest

    WRITE_SUMMARY = """
        import csv, os, sys
        status = {status!r}
        path = os.path.join(os.environ["QML_ZNO_RESULTS_DIR"], "dft_summary.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["material", "site", "dft_status"])
            for site in ("top_metal", "top_oxygen", "bridge"):
                writer.writerow(["ZnO", site, status.get(site, "ok")])
        """

    def test_nonzero_exit_marks_method_failed(self):
        error, manifest = self.run_main("import sys; sys.exit(3)")
        self.assertIsNotNone(error)
        self.assertEqual(manifest["methods"]["geometry"]["status"], "completed")
        self.assertEqual(manifest["methods"]["dft"]["status"], "failed")
        self.assertEqual(manifest["status"], "failed")

    def test_zero_exit_with_failed_dft_case_is_not_completed(self):
        error, manifest = self.run_main(self.WRITE_SUMMARY.format(status={"bridge": "ads_failed"}))
        self.assertIn("ZnO/bridge", str(error))
        self.assertEqual(manifest["methods"]["dft"]["status"], "failed")

    def test_zero_exit_with_all_cases_ok_is_completed(self):
        error, manifest = self.run_main(self.WRITE_SUMMARY.format(status={}))
        self.assertIsNone(error)
        self.assertEqual(manifest["methods"]["dft"]["status"], "completed")


class RetryFailedTests(unittest.TestCase):
    """--resume --retry-failed archives a failed method's outputs and runs it again."""

    TDDFT_FAIL = """
        import os, sys
        root = os.environ["QML_ZNO_RESULTS_DIR"]
        os.makedirs(os.path.join(root, "tddft", "ZnO"), exist_ok=True)
        open(os.path.join(root, "tddft", "ZnO", "partial.txt"), "w").write("old")
        open(os.path.join(root, "tddft_summary.csv"), "w").write("old")
        sys.exit(1)
        """

    def run(self, result=None):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        return super().run(result)

    def main(self, tddft_source, *extra):
        config = self.tmp / "config.yaml"
        if not config.exists():
            shutil.copy(PRODUCTION, config)
        commands = {
            "geometry": [[sys.executable, "-c", "pass"]],
            "dft": [[sys.executable, "-c", textwrap.dedent(MethodFailureTests.WRITE_SUMMARY.format(status={}))]],
            "tddft": [[sys.executable, "-c", textwrap.dedent(tddft_source)]],
        }
        argv = ["run_simulations", "dft", "tddft", "--config", str(config), "--run-id", "r1",
                "--mpi-processes", "1", *extra]
        error = None
        with mock.patch.object(simulations, "PROJECT_ROOT", self.tmp), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(simulations, "build_commands",
                                  side_effect=lambda method, *args: commands[method]), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                simulations.main()
            except BaseException as exc:
                error = exc
        run_dir = self.tmp / "results/simulations/r1"
        return error, json.loads((run_dir / "run_manifest.json").read_text()), run_dir

    def test_failed_method_needs_the_flag_and_is_archived_on_retry(self):
        error, manifest, run_dir = self.main(self.TDDFT_FAIL)
        self.assertIsNotNone(error)
        self.assertEqual(manifest["methods"]["tddft"]["status"], "failed")

        error, manifest, _ = self.main("pass", "--resume")
        self.assertIn("--retry-failed", str(error))

        error, manifest, _ = self.main("pass", "--resume", "--retry-failed")
        self.assertIsNone(error)
        record = manifest["methods"]["tddft"]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["previous_attempts"][0]["status"], "failed")
        archive = Path(record["previous_attempts"][0]["archived_to"])
        self.assertTrue((archive / "tddft" / "ZnO" / "partial.txt").exists())
        self.assertTrue((archive / "tddft_summary.csv").exists())
        self.assertTrue((archive / "tddft.log").exists())
        self.assertFalse((run_dir / "tddft_summary.csv").exists())
        self.assertEqual(manifest["methods"]["dft"]["status"], "completed")

    def test_retry_failed_requires_resume(self):
        with mock.patch.object(sys, "argv", ["run_simulations", "dft", "--retry-failed"]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                simulations.main()


if __name__ == "__main__":
    unittest.main()
