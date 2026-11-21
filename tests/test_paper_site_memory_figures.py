"""paper_site_memory_figures: every item from small synthetic inputs, and skipped items."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.analysis import site_memory as sm
from scripts.visualization import paper_site_memory_figures as paper
from site_memory_fixtures import write_run

FAST = ["--qsvm-kernel", "zz_exact", "--n-permutations", "50", "--models", "svm_rbf", "qsvm", "qsvm_bw"]
DFT_COLUMNS = ("material,site,dft_status,adsorption_energy_ev,c_surface_distance_ang,c_min_slab_distance_ang,"
               "oco_angle_deg,clean_band_gap_gamma_ev,adsorbed_band_gap_gamma_ev\n")


def write_simulation(root: Path, energies, neb_shift=0.0):
    root.mkdir(parents=True)
    lines = [DFT_COLUMNS] + [f"ZnO,{site},ok,{e},{h},{h + 0.4},179,0.85,0.86\n"
                             for site, e, h in zip(("top_metal", "top_oxygen", "bridge"), energies, (2.5, 3.4, 3.8))]
    (root / "dft_summary.csv").write_text("".join(lines), encoding="utf-8")
    for initial, final, rel in (("top_metal", "bridge", [0, -0.03, -0.09, -0.10, -0.06]),
                                ("bridge", "top_oxygen", [0, 0.0, 0.01, 0.03, 0.01])):
        folder = root / "neb" / "ZnO" / f"{initial}_to_{final}"
        folder.mkdir(parents=True)
        (folder / "neb_run_summary.json").write_text(json.dumps({
            "path": f"{initial}_to_{final}", "initial_site": initial, "final_site": final,
            "relative_energies": [value + neb_shift * k for k, value in enumerate(rel)]}), encoding="utf-8")


def write_ibm_summary(folder: Path, temperature: int):
    folder.mkdir(parents=True)
    rows = []
    for name, start, bandwidth, shots in ((f"ibm_{temperature}k_bw1_w0", 0, "A", 512),
                                          (f"ibm_{temperature}k_bw1_w0_rep2", 0, "A", 512),
                                          (f"ibm_{temperature}k_bwauto", 0, "B", 512),
                                          (f"ibm_{temperature}k_s256", 0, "A", 256)):
        rows.append({"folder": name, "t_start_fs": start, "bandwidth": bandwidth, "shots": shots, "reps": 1,
                     "svm_rbf_correct": 9, "qsvm_ideal_correct": 8, "qsvm_ibm_correct": 7, "n_trajectories": 9,
                     "kernel_mae": 0.02, "kernel_bias": -0.01, "depth_mean": 28.5, "two_qubit_gates_mean": 6.0,
                     "quantum_seconds": 12.0})
    pd.DataFrame(rows).to_csv(folder / "ibm_summary.csv", index=False)


class PaperSiteMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        write_simulation(root / "pbe", (0.04, -0.01, -0.02))
        write_simulation(root / "d3", (-0.17, -0.13, -0.18), neb_shift=0.01)
        write_run(root / "cold", [1, 2, 3], temperature=100.0, memory_fs=300.0)
        write_run(root / "hot", [1, 2, 3], temperature=300.0, memory_fs=60.0, rng_seed=4)
        with mock.patch("builtins.print"):
            sm.main(["--runs", str(root / "cold"), "--output-dir", str(root / "m100"), *FAST])
            sm.main(["--runs", str(root / "hot"), "--output-dir", str(root / "m300"), *FAST])
            sm.main(["--runs", str(root / "hot"), "--output-dir", str(root / "m300_full"), *FAST,
                     "--qsvm-entanglement", "full"])
        write_ibm_summary(root / "ibm100", 100)
        write_ibm_summary(root / "ibm300", 300)
        cls.root = root
        cls.out = root / "paper"
        with mock.patch("builtins.print"):
            paper.main(["--pbe-run", str(root / "pbe"), "--d3-run", str(root / "d3"),
                        "--memory-100k", str(root / "m100"), "--memory-300k", str(root / "m300"),
                        "--sensitivity", str(root / "m300_full"),
                        "--ibm-summaries", str(root / "ibm100"), str(root / "ibm300"),
                        "--dpi", "60", "--output-dir", str(cls.out)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_item_and_the_registry(self):
        registry = pd.read_csv(self.out / "registry.csv")
        self.assertEqual(set(registry["item"]), set(paper.ITEMS))
        for name in registry["file"]:
            self.assertTrue((self.out / name).is_file(), name)
        self.assertTrue(registry["sha256"].str.len().eq(64).all())

    def test_adsorption_table_and_neb_methods(self):
        table = pd.read_csv(self.out / "table_adsorption.csv")
        self.assertEqual(list(table.columns), ["Site", "E_ads PBE (eV)", "E_ads PBE-D3 (eV)"])
        self.assertEqual(table.loc[table["Site"] == "Bridge", "E_ads PBE-D3 (eV)"].item(), -0.18)
        neb = pd.read_csv(self.out / "neb_pbe_d3_data.csv")
        self.assertEqual(set(neb["method"]), {"PBE", "PBE-D3"})
        self.assertEqual(neb["global_index"].max(), 8)  # two chained 5-image bands

    def test_site_memory_table(self):
        table = pd.read_csv(self.out / "table_site_memory.csv")
        self.assertEqual(sorted(set(table["Temperature (K)"])), [100.0, 300.0])
        self.assertEqual(set(table["Model"]), {"SVM (RBF)", "QSVM (ZZ, standard)", "QSVM (ZZ, scaled)"})
        sensitivity = pd.read_csv(self.out / "supp_qsvm_sensitivity.csv")
        self.assertIn("full", set(sensitivity["Entanglement"]))

    def test_ibm_main_figure_keeps_one_run_per_configuration(self):
        data = pd.read_csv(self.out / "ibm_hardware_data.csv")
        self.assertEqual(len(data), 2 * 2)  # per temperature: standard and scaled at 0 fs, 512 shots
        self.assertFalse(data["folder"].str.contains("rep2|s256").any())
        runs = pd.read_csv(self.out / "supp_ibm_runs.csv")
        self.assertEqual(len(runs), 8)
        self.assertEqual(set(runs["bandwidth"]), {"standard", "scaled"})

    def test_missing_inputs_are_skipped(self):
        out = self.root / "only_memory"
        with mock.patch("builtins.print"):
            paper.main(["--memory-300k", str(self.root / "m300"), "--dpi", "60", "--output-dir", str(out)])
        items = set(pd.read_csv(out / "registry.csv")["item"])
        self.assertEqual(items, {"site_memory", "table_site_memory", "supp_height_time", "supp_qsvm_sensitivity"})


if __name__ == "__main__":
    unittest.main()
