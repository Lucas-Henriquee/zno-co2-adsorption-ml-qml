"""Synthetic AIMD frame files for the site-memory tests (imported by name, not as tests.*)."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import site_memory as sm

SITES = ["top_metal", "top_oxygen", "bridge"]
START_HEIGHT = {"top_metal": 2.7, "top_oxygen": 3.6, "bridge": 4.0}


def write_run(root: Path, seeds, temperature=300.0, memory_fs=100.0, eq_steps=100, prod_steps=200,
              interval=20, rng_seed=0):
    """Frames whose geometry keeps the site for memory_fs, then is pure noise around 3.8 A."""
    rng = np.random.default_rng(rng_seed)
    (root).mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(
        "methods:\n  md:\n"
        f"    temperature_kelvin: {temperature}\n    timestep_fs: 1.0\n"
        f"    log_interval: {interval}\n    equilibration_steps: {eq_steps}\n", encoding="utf-8")
    for site in SITES:
        rows = []
        for seed in seeds:
            for phase, offset, steps in (("equilibration", 0, eq_steps), ("production", eq_steps, prod_steps)):
                for index in range(steps // interval + 1):
                    step = offset + index * interval
                    t = float(step)
                    height = START_HEIGHT[site] if t < memory_fs else 3.8
                    height += rng.normal(0, 0.02 if t < memory_fs else 0.4)
                    rows.append({"seed": seed, "phase": phase, "frame_index": index, "step": step,
                                 "time_fs": index * interval, "nearest_site": site,
                                 "nearest_site_lateral_ang": 0.0, "start_site_lateral_ang": 0.0,
                                 "c_height_top_plane_ang": height,
                                 "c_min_slab_distance_ang": height + 0.2 + rng.normal(0, 0.02),
                                 "c_o_min_slab_distance_ang": height - 0.1 + rng.normal(0, 0.02),
                                 "desorbed": 0, "msd_lateral_ang2": 0.0, "site_assignment": site})
        folder = root / "md" / "ZnO" / site
        folder.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(folder / sm.FRAME_FILE, index=False)
