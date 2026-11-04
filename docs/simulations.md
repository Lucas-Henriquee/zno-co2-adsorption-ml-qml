# Simulations

`run_simulations.sh` (or `python -m scripts.workflows.run_simulations` inside the
simulation environment) runs the GPAW methods of `scripts/simulations/` into an
isolated run directory, `results/simulations/<run-id>/`.

```bash
./run_simulations.sh all --dry-run
./run_simulations.sh dft --run-id zno_dft_01
./run_simulations.sh dft tddft --run-id zno_optical_01
./run_simulations.sh md neb --run-id zno_dynamics_01
./run_simulations.sh all --run-id zno_dft_01 --resume
```

Environment selection: [environments/README.md](../environments/README.md).
`notebooks/zno-co2-lucas.ipynb` calls the same module.

## Methods

`geometry`, `dft`, `tddft`, `co2rr` (alias `pathways`), `md`, `neb`, or `all`.
Prerequisites are added automatically: `geometry` -> `dft` -> {`tddft`, `co2rr`, `md`,
`neb`}. For example, `tddft` resolves to `geometry, dft, tddft`.

## Options

| Option | Effect |
| --- | --- |
| `--config PATH` | YAML file (default `configs/simulations/zno_production.yaml`; with `--resume`, the run's `config.yaml`) |
| `--run-id NAME` | Run directory name (letters, digits, `.`, `_`, `-`); default: next `simulation_NN` |
| `--resume` | Continue an existing run; needs `--run-id` |
| `--retry-failed` | With `--resume`: archive outputs of failed/interrupted methods to `failed_attempts/` and rerun them |
| `--no-dependencies` | Do not add prerequisites; refused without `--resume` when a selected method has prerequisites |
| `--dry-run` | Validate and print the commands; creates nothing |
| `--materials ZnO` | Comma-separated material override |
| `--sites top_metal,bridge` | Comma-separated site override; NEB needs at least two sites |
| `--mpi-processes N` | Override `execution.mpi.processes` |
| `--list-methods` | Print the methods |

`carbonate` and `tridentate` are exploratory sites, accepted only with `geometry` and
`dft` (e.g. `dft --sites carbonate`).

## Run directory and resume

- A new run copies the configuration to `config.yaml` and writes `run_manifest.json`
  (commands, configuration hash, environment name, package versions, materials, sites,
  MPI processes, per-method status and timestamps). An existing run directory is
  never reused without `--resume`.
- `--resume` without `--config` reads the run's `config.yaml` snapshot. A config file
  whose hash differs from the recorded one is refused, so an edited reference YAML
  cannot continue an old run.
- Resume also refuses a different environment name, package versions, materials,
  sites, MPI process count or `OMP_NUM_THREADS`. The launcher, `python -m` and the
  notebook record the same environment name (a prefix path by its last component).
  If the recorded name differs (`environment` in `run_manifest.json`) but it is the
  same environment, set `QML_ZNO_ACTIVE_ENVIRONMENT=<recorded name>`.
- `run_simulations` writes only the method outputs. The CO₂RR corrections, NEB tables
  and MD trajectory analysis are analysis methods (`run_analysis.sh science` or
  `paper`, [analysis](analysis.md)).
- One invocation per run: `run_simulations` holds an exclusive lock on
  `<run>/.run_simulations.lock` until it exits. A second invocation on the same run
  (for example a `--resume --retry-failed` while the first is alive) stops with
  "another run_simulations is using this run". `run_md` started by hand is not locked.
- Completed methods are skipped. A method with status `failed` or `running`
  (interrupted) stops the run unless `--retry-failed` is given; otherwise use a new
  run ID.
- `--resume --dry-run` only prints the plan; the manifest checks run without
  `--dry-run`.

Run layout: [results/simulations/README.md](../results/simulations/README.md).

## Status dashboard

Read-only; without a run ID it shows the most recently modified run.

```bash
python -m scripts.workflows.run_status zno_run_01
FORCE_COLOR=1 watch -c -n 60 python -m scripts.workflows.run_status zno_run_01
```

Options: `--tz ZONE` (default Brasília time), `--stale-minutes N`, `--no-color`.

## Configuration

`configs/simulations/zno_production.yaml` is the reference configuration. MD and NEB
read their remaining settings from the run's `config.yaml`; CO₂RR reads its
thermochemistry from the `--config` file. The CLI defaults of the standalone DFT,
CO₂RR, MD and NEB scripts equal the YAML (checked by tests), except that a standalone
`run_md.py` without `--seeds` and outside a run directory runs seed 1234 only.

| Setting | Value |
| --- | --- |
| Cutoff / k-points (periodic methods) | 500 eV / 2×3×1 |
| DFT | BFGS, fmax 0.05 eV/Å, 180 steps + 1 continued block |
| TDDFT (isolated CO₂ in its adsorbed geometry, no slab) | fd, h = 0.25 Å, 18 Å box, 68 converged + 8 buffer bands, `jend` = last converged band, 30 eV window, bright if f >= 0.001 |
| CO₂RR | fmax 0.05 eV/Å, 260 steps; vibrations on, 298.15 K, 0 V vs RHE |
| MD | Langevin, 300 K, 1 fs, friction 0.01 fs⁻¹, 500 equilibration + 1500 production steps, log every 20 steps, seeds 1234, 2345, 3456 |
| NEB | 3 internal images, idpp, FIRE, 300 steps; plain NEB to fmax 0.08, then 0.05 eV/Å; pairs top_metal–bridge, bridge–top_oxygen |
| MPI | 6 processes, `OMP_NUM_THREADS=1` |

- MD: equilibration frames are written separately and never used for descriptors.
  Each seed is an independent trajectory, `md/<material>/<site>/seed_<n>/`.
- NEB: the second phase climbs only when an internal image is above both endpoints;
  otherwise the plain NEB continues. Barriers <= 0.03 eV (`barrier_resolution_ev`)
  are reported as not resolved. With 6 processes each image gets 2 ranks.
- CO₂RR free energies are stored as computed (variant a). The analysis method
  `co2rr_corrections` adds variants b (quasi-harmonic floor) and c (b + PBE gas
  corrections CO₂ +0.19 eV, CO -0.24 eV) in the analysis run; every analysis uses the
  configured variant (`methods.co2rr_corrections.variant`, c).

A finished process does not prove optimizer or SCF convergence; check the method logs
before using the outputs.

## Optional: relax a NEB image

When a NEB band has an internal image below both endpoints (`endpoint_not_minimum`),
relax that image as a free structure with the run's DFT settings:

```bash
mpiexec -n 6 gpaw python -m scripts.simulations.neb.relax_neb_image \
  --results-dir results/simulations/<run-id> --pair top_metal_to_bridge --image 3 \
  --fmax 0.02 --steps 300
```

It reads `neb/<material>/<pair>/neb_final_images.traj` and writes
`neb/<material>/<pair>/image<N>_relaxed/` (structure, trajectory, log and a JSON with
energy, fmax, steps, convergence, energy relative to each endpoint and the implied
E_ads). An existing folder is kept unless `--overwrite` is given. It is not a
`run_simulations` method and is not recorded in `run_manifest.json`.
