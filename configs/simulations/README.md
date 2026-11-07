# Simulation configuration

`zno_production.yaml` is the reference configuration of the ZnO runs: materials, sites,
MPI settings and output root once, then one section per method. `zno_d3_stage1.yaml`
is the same with `methods.dft.dispersion: d3bj`, for DFT and NEB only (PBE-D3
exploration). `zno_md_100k.yaml` and `zno_md_300k_extra.yaml` are short AIMD runs
(500 + 500 steps) for the site-memory analysis: 100 K with seeds 1234, 2345, 3456, and
300 K with the new seeds 4567, 5678, 6789. Values and resume
rules: [docs/simulations.md](../../docs/simulations.md).

```bash
./run_simulations.sh all --dry-run
```

- Every run copies the file to `results/simulations/<run-id>/config.yaml`; a resumed run
  continues from that copy.
- `execution.environment` is recorded only when no environment name can be detected
  (see [environments/README.md](../../environments/README.md)).
- The filename names a configuration; it is not evidence of convergence.
- `methods.md.rng_stream: per_site` gives every (seed, site) its own random stream;
  a `config.yaml` without the key (older runs) uses `shared`. `run_md` locks each
  trajectory, skips seeds that are already complete and needs `--overwrite` for an
  incomplete one; `OMP_NUM_THREADS` defaults to 1. Details:
  [docs/simulations.md](../../docs/simulations.md).
