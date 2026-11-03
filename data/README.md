# Data

| Path | Content |
| --- | --- |
| `geometries/ZnO/` | Starting structures: 2×1 ZnO(10-10) slab and CO₂ at `top_metal`, `top_oxygen`, `bridge`, plus the exploratory `carbonate` and `tridentate`; used by standalone scripts and as site references |
| `generated/<run-id>/` | Dataset runs of `run_datasets.sh` ([layout](generated/README.md)) |

Workflow simulation runs build their own geometries inside the run directory
(`data/geometries/`). First-campaign datasets and TDDFT tables are archived, invalid,
in `results/_invalid_ic_v1/`. Dataset generation: [docs/datasets.md](../docs/datasets.md).
