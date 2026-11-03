# Results

| Path | Content |
| --- | --- |
| `simulations/<run-id>/` | Simulation runs ([layout](simulations/README.md)) |
| `models/<run-id>/` | Model runs ([layout](models/README.md)) |
| `analyses/<run-id>/` | Analysis runs ([layout](analyses/README.md)) |
| `workflows/<run-id>/` | `run_all.sh` manifests and stage logs |
| `_invalid_ic_v1/` | First campaign, invalid ([README](_invalid_ic_v1/README.md)) |
| `_local_archive_pre_audit/` | Untracked local runs made before the 2026-10 audit, invalid; ignored by Git |

Run directories are ignored by Git. Manuscript figures must be regenerated from new
runs ([docs/figures.md](../docs/figures.md)).
