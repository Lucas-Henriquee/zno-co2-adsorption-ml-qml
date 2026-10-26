# Results

| Location | Contents |
| --- | --- |
| `simulations/<run-id>/` | Simulation campaigns |
| `models/<run-id>/` | Model runs |
| `analyses/<run-id>/` | Tables and plots |
| `workflows/<run-id>/` | Combined-workflow manifests and logs |
| `_invalid_ic_v1/` | Archived results of the first IC campaign; **invalid**, see its README |

New runs use distinct IDs and record their inputs and settings.

`simulations/zno_wurtzite_production/` is an interrupted campaign with
different input geometries. Do not use it.

The [figure guide](../docs/figures.md) describes the manuscript figures; they must be
regenerated from new runs, since the earlier benchmarks are in `_invalid_ic_v1/`.
