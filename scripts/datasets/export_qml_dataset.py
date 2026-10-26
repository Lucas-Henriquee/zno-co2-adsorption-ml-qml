"""Export QML descriptor features from the ranked DFT table (analyze_results.py).

Audit A2 corrections:

* Missing values are never replaced by 0.0. Rows with a missing selected feature
  are dropped and reported (a missing adsorption energy is not "0 eV").
* The target is the adsorption site label, not ``total_score``, which was a
  function of the exported features themselves (and has been removed).
* Raw values are exported alongside min-max scaled ones. The scaled columns are
  fitted on all exported rows, for inspection only; a model must refit its scaler
  on its own training split (the model scripts do so).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scripts.common.project_config import RESULTS_DIR

SITE_LABELS = {"top_metal": 0, "top_oxygen": 1, "bridge": 2}

# Ranked-table column -> exported feature name. The band gap is a PBE Kohn-Sham
# gap (Gamma grid); the TDDFT onset belongs to the isolated CO2 molecule.
FEATURE_COLUMNS = {
    "adsorption_energy_ev": "x_adsorption_energy",
    "adsorbed_band_gap_gamma_ev": "x_pbe_band_gap",
    "co2_isolated_onset_ev": "x_co2_isolated_onset",
}
DEFAULT_FEATURES = ["adsorption_energy_ev", "adsorbed_band_gap_gamma_ev"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export QML features (no zero imputation, site label as target)."
    )
    parser.add_argument("--ranking", type=Path, default=RESULTS_DIR / "analysis" / "ranked_candidates.csv")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "datasets" / "qml_features.csv")
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        choices=sorted(FEATURE_COLUMNS),
        help="Ranked-table columns to export.",
    )
    parser.add_argument(
        "--default-material",
        default=None,
        help="Material name used only when the ranking table has no material column.",
    )
    return parser.parse_args()


def read_ranked_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing ranking file: {path}. Run analyze_results.py first.")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float:
    """Parse a number; missing or non-finite values become NaN (never 0.0)."""
    try:
        parsed = float(value) if value is not None else float("nan")
    except ValueError:
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def min_max_scale(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def build_qml_rows(
    rows: list[dict[str, str]],
    default_material: str | None = None,
    features: list[str] | None = None,
) -> list[dict[str, str | float | int]]:
    if not rows:
        raise SystemExit("The ranking table contains no rows.")
    features = list(features or DEFAULT_FEATURES)
    required = {"site", *features}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise SystemExit(f"Ranking table is missing columns: {missing}")
    if "material" not in rows[0] and not default_material:
        raise SystemExit(
            "Ranking table has no material column; provide --default-material."
        )

    complete: list[tuple[dict[str, str], dict[str, float]]] = []
    for row in rows:
        if row["site"] not in SITE_LABELS:
            print(f"[qml-export] skipped site {row['site']!r}: not a classification site")
            continue
        values = {name: parse_float(row.get(name)) for name in features}
        absent = [name for name, value in values.items() if math.isnan(value)]
        if absent:
            print(f"[qml-export] dropped {row.get('material', default_material)} @ {row['site']}: missing {absent}")
            continue
        complete.append((row, values))
    if not complete:
        raise SystemExit("No rows with every selected feature present.")

    bounds = {
        name: (min(v[name] for _, v in complete), max(v[name] for _, v in complete))
        for name in features
    }
    qml_rows: list[dict[str, str | float | int]] = []
    for row, values in complete:
        item: dict[str, str | float | int] = {
            "material": row.get("material") or str(default_material),
            "site": row["site"],
        }
        for name in features:
            item[f"{FEATURE_COLUMNS[name]}_raw"] = values[name]
            item[f"{FEATURE_COLUMNS[name]}_scaled"] = min_max_scale(values[name], *bounds[name])
        item["label_site"] = row["site"]
        item["label_site_id"] = SITE_LABELS[row["site"]]
        qml_rows.append(item)
    return qml_rows


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        raise SystemExit("No rows available for QML dataset export.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    ranked_rows = read_ranked_results(args.ranking)
    qml_rows = build_qml_rows(ranked_rows, default_material=args.default_material, features=args.features)
    write_csv(args.output, qml_rows)
    print(f"[qml-export] wrote {args.output}")


if __name__ == "__main__":
    main()
