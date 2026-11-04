"""CO2-reduction analysis tables (co2_reduction) without the removed proxy score.

The screening figures of render_publication_figures were retired to scripts/legacy/.
"""

import contextlib
import io
import math

import pytest
from ase import Atoms
from ase.io import write

from scripts.analysis import analysis_co2_reduction as analysis


def _dft_row(site, eads, status="ok", profile="base", bond=1.20):
    return {
        "material": "ZnO", "site": site, "dft_status": status, "dft_profile": profile,
        "adsorption_energy_ev": str(eads), "co_bond_1_ang": str(bond), "co_bond_2_ang": str(bond),
        "oco_angle_deg": "150.0", "c_surface_distance_ang": "2.0", "final_state": "physisorbed",
    }


def _co2(bond):
    atoms = Atoms("OCO", positions=[(-bond, 0, 0), (0, 0, 0), (bond, 0, 0)], cell=[10, 10, 10], pbc=True)
    atoms.center()
    return atoms


def test_proxy_score_is_gone():
    for name in ("BASELINE_CO_BOND_ANG", "TARGET_ADSORPTION_EV", "gaussian_score", "clamp01"):
        assert not hasattr(analysis, name)
    rows = analysis.merge_dft_tddft([_dft_row("top_metal", -0.3)], [])
    assert not any("score" in key for key in rows[0])
    for table in analysis.build_figure_tables(rows).values():
        assert not any("score" in key for key in table[0])


def test_co2_baseline_from_run_reference(tmp_path):
    ref = tmp_path / "dft" / "_reference" / "base"
    ref.mkdir(parents=True)
    write(ref / "co2_ref.base_optimized.cif", _co2(1.176))
    lookup = analysis.co2_baseline_lookup(tmp_path)
    rows = analysis.merge_dft_tddft([_dft_row("top_metal", -0.3, bond=1.20)], [], co2_baseline=lookup)
    assert rows[0]["co_bond_baseline_ang"] == pytest.approx(1.176, abs=1e-6)
    assert rows[0]["co_bond_stretch_ang"] == pytest.approx(1.20 - 1.176, abs=1e-6)
    assert rows[0]["co_bond_baseline_source"].endswith("co2_ref.base_optimized.cif")

    # Pathway reference of the same profile as the fallback.
    other = tmp_path / "pathways" / "_references" / "alt" / "co2"
    other.mkdir(parents=True)
    write(other / "co2_optimized.xyz", _co2(1.18))
    assert lookup("alt")[0] == pytest.approx(1.18, abs=1e-6)

    # Explicit value wins; missing reference -> empty stretch, never 1.16.
    assert analysis.co2_baseline_lookup(tmp_path, 1.17)("base") == (1.17, "--co2-gas-bond-ang")
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rows = analysis.merge_dft_tddft(
            [_dft_row("top_metal", -0.3, profile="missing")], [], co2_baseline=analysis.co2_baseline_lookup(tmp_path)
        )
    assert math.isnan(rows[0]["co_bond_stretch_ang"]) and math.isnan(rows[0]["co_bond_baseline_ang"])
    assert "WARNING" in out.getvalue()
    assert math.isnan(analysis.merge_dft_tddft([_dft_row("top_metal", -0.3)], [])[0]["co_bond_stretch_ang"])


def test_positive_adsorption_energy_warning_keeps_final_state():
    rows = {row["site"]: row for row in analysis.merge_dft_tddft(
        [_dft_row("top_metal", 0.02), _dft_row("bridge", 0.30), _dft_row("top_oxygen", -0.40)], []
    )}
    assert "not bound at PBE level" in rows["top_metal"]["eads_warning"]
    assert "0.05 eV numerical uncertainty" in rows["top_metal"]["eads_warning"]
    assert rows["bridge"]["eads_warning"] == analysis.EADS_POSITIVE_WARNING
    assert rows["top_oxygen"]["eads_warning"] == ""
    assert rows["top_metal"]["final_state"] == "physisorbed"
    heatmap = analysis.build_figure_tables(list(rows.values()))["figure_adsorption_heatmap.csv"]
    assert "eads_warning" in heatmap[0]


def test_tddft_renamed_columns_with_fallback():
    old = {"material": "ZnO", "site": "top_metal", "co2_isolated_onset_ev": "8.5", "co2_isolated_total_f": "0.3"}
    new = {"material": "ZnO", "site": "bridge", "co2_isolated_first_bright_excitation_ev": "8.7"}
    rows = {row["site"]: row for row in analysis.merge_dft_tddft(
        [_dft_row("top_metal", -0.3), _dft_row("bridge", -0.3)], [old, new]
    )}
    assert rows["top_metal"]["co2_isolated_first_bright_excitation_ev"] == 8.5
    assert rows["bridge"]["co2_isolated_first_bright_excitation_ev"] == 8.7
    assert "co2_isolated_total_f" not in rows["top_metal"]


def test_ranking_only_by_limiting_potential():
    merged = analysis.merge_dft_tddft(
        [_dft_row("top_metal", -0.9), _dft_row("top_oxygen", -0.1), _dft_row("bridge", -0.5, status="failed")], []
    )
    table = [
        {"material": "ZnO", "site": "top_metal", "pathway_validity": "ok", "limiting_potential_v": -1.2, "limiting_step": "a"},
        {"material": "ZnO", "site": "top_oxygen", "pathway_validity": "ok", "limiting_potential_v": -0.6, "limiting_step": "b"},
        {"material": "ZnO", "site": "bridge", "pathway_validity": "ok", "limiting_potential_v": -0.1, "limiting_step": "c"},
    ]
    ranked = analysis.attach_limiting_potentials(merged, table)
    # E_ads plays no role; the failed DFT case is not ranked.
    assert [(row["site"], row["rank_by_limiting_potential"]) for row in ranked] == [
        ("top_oxygen", 1), ("top_metal", 2), ("bridge", ""),
    ]
    ranking = analysis.build_figure_tables(ranked)["figure_site_ranking.csv"]
    assert ranking[0]["limiting_potential_v"] == -0.6
