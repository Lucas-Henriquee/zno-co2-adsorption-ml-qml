"""paper_figures (fig03-fig07) and paper_tables (table02) on synthetic run directories."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from scripts.analysis import paper_tables
from scripts.visualization import paper_figures

MAT = "ZnO"
SITES = ("top_metal", "top_oxygen", "bridge")
EADS = {"top_metal": 0.042, "top_oxygen": -0.011, "bridge": -0.025}
RANKING_WORDS = ("rank", "best", "most stable", "strongest", "preferred", "winner")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dft_rows():
    rows = []
    for i, site in enumerate(SITES):
        rows.append({
            "material": MAT, "site": site, "dft_status": "ok", "dft_profile": "base", "error_message": "",
            "adsorption_energy_ev": EADS[site], "clean_band_gap_gamma_ev": 1.70,
            "adsorbed_band_gap_gamma_ev": 1.68 - 0.01 * i, "co_bond_1_ang": 1.180 + 0.002 * i,
            "co_bond_2_ang": 1.178, "oco_angle_deg": 178.5 - i, "c_surface_distance_ang": 2.9,
            "c_min_slab_distance_ang": 3.10 + 0.1 * i, "o_min_metal_distance_ang": 2.5,
            "final_state": "physisorbed", "warnings": "",
        })
    return rows


def _tddft(run: Path) -> None:
    rows = []
    for site in SITES:
        rows.append({
            "material": MAT, "site": site, "tddft_status": "ok", "osc_threshold": 0.01,
            "co2_isolated_first_excitation_ev": 8.18, "co2_isolated_first_excitation_f": 0.0,
            "co2_isolated_first_bright_excitation_ev": 9.52, "co2_isolated_peak_ev": 11.0,
            "co2_isolated_peak_f": 0.4, "co2_isolated_ks_ionization_threshold_ev": 9.10,
        })
        _write_csv(run / "tddft" / MAT / site / "transitions.csv", [
            {"transition_index": 1, "energy_ev": 8.18, "oscillator_strength": 0.0,
             "oscillator_strength_x": 0, "oscillator_strength_y": 0, "oscillator_strength_z": 0},
            {"transition_index": 2, "energy_ev": 9.52, "oscillator_strength": 0.05,
             "oscillator_strength_x": 0.05, "oscillator_strength_y": 0, "oscillator_strength_z": 0},
            {"transition_index": 3, "energy_ev": 11.0, "oscillator_strength": 0.4,
             "oscillator_strength_x": 0.1, "oscillator_strength_y": 0.1, "oscillator_strength_z": 0.2},
        ])
    _write_csv(run / "tddft_summary.csv", rows)


def _co2rr(analysis: Path) -> None:
    values = {"top_metal": ("ok", -1.982, "CO2_to_COOH"), "top_oxygen": ("unbound", -2.30, ""),
              "bridge": ("ok", -2.047, "CO2_to_COOH")}
    rows = []
    for site, (validity, u_l, step) in values.items():
        for variant in ("a", "c"):
            shift = 0.0 if variant == "c" else 0.3
            rows.append({
                "material": MAT, "site": site, "variant": variant, "status": validity,
                "pathway_validity": validity, "dG_CO2_to_COOH_ev": -u_l + shift,
                "dG_COOH_to_CO_ev": -0.70, "dG_CO_desorption_ev": 0.10,
                "limiting_step": step if validity == "ok" else "",
                "limiting_potential_v": (u_l - shift) if validity == "ok" else "",
                "limiting_potential_if_bound_v": u_l - shift, "rank_by_limiting_potential": "",
            })
    _write_csv(analysis / "science" / "co2rr" / "co2rr_corrected_summary.csv", rows)


def _neb(analysis: Path) -> None:
    common = {"status": "converged", "barrier_resolution_eV": 0.03, "climb_activated": True, "climb_skipped": False}
    _write_csv(analysis / "science" / "neb" / "neb_barriers.csv", [
        {"pair": "top_metal_to_bridge", "initial_site": "top_metal", "final_site": "bridge", **common,
         "forward_barrier_eV": 0.06, "reverse_barrier_eV": 0.09, "reaction_energy_eV": -0.03,
         "no_internal_saddle": False, "barrier_resolved": False, "reverse_barrier_resolved": False,
         "initial_dip_eV": 0.12, "final_dip_eV": -0.01, "endpoint_not_minimum": True,
         "barrier_from_lowest_eV": 0.18},
        {"pair": "bridge_to_top_oxygen", "initial_site": "bridge", "final_site": "top_oxygen", **common,
         "forward_barrier_eV": 0.0287, "reverse_barrier_eV": 0.0427, "reaction_energy_eV": -0.014,
         "no_internal_saddle": False, "barrier_resolved": False, "reverse_barrier_resolved": True,
         "initial_dip_eV": -0.01, "final_dip_eV": -0.02, "endpoint_not_minimum": False,
         "barrier_from_lowest_eV": 0.0427},
    ])
    images = []
    for pair, initial, final, energies in (
        ("top_metal_to_bridge", "top_metal", "bridge", (0.0, 0.06, -0.12, -0.03)),
        ("bridge_to_top_oxygen", "bridge", "top_oxygen", (0.0, 0.0287, -0.014)),
    ):
        for index, energy in enumerate(energies):
            images.append({"pair": pair, "initial_site": initial, "final_site": final, "status": "converged",
                           "image_index": index, "abs_energy_eV": -150.0 + energy, "rel_energy_eV": energy})
    _write_csv(analysis / "science" / "neb" / "neb_image_energies.csv", images)


def _md(run: Path, analysis: Path, sites=SITES, seeds=("1", "2")) -> None:
    summary = []
    for site in sites:
        frames = []
        for seed in seeds:
            for k in range(6):
                desorbing = site == "top_oxygen" and seed == "2" and k >= 3
                assignment = site if k < 4 else ("other" if k == 4 else "bridge" if site != "bridge" else "top_metal")
                frames.append({
                    "seed": seed, "phase": "production", "frame_index": k, "step": 500 + 20 * k,
                    "time_fs": 20.0 * k, "nearest_site": site, "c_min_slab_distance_ang": 3.0,
                    "desorbed": 0, "c_o_min_slab_distance_ang": 5.0 if desorbing else 2.8 + 0.01 * k,
                    "desorbed_persistent": int(desorbing), "site_assignment": assignment,
                })
            frames.append(dict(frames[-1], phase="equilibration", time_fs=0.0))
            summary.append({
                "material": MAT, "site": site, "seed": seed, "status": "complete",
                "n_production_frames": 6, "fraction_production_at_start_site": 4 / 6,
                "fraction_production_desorbed_persistent": 0.5 if (site == "top_oxygen" and seed == "2") else 0.0,
            })
        _write_csv(run / "md" / MAT / site / "md_frame_analysis.csv", frames)
    summary.append({"material": MAT, "site": "bridge", "seed": "9", "status": "missing",
                    "n_production_frames": 0, "fraction_production_at_start_site": "",
                    "fraction_production_desorbed_persistent": ""})
    _write_csv(analysis / "science" / "md" / "md_trajectory_analysis_summary.csv", summary)


@pytest.fixture()
def dirs(tmp_path):
    run, analysis = tmp_path / "RUN", tmp_path / "A"
    _write_csv(run / "dft_summary.csv", _dft_rows())
    return run, analysis


@pytest.fixture()
def full(dirs):
    run, analysis = dirs
    _tddft(run)
    _co2rr(analysis)
    _neb(analysis)
    _md(run, analysis)
    return run, analysis


def _figures(run, analysis, *extra):
    return paper_figures.main([
        "--results-dir", str(run), "--analysis-dir", str(analysis), "--output-dir", str(analysis / "paper"),
        "--material", MAT, "--dpi", "40", "--tddft-sigma-ev", "0.10", *extra,
    ])


def _tables(run, analysis):
    return paper_tables.main([
        "--results-dir", str(run), "--analysis-dir", str(analysis), "--output-dir", str(analysis / "paper"),
        "--material", MAT, "--co2rr-variant", "c",
    ])


def test_figures_written_with_plotted_data(full):
    run, analysis = full
    assert _figures(run, analysis, "--md-seed", "2") == 0
    out = analysis / "paper"
    assert set(paper_figures.FIGURE_NAMES.values()) == {
        "fig03_dft_descriptors", "fig04_tddft_spectra", "fig05_co2rr_pathway", "fig06_md_distance",
        "fig07_neb_profiles",
    }
    for name in paper_figures.FIGURE_NAMES.values():
        for suffix in (".png", ".pdf", "_data.csv"):
            assert (out / f"{name}{suffix}").stat().st_size > 0, name + suffix

    fig03 = pd.read_csv(out / "fig03_dft_descriptors_data.csv")
    assert set(fig03["panel"]) == {"adsorption_energy", "c_surface_distance", "band_gap"}
    eads = fig03[fig03["panel"] == "adsorption_energy"].set_index("site")
    assert eads.loc["top_metal", "value"] == pytest.approx(0.042)
    assert eads.loc["bridge", "value"] == pytest.approx(-0.025)
    dist = fig03[fig03["panel"] == "c_surface_distance"].set_index("site")
    assert (dist["quantity"] == "c_surface_distance_ang").all()
    assert dist.loc["bridge", "value"] == pytest.approx(2.9)
    gaps = fig03[fig03["panel"] == "band_gap"].set_index(["site", "quantity"])["value"]
    assert gaps[("bridge", "clean_band_gap_gamma_ev")] == pytest.approx(1.70)
    assert gaps[("bridge", "adsorbed_band_gap_gamma_ev")] == pytest.approx(1.66)

    fig04 = pd.read_csv(out / "fig04_tddft_spectra_data.csv")
    assert set(fig04["site"]) == set(SITES)
    assert fig04["energy_ev"].min() == pytest.approx(3.0)
    assert fig04["energy_ev"].max() == pytest.approx(13.0)  # max(13 eV, last transition + 4 sigma)
    assert set(fig04["curve"]) == {"ZnO | top-metal", "ZnO | top-oxygen", "ZnO | bridge"}
    assert fig04["peak_energy_ev"].to_numpy() == pytest.approx(11.0, abs=0.01)
    assert fig04.groupby("site")["peak_absorption_au"].max().max() == pytest.approx(1.0)
    # One common normalisation factor: the global maximum is 1 and the ratio is the raw one.
    assert fig04["absorption_au"].max() == pytest.approx(1.0)
    ratio = fig04["absorption_au"] / fig04["intensity_per_ev"]
    assert ratio[fig04["intensity_per_ev"] > 1e-6].std() < 1e-9

    fig05 = pd.read_csv(out / "fig05_co2rr_pathway_data.csv").set_index(["site", "state"])
    assert list(dict.fromkeys(fig05.index.get_level_values("state"))) == ["CO2(g)", "COOH*", "CO*", "CO(g)"]
    assert (fig05["variant"] == "c").all()
    assert fig05.loc[("top_metal", "CO2(g)"), "dG_ev"] == pytest.approx(0.0)
    assert fig05.loc[("top_metal", "COOH*"), "dG_ev"] == pytest.approx(1.982)
    assert fig05.loc[("top_metal", "CO(g)"), "dG_ev"] == pytest.approx(1.982 - 0.70 + 0.10)
    assert fig05.loc[("top_metal", "COOH*"), "limiting_potential_v"] == pytest.approx(-1.982)
    assert fig05.loc[("top_oxygen", "COOH*"), "pathway_validity"] == "unbound"
    assert math.isnan(fig05.loc[("top_oxygen", "COOH*"), "limiting_potential_v"])

    fig06 = pd.read_csv(out / "fig06_md_distance_data.csv", dtype={"seed": str})
    assert len(fig06) == 3 * 6  # one seed per site, production frames only
    assert set(fig06["seed"]) == {"2"}
    assert fig06["time_ps"].min() == pytest.approx(0.0) and fig06["time_ps"].max() == pytest.approx(0.1)
    top_oxygen = fig06[fig06["site"] == "top_oxygen"]["co2_surface_distance_ang"].to_numpy()
    assert top_oxygen[-1] == pytest.approx(5.0)

    fig07 = pd.read_csv(out / "fig07_neb_profiles_data.csv")
    assert list(dict.fromkeys(fig07["pair"])) == ["top_metal_to_bridge", "bridge_to_top_oxygen"]
    assert (fig07["mode"] == "chain").all()
    tmb = fig07[fig07["pair"] == "top_metal_to_bridge"]
    assert list(tmb["image_index"]) == [0, 1, 2, 3]
    assert tmb["rel_energy_ev"].iloc[2] == pytest.approx(-0.12)
    bto = fig07[fig07["pair"] == "bridge_to_top_oxygen"]
    assert list(bto["global_index"]) == [3, 4, 5]
    assert bto["rel_energy_ev"].to_numpy() == pytest.approx([-0.03, -0.03 + 0.0287, -0.03 - 0.014])
    ends = fig07[fig07["endpoint_site"].fillna("") != ""].drop_duplicates("global_index")
    assert dict(zip(ends["global_index"], ends["endpoint_site"])) == {
        0: "top_metal", 3: "bridge", 5: "top_oxygen"}

    for data in (fig03, fig04, fig05, fig06, fig07):
        assert not any("rank" in column for column in data.columns)
    for title in paper_figures.TITLES.values():
        assert not any(word in title.lower() for word in RANKING_WORDS)


def test_figures_use_requested_variant_and_seed_fallback(full, capsys):
    run, analysis = full
    assert _figures(run, analysis, "--co2rr-variant", "a", "--md-seed", "777",
                    "--figures", "fig05", "fig06") == 0
    out = capsys.readouterr().out
    assert "seed 777 not available, using seed 1" in out
    paper = analysis / "paper"
    fig05 = pd.read_csv(paper / "fig05_co2rr_pathway_data.csv").set_index(["site", "state"])
    assert fig05.loc[("top_metal", "COOH*"), "dG_ev"] == pytest.approx(1.982 + 0.3)
    fig06 = pd.read_csv(paper / "fig06_md_distance_data.csv", dtype={"seed": str})
    assert set(fig06["seed"]) == {"1"}
    assert not (paper / "fig03_dft_descriptors.png").exists()


def test_figures_skip_missing_optional_inputs(dirs, capsys):
    run, analysis = dirs  # only dft_summary.csv: no TDDFT, CO2RR, MD or NEB
    assert _figures(run, analysis) == 0
    out = capsys.readouterr().out
    for figure in ("fig04", "fig05", "fig06", "fig07"):
        assert f"[skip] {figure}" in out
    paper = analysis / "paper"
    assert (paper / "fig03_dft_descriptors.png").exists()
    for name in ("fig04_tddft_spectra", "fig05_co2rr_pathway", "fig06_md_distance", "fig07_neb_profiles"):
        assert not (paper / f"{name}.png").exists()


def test_fig03_only_ok_rows(dirs):
    run, analysis = dirs
    rows = _dft_rows()
    rows[1]["dft_status"] = "failed"
    _write_csv(run / "dft_summary.csv", rows)
    assert _figures(run, analysis, "--figures", "fig03") == 0
    data = pd.read_csv(analysis / "paper" / "fig03_dft_descriptors_data.csv")
    assert set(data["site"]) == {"top_metal", "bridge"}


def test_figures_missing_required_input_is_an_error(tmp_path):
    run = tmp_path / "RUN"
    run.mkdir()
    with pytest.raises(SystemExit, match="dft_summary.csv"):
        _figures(run, tmp_path / "A", "--figures", "fig03")


def test_md_figure_skips_site_without_frames(full, capsys):
    run, analysis = full
    (run / "md" / MAT / "bridge" / "md_frame_analysis.csv").unlink()
    assert _figures(run, analysis, "--figures", "fig06") == 0
    assert "[skip] fig06 bridge" in capsys.readouterr().out
    data = pd.read_csv(analysis / "paper" / "fig06_md_distance_data.csv")
    assert set(data["site"]) == {"top_metal", "top_oxygen"}


def test_table02(full):
    run, analysis = full
    assert _tables(run, analysis) == 0
    paper = analysis / "paper"
    table = pd.read_csv(paper / "table02_site_summary.csv").set_index("site")
    assert list(table.index) == list(SITES)
    assert table.loc["top_metal", "adsorption_energy_ev"] == pytest.approx(0.042)
    assert table.loc["top_metal", "limiting_potential_v"] == pytest.approx(-1.982)
    assert table.loc["bridge", "limiting_potential_v"] == pytest.approx(-2.047)
    assert math.isnan(table.loc["top_oxygen", "limiting_potential_v"])
    assert table.loc["top_oxygen", "co2rr_pathway_validity"] == "unbound"
    assert "top_metal -> bridge: endpoint not a minimum" in table.loc["bridge", "neb_paths"]
    assert "bridge -> top_oxygen: fwd < 0.03 eV, rev 0.0427 eV" in table.loc["top_oxygen", "neb_paths"]
    assert table.loc["bridge", "md_n_trajectories"] == 2  # the missing seed is not counted
    assert table.loc["top_oxygen", "md_fraction_desorbed_persistent_mean"] == pytest.approx(0.25)
    assert table.loc["top_metal", "tddft_first_excitation_ev"] == pytest.approx(8.18)
    assert not any("rank" in column for column in table.columns)

    md = (paper / "table02_site_summary.md").read_text()
    tex = (paper / "table02_site_summary.tex").read_text()
    assert "+0.042" in md and "-1.98" in md and "unbound" in md and "0.25 (n=2)" in md
    assert "< 0.03 eV" in md and "8.180" in md
    assert "~0.05 eV" in md and "~0.1 V" in md
    assert r"\toprule" in tex and r"\bottomrule" in tex and r"top\_metal" in tex
    for text in (md.lower(), tex.lower()):
        assert not any(word in text for word in RANKING_WORDS)


def test_table02_optional_inputs_missing(dirs, capsys):
    run, analysis = dirs
    assert _tables(run, analysis) == 0
    out = capsys.readouterr().out
    assert out.count("[skip]") == 4
    table = pd.read_csv(analysis / "paper" / "table02_site_summary.csv")
    assert table["limiting_potential_v"].isna().all()
    assert "n/a" in (analysis / "paper" / "table02_site_summary.md").read_text()


def test_table02_missing_dft_summary_is_an_error(tmp_path):
    run = tmp_path / "RUN"
    run.mkdir()
    with pytest.raises(SystemExit, match="dft_summary.csv"):
        _tables(run, tmp_path / "A")


def test_desorbed_frames_do_not_count_as_start_site(tmp_path):
    """site_assignment is lateral: a CO2 desorbed above its start site must not count as there."""
    from scripts.analysis.paper_tables import md_for, occupancy_fractions

    fractions = occupancy_fractions(
        ["top_metal", "top_metal", "top_metal", "bridge", "other"], [1, 1, 0, 0, 0], "top_metal"
    )
    assert fractions == pytest.approx(
        {"start site": 0.2, "other sites": 0.2, "other": 0.2, "desorbed": 0.4}
    )
    assert sum(fractions.values()) == pytest.approx(1.0)

    site_dir = tmp_path / "RUN" / "md" / "ZnO" / "top_metal"
    site_dir.mkdir(parents=True)
    pd.DataFrame({
        "seed": ["1"] * 4,
        "phase": ["production"] * 4,
        "site_assignment": ["top_metal"] * 4,
        "desorbed_persistent": [0, 1, 1, 1],
    }).to_csv(site_dir / "md_frame_analysis.csv", index=False)
    summary = [{"material": "ZnO", "site": "top_metal", "seed": "1", "status": "complete",
                "fraction_production_at_start_site": "1.0",
                "fraction_production_desorbed_persistent": "0.75"}]
    n, at_start, desorbed = md_for(summary, "ZnO", "top_metal", tmp_path / "RUN")
    assert (n, at_start, desorbed) == (1, pytest.approx(0.25), pytest.approx(0.75))
    # Without the frame file the summary's lateral fraction is the fallback.
    assert md_for(summary, "ZnO", "top_metal", tmp_path / "none")[1] == pytest.approx(1.0)


# ---------------------------------------------------------------- fig03 / fig04 / fig05 / fig07 details


def test_fig03_symmetric_adsorption_axis(dirs):
    run, analysis = dirs
    data = paper_figures.fig03_data(run, MAT)
    fig = paper_figures.plot_fig03(data, MAT)
    ax = fig.axes[0]
    limit = max(0.1, 1.2 * max(abs(v) for v in EADS.values()))
    assert ax.get_ylim() == pytest.approx((-limit, limit))
    texts = {t.get_text(): t for t in ax.texts}
    assert texts["+0.042"].get_va() == "bottom" and texts["-0.025"].get_va() == "top"
    big = {site: value * 10 for site, value in EADS.items()}
    rows = _dft_rows()
    for row in rows:
        row["adsorption_energy_ev"] = big[row["site"]]
    _write_csv(run / "dft_summary.csv", rows)
    fig = paper_figures.plot_fig03(paper_figures.fig03_data(run, MAT), MAT)
    assert fig.axes[0].get_ylim() == pytest.approx((-1.2 * 0.42, 1.2 * 0.42))


def test_fig04_layout(full):
    run, _analysis = full
    data = paper_figures.fig04_data(run, MAT, 0.10)
    fig = paper_figures.plot_fig04(data)
    ax = fig.axes[0]
    assert ax.get_xlim() == pytest.approx((3.0, 13.0))
    assert ax.get_ylabel() == "Normalized absorption (a.u.)"
    texts = [t.get_text() for t in ax.texts]
    assert "UV (400–200 nm)" in texts and "Vacuum UV (< 200 nm)" in texts
    assert "top-metal\n11.00 eV" in texts
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["ZnO | top-metal", "ZnO | top-oxygen", "ZnO | bridge"]
    top = [child for child in ax.get_children() if child.__class__.__name__ == "SecondaryAxis"]
    assert top and list(top[0].get_xticks()) == [400, 300, 200, 150, 120, 100]


def test_fig04_window_extends_past_13_ev():
    spectra = {"top_metal": pd.DataFrame({"energy_ev": [9.0, 14.0]})}
    assert paper_figures.fig04_window(spectra, 0.1) == pytest.approx((3.0, 14.4))


def test_fig05_legend_labels(full):
    _run, analysis = full
    rows = list(csv.DictReader((analysis / "science/co2rr/co2rr_corrected_summary.csv").open()))
    for row in rows:
        if row["site"] == "bridge":
            row["pathway_validity"] = row["status"] = "wrong_intermediate"
            row["limiting_potential_v"] = ""
    _write_csv(analysis / "science/co2rr/co2rr_corrected_summary.csv", rows)
    fig = paper_figures.plot_fig05(paper_figures.fig05_data(analysis, MAT, "c"))
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Top metal (U$_\\mathrm{L}$ = -1.98 V)", "Top oxygen (intermediates desorbed)",
                      "Bridge (not COOH*/CO*)"]
    colors = [line.get_color() for line in ax.get_lines()[:3]]
    assert colors == [paper_figures.SITE_COLORS[s] for s in SITES]
    assert all(line.get_linestyle() == "-" and line.get_marker() == "o" for line in ax.get_lines()[:3])


def _lab_neb_table(reverse_second: bool = False) -> pd.DataFrame:
    rows = []
    first = (0.0, -0.0335, -0.0953, -0.1027, -0.0664)
    second = (0.0, 0.0003, 0.0130, 0.0287, 0.0136)
    for k, value in enumerate(first):
        rows.append({"pair": "top_metal_to_bridge", "initial_site": "top_metal", "final_site": "bridge",
                     "image_index": k, "rel_energy_eV": value})
    if reverse_second:  # stored as top_oxygen -> bridge
        stored = [v - second[-1] for v in reversed(second)]
        for k, value in enumerate(stored):
            rows.append({"pair": "top_oxygen_to_bridge", "initial_site": "top_oxygen", "final_site": "bridge",
                         "image_index": k, "rel_energy_eV": value})
    else:
        for k, value in enumerate(second):
            rows.append({"pair": "bridge_to_top_oxygen", "initial_site": "bridge", "final_site": "top_oxygen",
                         "image_index": k, "rel_energy_eV": value})
    return pd.DataFrame(rows)


LAB_CHAIN = [0.0, -0.0335, -0.0953, -0.1027, -0.0664, -0.0664, -0.0661, -0.0534, -0.0377, -0.0528]


@pytest.mark.parametrize("reverse_second", [False, True])
def test_fig07_chain_with_lab_energies(reverse_second):
    data = paper_figures.fig07_frame(_lab_neb_table(reverse_second))
    assert (data["mode"] == "chain").all()
    assert list(data["global_index"]) == [0, 1, 2, 3, 4, 4, 5, 6, 7, 8]
    assert data["rel_energy_ev"].to_numpy() == pytest.approx(LAB_CHAIN, abs=1e-9)
    assert list(dict.fromkeys(data["segment"])) == ["top metal → bridge", "bridge → top oxygen"]
    assert list(data["reversed"]) == [False] * 5 + [reverse_second] * 5
    ends = data[data["endpoint_site"].fillna("") != ""].drop_duplicates("global_index")
    assert dict(zip(ends["global_index"], ends["endpoint_site"])) == {0: "top_metal", 4: "bridge", 8: "top_oxygen"}
    fig = paper_figures.plot_fig07(data)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Energy relative to top metal (eV)"
    assert ax.get_xlabel() == "Migration path along the Zn–O dimer (NEB images)"
    assert {t.get_text() for t in ax.texts} == {"top metal", "bridge", "top oxygen"}
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["top metal → bridge", "bridge → top oxygen"]


def test_fig07_unchainable_pairs_share_the_axes(capsys):
    table = _lab_neb_table()
    table = table[table["pair"] == "bridge_to_top_oxygen"].copy()
    extra = table.copy()
    extra["pair"], extra["initial_site"], extra["final_site"] = "top_metal_to_top_oxygen", "top_metal", "top_oxygen"
    data = paper_figures.fig07_frame(pd.concat([table, extra]))
    assert "[skip] fig07 chain: no NEB pair for top_metal -> bridge" in capsys.readouterr().out
    assert (data["mode"] == "separate").all()
    for _pair, part in data.groupby("pair"):
        assert part["rel_energy_ev"].iloc[0] == pytest.approx(0.0)
        assert list(part["global_index"]) == [0, 1, 2, 3, 4]
    fig = paper_figures.plot_fig07(data)
    assert len(fig.axes) == 1 and len(fig.axes[0].get_legend().get_texts()) == 2
