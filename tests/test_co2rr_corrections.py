"""Post-processing corrections of the CO2RR free energies on a synthetic run (no DFT)."""

import contextlib
import csv
import io
import json
import math
from argparse import Namespace
from pathlib import Path

import pytest
from ase.io import write
from ase.units import invcm

from scripts.analysis import co2rr_corrections as corr
from scripts.simulations.co2rr import run_co2rr_pathways as co2rr

PROFILE = "base"
SITES = ("top_metal", "top_oxygen", "bridge")
E = {
    "*": -100.0, "CO2*": -123.05, "COOH*": -126.10, "CO*": -115.40,
    "CO2": -22.95, "CO": -14.80, "H2O": -14.20, "H2": -6.75,
}
GAS_VIB = {
    "CO2": [0.0] * 5 + [0.08, 0.08, 0.165, 0.29], "CO": [0.0] * 5 + [0.265],
    "H2O": [0.0] * 6 + [0.198, 0.455, 0.468], "H2": [0.0] * 5 + [0.54],
}
LOW = 20.0 * invcm  # real mode below the 50 cm^-1 floor


def adsorbate_vib(site):
    """COOH* carries two imaginary modes on every site; top_oxygen also a 20 cm^-1 mode."""
    cooh = [0.004j, 0.002j] + [0.02 + 0.03 * i for i in range(10)]
    if site == "top_oxygen":
        cooh[2] = LOW
    return {
        "CO2*": [0.015 + 0.03 * i for i in range(9)],
        "COOH*": cooh,
        "CO*": [0.012, 0.02, 0.03, 0.04, 0.05, 0.26],
    }


def site_energies(site):
    shift = {"top_metal": 0.0, "top_oxygen": 0.11, "bridge": -0.07}[site]
    return {"CO2*": E["CO2*"] + shift, "COOH*": E["COOH*"] + 2 * shift, "CO*": E["CO*"] - shift}


def _encode(values):
    return [[complex(v).real, complex(v).imag] for v in values]


def _write_energy(folder: Path, energy, vib=None, n_atoms=3):
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": {"profile": PROFILE}, "energy_ev": energy, "source": "test"}
    if vib is not None:
        payload["vibrations"] = {"delta_ang": 0.01, "indices": list(range(n_atoms)), "energies_ev": _encode(vib)}
    (folder / "energy.json").write_text(json.dumps(payload))


def _molecule(name):
    atoms = co2rr.molecule_atoms(name)
    atoms.center(vacuum=5.0)
    return atoms


def _summary_row(site, flags):
    """Row exactly as run_co2rr_pathways.build_result_row writes it."""
    vib = adsorbate_vib(site)
    energies = site_energies(site)

    def term(species, energy, vib=None, atoms=None):
        return co2rr.TermResult(
            term=species, profile=PROFILE, energy_ev=energy, atoms=atoms, source="test",
            metadata={"n_slab_atoms": 16, "cell_hash": "abc"}, vib_energies=vib,
        )

    terms = {"*": term("*", E["*"])}
    for species in ("CO2*", "COOH*", "CO*"):
        terms[species] = term(species, energies[species], vib[species])
    for name in co2rr.GAS_SPECIES:
        terms[name] = term(name, E[name], GAS_VIB[name], _molecule(name))
    settings = {k: (dict(v) if isinstance(v, dict) else v) for k, v in co2rr.DEFAULT_THERMO_SETTINGS.items()}
    settings["vibrations"] = True
    settings["electrode_potential_v_rhe"] = -0.2  # summary at U != 0: the check must undo it
    return co2rr.build_result_row(
        material="ZnO", site=site, terms=terms, settings=settings,
        args=Namespace(ecut=500.0, kpts=(2, 3, 1), fmax=0.05), flags=flags, distances={},
    )


@pytest.fixture
def run_dir(tmp_path):
    pathways = tmp_path / "pathways"
    _write_energy(pathways / "ZnO" / "_clean" / PROFILE, E["*"])
    for name in co2rr.GAS_SPECIES:
        folder = pathways / "_references" / PROFILE / name.lower()
        _write_energy(folder, E[name], GAS_VIB[name])
        write(folder / f"{name.lower()}_optimized.xyz", _molecule(name))
    rows = []
    for site in SITES:
        vib = adsorbate_vib(site)
        energies = site_energies(site)
        for species, folder_name in (("CO2*", "co2_star"), ("COOH*", "cooh"), ("CO*", "co")):
            _write_energy(pathways / "ZnO" / site / PROFILE / folder_name, energies[species], vib[species])
        rows.append(_summary_row(site, ["cooh_desorbed"] if site == "bridge" else []))
    fieldnames = []
    for row in rows:
        fieldnames += [key for key in row if key not in fieldnames]
    with (tmp_path / "co2rr_pathway_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def _run(run_dir, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rows = corr.run_corrections(run_dir, **kwargs)
    return {(row["site"], row["variant"]): row for row in rows}, out.getvalue()


def test_floor_counts_imaginary_and_low_modes():
    floor = 50.0 * invcm
    modes, count = corr.floor_vib_energies([0.003j, LOW, 0.1, complex(0.2, 0.0)], floor)
    assert count == 2
    assert modes[:2] == [complex(floor), complex(floor)]
    assert modes[2:] == [complex(0.1), complex(0.2)]


def test_variant_a_reproduces_run_summary(run_dir):
    _, output = _run(run_dir)
    deviation = float(output.split("max |difference| = ")[1].split()[0])
    assert deviation < 1e-9


def test_variant_b_flooring_count_and_sign(run_dir):
    rows, _ = _run(run_dir)
    assert rows[("top_metal", "a")]["n_modes_floored"] == 0
    assert rows[("top_metal", "a")]["n_imag_modes_removed"] == 2
    # top_metal: two imaginary COOH* modes, plus the CO2*/CO* modes below 50 cm^-1 (0.012 eV ~ 97 cm^-1
    # and 0.015 eV ~ 121 cm^-1 are above it, so none of those).
    assert rows[("top_metal", "b")]["n_modes_floored"] == 2
    assert rows[("top_oxygen", "b")]["n_modes_floored"] == 3
    assert rows[("top_metal", "b")]["n_imag_modes_removed"] == 0
    # A floored imaginary mode adds a soft real mode: G(COOH*) and dG(CO2 -> COOH*) go down.
    assert rows[("top_metal", "b")]["dG_CO2_to_COOH_ev"] < rows[("top_metal", "a")]["dG_CO2_to_COOH_ev"]
    assert rows[("top_metal", "b")]["dG_COOH_to_CO_ev"] > rows[("top_metal", "a")]["dG_COOH_to_CO_ev"]
    # Gases are unchanged by the floor.
    assert rows[("top_metal", "b")]["dG_overall_ev"] == pytest.approx(rows[("top_metal", "a")]["dG_overall_ev"])


def test_variant_c_shifts_overall_by_gas_corrections(run_dir):
    rows, output = _run(run_dir)
    for site in SITES:
        shift = rows[(site, "c")]["dG_overall_ev"] - rows[(site, "b")]["dG_overall_ev"]
        assert shift == pytest.approx(-0.43)
        assert rows[(site, "c")]["dG_CO2_to_COOH_ev"] - rows[(site, "b")]["dG_CO2_to_COOH_ev"] == pytest.approx(-0.19)
        assert rows[(site, "c")]["dG_CO_desorption_ev"] - rows[(site, "b")]["dG_CO_desorption_ev"] == pytest.approx(-0.24)
        assert rows[(site, "c")]["gas_correction_CO2_ev"] == 0.19
    assert "experiment +0.20 eV" in output


def test_overall_identical_across_sites(run_dir):
    rows, _ = _run(run_dir)
    for variant in corr.VARIANTS:
        values = [rows[(site, variant)]["dG_overall_ev"] for site in SITES]
        assert max(values) - min(values) < 1e-9
    broken = [dict(rows[("top_metal", "a")]), dict(rows[("bridge", "a")])]
    broken[1]["dG_overall_ev"] += 0.01
    with pytest.raises(AssertionError):
        corr.check_overall(broken)


def test_unbound_sites_are_listed_but_not_ranked(run_dir):
    rows, _ = _run(run_dir)
    for variant in corr.VARIANTS:
        unbound = rows[("bridge", variant)]
        assert unbound["status"] == "unbound"
        assert math.isnan(unbound["limiting_potential_v"])
        assert math.isfinite(unbound["limiting_potential_if_bound_v"])
        assert unbound["rank_by_limiting_potential"] == ""
        ranks = sorted(rows[(site, variant)]["rank_by_limiting_potential"] for site in ("top_metal", "top_oxygen"))
        assert ranks == [1, 2]
    with (run_dir / corr.CORRECTED_CSV).open() as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 3 * len(corr.VARIANTS)
    assert {"variant", "site", "status", "dG_overall_ev", "limiting_potential_v", "n_modes_floored"} <= set(written[0])
    assert "Granda-Marulanda" in (run_dir / corr.NOTE_MD).read_text()


def test_missing_checkpoint_is_reported(run_dir):
    (run_dir / "pathways" / "ZnO" / "top_oxygen" / PROFILE / "cooh" / "energy.json").unlink()
    rows, _ = _run(run_dir)
    assert rows[("top_oxygen", "c")]["status"] == "missing_checkpoint"
    assert rows[("top_metal", "c")]["status"] == "ok"


def test_gas_correction_cli():
    args = corr.parse_args([])
    assert corr.parse_gas_corrections(args.gas_correction) == {"CO2": 0.19, "CO": -0.24}
    assert args.low_mode_cm == 50.0
    assert corr.parse_gas_corrections(["CO=-0.1"]) == {"CO2": 0.0, "CO": -0.1}
    with pytest.raises(SystemExit):
        corr.parse_gas_corrections(["H2=0.1"])


def test_plot_uses_corrected_variant_and_greys_unbound(run_dir):
    from scripts.visualization import plot_co2rr_pathway as plot

    _run(run_dir)
    out = run_dir / "plots"
    with contextlib.redirect_stdout(io.StringIO()):
        plot.main(["--summary", str(run_dir / "co2rr_pathway_summary.csv"), "--output-dir", str(out), "--variant", "b"])
    with (out / "co2rr_pathway_profile_table.csv").open() as handle:
        table = list(csv.DictReader(handle))
    assert {row["source"] for row in table} == {"corrected variant b"}
    assert {row["site"] for row in table if row["unbound"] == "True"} == {"bridge"}
    assert "relative_free_energy_ev" in table[0]
    assert (out / "co2rr_pathway.png").exists()

    rows = corr.load_corrected_pathway_rows(run_dir / corr.CORRECTED_CSV, "c")
    profiles, basis = plot.site_profiles(rows, "ZnO", include_unbound=False)
    assert basis == "free_energy_CHE"
    assert [item["site"] for item in profiles] == ["top_metal", "top_oxygen"]
    # Uncorrected summary: the desorbed site is still recognised from geometry_flags.
    with (run_dir / "co2rr_pathway_summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    profiles, _ = plot.site_profiles(summary, "ZnO")
    assert [item["unbound"] for item in profiles] == [False, False, True]
    assert math.isnan(profiles[2]["limiting_potential_v"])


def test_analysis_table_reads_corrected_rows(run_dir):
    from scripts.analysis import analysis_co2_reduction as analysis

    rows, _ = _run(run_dir)
    merged = [{"material": "ZnO", "site": site, "quality_flag": "ok"} for site in SITES]
    corrected = corr.load_corrected_pathway_rows(run_dir / corr.CORRECTED_CSV, "c")
    table = {row["site"]: row for row in analysis.build_pathway_table(merged, corrected)}
    assert table["top_metal"]["dG_CO2_to_COOH_ev"] == pytest.approx(rows[("top_metal", "c")]["dG_CO2_to_COOH_ev"])
    assert table["top_metal"]["co2rr_variant"] == "c"
    assert table["bridge"]["pathway_status"] == "unbound"
    assert math.isfinite(table["bridge"]["dG_CO2_to_COOH_ev"])
    assert math.isnan(table["bridge"]["limiting_potential_v"])
    assert "excluded from U_L" in table["bridge"]["notes"]
    with pytest.raises(SystemExit):
        corr.load_corrected_pathway_rows(run_dir / corr.CORRECTED_CSV, "z")
    assert corr.load_corrected_pathway_rows(run_dir / "missing.csv") is None


# --------------------------------------------------------------------------- validity

WRONG_FLAGS = ("cooh_c_bonded_to_lattice_o", "cooh_dissociated", "cooh_deprotonated", "co_dissociated")


def _write_summary(run_dir, flags_by_site):
    rows = [_summary_row(site, flags_by_site.get(site, [])) for site in SITES]
    fieldnames = []
    for row in rows:
        fieldnames += [key for key in row if key not in fieldnames]
    with (run_dir / "co2rr_pathway_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


@pytest.mark.parametrize("flag", WRONG_FLAGS)
def test_pathway_validity_wrong_intermediate_per_flag(flag):
    assert co2rr.pathway_validity(flag) == "wrong_intermediate"
    assert co2rr.pathway_validity([flag]) == "wrong_intermediate"
    # Takes precedence over a desorbed flag.
    assert co2rr.pathway_validity(f"co_desorbed;{flag}") == "wrong_intermediate"
    assert not corr.is_unbound(flag)


def test_pathway_validity_unbound_and_ok():
    assert co2rr.pathway_validity("cooh_desorbed") == "unbound"
    assert co2rr.pathway_validity(["co_desorbed"]) == "unbound"
    assert co2rr.pathway_validity("") == "ok"
    assert co2rr.pathway_validity("o_star_peroxo") == "ok"  # O* is not in the profile


@pytest.mark.parametrize("flag", WRONG_FLAGS)
def test_summary_row_keeps_status_ok_but_withholds_u_l(flag):
    """pathway_status drives resume/retry and stays ok; validity is a separate column."""
    row = _summary_row("top_metal", [flag])
    assert row["pathway_status"] == "ok"
    assert row["pathway_validity"] == "wrong_intermediate"
    assert math.isnan(row["limiting_potential_v"]) and row["limiting_step"] == ""
    assert math.isfinite(row["limiting_potential_if_bound_v"])
    good = _summary_row("top_metal", [])
    assert good["pathway_validity"] == "ok"
    assert good["limiting_potential_v"] == good["limiting_potential_if_bound_v"]


@pytest.mark.parametrize("flag", WRONG_FLAGS)
def test_wrong_intermediate_sites_are_not_ranked(run_dir, flag):
    _write_summary(run_dir, {"top_oxygen": [flag], "bridge": ["cooh_desorbed"]})
    rows, _ = _run(run_dir)
    for variant in corr.VARIANTS:
        wrong = rows[("top_oxygen", variant)]
        assert wrong["status"] == wrong["pathway_validity"] == "wrong_intermediate"
        assert math.isnan(wrong["limiting_potential_v"])
        assert math.isfinite(wrong["limiting_potential_if_bound_v"])
        assert wrong["rank_by_limiting_potential"] == "" and wrong["limiting_step"] == ""
        assert rows[("bridge", variant)]["status"] == "unbound"
        assert rows[("top_metal", variant)]["rank_by_limiting_potential"] == 1
    assert "wrong_intermediate" in (run_dir / corr.NOTE_MD).read_text()

    from scripts.analysis import analysis_co2_reduction as analysis

    merged = [{"material": "ZnO", "site": site, "quality_flag": "ok"} for site in SITES]
    corrected = corr.load_corrected_pathway_rows(run_dir / corr.CORRECTED_CSV, "c")
    table = {row["site"]: row for row in analysis.build_pathway_table(merged, corrected)}
    assert table["top_oxygen"]["pathway_status"] == "wrong_intermediate"
    assert math.isnan(table["top_oxygen"]["limiting_potential_v"])
    assert "wrong intermediate" in table["top_oxygen"]["notes"]

    # --uncorrected: the run summary (pathway_status ok) must be honoured as well.
    with (run_dir / "co2rr_pathway_summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    table = {row["site"]: row for row in analysis.build_pathway_table(merged, summary)}
    assert table["top_oxygen"]["pathway_status"] == "ok"
    assert table["top_oxygen"]["pathway_validity"] == "wrong_intermediate"
    assert math.isnan(table["top_oxygen"]["limiting_potential_v"])
    assert math.isfinite(table["top_oxygen"]["limiting_potential_if_bound_v"])
    assert table["bridge"]["pathway_validity"] == "unbound"
    assert table["top_metal"]["pathway_validity"] == "ok"
    ranked = analysis.attach_limiting_potentials(
        [dict(row) for row in merged], list(table.values())
    )
    assert [(row["site"], row["rank_by_limiting_potential"]) for row in ranked] == [
        ("top_metal", 1), ("top_oxygen", ""), ("bridge", ""),
    ]

    # Old summaries without pathway_validity: validity derived from the flags.
    for row in summary:
        row.pop("pathway_validity")
    table = {row["site"]: row for row in analysis.build_pathway_table(merged, summary)}
    assert table["top_oxygen"]["pathway_validity"] == "wrong_intermediate"
    assert math.isnan(table["top_oxygen"]["limiting_potential_v"])


def test_plot_marks_wrong_intermediate(run_dir):
    from scripts.visualization import plot_co2rr_pathway as plot

    _write_summary(run_dir, {"top_oxygen": ["cooh_deprotonated"], "bridge": ["cooh_desorbed"]})
    _run(run_dir)
    rows = corr.load_corrected_pathway_rows(run_dir / corr.CORRECTED_CSV, "c")
    profiles, _ = plot.site_profiles(rows, "ZnO")
    by_site = {item["site"]: item for item in profiles}
    assert by_site["top_oxygen"]["validity"] == "wrong_intermediate"
    assert not by_site["top_oxygen"]["unbound"]
    assert math.isnan(by_site["top_oxygen"]["limiting_potential_v"])
    assert by_site["bridge"]["validity"] == "unbound"
    assert math.isfinite(by_site["top_metal"]["limiting_potential_v"])
    hidden, _ = plot.site_profiles(rows, "ZnO", include_unbound=False)
    assert [item["site"] for item in hidden] == ["top_metal"]
    # --uncorrected summary: recognised from pathway_validity / geometry_flags.
    with (run_dir / "co2rr_pathway_summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    profiles, _ = plot.site_profiles(summary, "ZnO")
    assert [item["validity"] for item in profiles] == ["ok", "wrong_intermediate", "unbound"]
    out = run_dir / "plots_wrong"
    with contextlib.redirect_stdout(io.StringIO()):
        plot.main(["--summary", str(run_dir / "co2rr_pathway_summary.csv"), "--output-dir", str(out)])
    with (out / "co2rr_pathway_profile_table.csv").open() as handle:
        table = list(csv.DictReader(handle))
    assert {row["validity"] for row in table if row["site"] == "top_oxygen"} == {"wrong_intermediate"}


def test_note_states_floor_choice_and_gas_only_correction(run_dir):
    _run(run_dir)
    note = (run_dir / corr.NOTE_MD).read_text()
    assert "usual quasi-harmonic" not in note
    assert "100 cm^-1" in note and "--low-mode-cm 100" in note
    assert "CO2(g) and CO(g) only" in note
    assert "-0.19 eV" in note and "-0.24 eV" in note
