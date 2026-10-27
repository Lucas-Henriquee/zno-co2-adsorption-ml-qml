"""CHE arithmetic, stoichiometry, limiting potential and column naming (no DFT)."""

import math
from argparse import Namespace

import numpy as np
import pytest
from ase import Atoms
from ase.thermochemistry import HarmonicThermo, IdealGasThermo

from scripts.simulations.co2rr import run_co2rr_pathways as co2rr
from scripts.visualization import plot_co2rr_pathway as plot

# Synthetic G (eV) per species; values chosen so every step is distinct.
SYNTHETIC = {
    "*": -100.0, "CO2*": -123.05, "COOH*": -126.10, "CO*": -115.40, "O*": -105.0,
    "CO2": -22.95, "CO": -14.80, "H2O": -14.20, "H2": -6.75,
}


def test_each_step_is_atom_and_electron_balanced():
    for step, coefficients in co2rr.STEP_STOICHIOMETRY.items():
        balance: dict[str, int] = {}
        for species, coefficient in coefficients.items():
            for element, count in co2rr.SPECIES_COMPOSITION[species].items():
                balance[element] = balance.get(element, 0) + coefficient * count
        assert all(value == 0 for value in balance.values()), (step, balance)
        # Surface sites are conserved too.
        sites = sum(c for s, c in coefficients.items() if s.endswith("*"))
        assert sites == 0, step


def test_pathway_steps_sum_to_overall_reaction():
    total: dict[str, int] = {}
    for step in co2rr.PATHWAY_STEPS:
        for species, coefficient in co2rr.STEP_STOICHIOMETRY[step].items():
            total[species] = total.get(species, 0) + coefficient
    total = {species: value for species, value in total.items() if value}
    assert total == co2rr.STEP_STOICHIOMETRY["overall"]
    for potential in (0.0, -0.7):
        values = co2rr.che_reaction_energies(SYNTHETIC, potential)
        assert math.isclose(sum(values[s] for s in co2rr.PATHWAY_STEPS), values["overall"], abs_tol=1e-12)


def test_che_arithmetic_explicit():
    g, h2 = SYNTHETIC, SYNTHETIC["H2"]
    u = -0.5
    values = co2rr.che_reaction_energies(g, u)
    assert values["CO2_to_COOH"] == pytest.approx(g["COOH*"] - g["*"] - g["CO2"] - 0.5 * h2 + u)
    assert values["COOH_to_CO"] == pytest.approx(g["CO*"] + g["H2O"] - g["COOH*"] - 0.5 * h2 + u)
    assert values["CO_desorption"] == pytest.approx(g["*"] + g["CO"] - g["CO*"])
    assert values["CO2_adsorption"] == pytest.approx(g["CO2*"] - g["*"] - g["CO2"])
    assert values["overall"] == pytest.approx(g["CO"] + g["H2O"] - g["CO2"] - h2 + 2 * u)
    assert values["O_formation_from_H2O"] == pytest.approx(g["O*"] + h2 - 2 * u - g["*"] - g["H2O"])
    # Pathway starts from * + CO2(g): CO2* never enters the PCET steps.
    assert "CO2*" not in co2rr.STEP_STOICHIOMETRY["CO2_to_COOH"]


def test_limiting_potential_uses_pcet_steps_only_with_rhe_sign():
    steps = {"CO2_to_COOH": 0.62, "COOH_to_CO": 0.10, "CO_desorption": 1.5, "O_formation_from_H2O": 3.0}
    u_l, step = co2rr.limiting_potential(steps)
    assert step == "CO2_to_COOH"
    assert u_l == pytest.approx(-0.62)
    # All-downhill PCET steps give a positive U_L (no max(0, .) clamp).
    u_l, step = co2rr.limiting_potential({"CO2_to_COOH": -0.3, "COOH_to_CO": -0.1, "CO_desorption": 2.0})
    assert (u_l, step) == (pytest.approx(0.1), "COOH_to_CO")
    assert set(co2rr.PCET_STEPS) == {"CO2_to_COOH", "COOH_to_CO"}
    assert all(co2rr.electrons_transferred(s) == 1 for s in co2rr.PCET_STEPS)
    assert co2rr.electrons_transferred("CO_desorption") == 0


def test_o_star_not_in_profile():
    assert all("O*" not in co2rr.STEP_STOICHIOMETRY[s] for s in co2rr.PATHWAY_STEPS + ("overall",))
    assert "O_formation_from_H2O" not in co2rr.PATHWAY_STEPS
    assert len(co2rr.PATHWAY_STATES) == len(co2rr.PATHWAY_STEPS) + 1
    assert co2rr.PATHWAY_STATES[0] == "* + CO2(g)"
    assert all("O*" not in state.split() for state in co2rr.PATHWAY_STATES)


def test_column_prefix_depends_on_vibrations():
    assert co2rr.energy_prefix(True) == "dG"
    assert co2rr.energy_prefix(False) == "dE"
    off = co2rr.summary_step_columns(False, o_star=False)
    assert off and all(c.startswith("dE_") for c in off)
    on = co2rr.summary_step_columns(True, o_star=True)
    assert all(c.startswith("dG_") for c in on)
    assert "dG_O_formation_from_H2O_ev" in on


def _term(species, energy, vib=None, atoms=None):
    return co2rr.TermResult(
        term=species, profile="base", energy_ev=energy, atoms=atoms, source="test",
        metadata={"n_slab_atoms": 16, "cell_hash": "abc"}, vib_energies=vib,
    )


def _args():
    return Namespace(ecut=500.0, kpts=(2, 3, 1), fmax=0.05)


def _settings(vibrations):
    settings = {k: (dict(v) if isinstance(v, dict) else v) for k, v in co2rr.DEFAULT_THERMO_SETTINGS.items()}
    settings["vibrations"] = vibrations
    return settings


def test_electronic_only_row_has_no_dg_columns():
    terms = {name: _term(name, value) for name, value in SYNTHETIC.items() if name != "O*"}
    row = co2rr.build_result_row(
        material="ZnO", site="top_metal", terms=terms, settings=_settings(False), args=_args(),
        flags=[], distances={},
    )
    assert row["energy_basis"] == "electronic_energies_only"
    assert "electronic energies only" in row["notes"]
    assert not any(key.startswith("dG_") or key.startswith("deltaG") for key in row)
    expected = co2rr.che_reaction_energies(SYNTHETIC, 0.0)
    assert row["dE_CO2_to_COOH_ev"] == pytest.approx(expected["CO2_to_COOH"])
    assert row["limiting_potential_v"] == pytest.approx(-max(expected["CO2_to_COOH"], expected["COOH_to_CO"]))
    assert "dE_O_formation_from_H2O_ev" not in row


def _molecule(name):
    atoms = co2rr.molecule_atoms(name)
    atoms.center(vacuum=5.0)
    return atoms


def test_free_energy_row_matches_ase_thermochemistry():
    rng = np.random.default_rng(0)
    vib = {
        "CO2*": list(rng.uniform(0.01, 0.3, 9)), "COOH*": list(rng.uniform(0.01, 0.45, 12)),
        "CO*": list(rng.uniform(0.01, 0.26, 6)),
        "CO2": [0.0] * 5 + [0.08, 0.08, 0.165, 0.29], "CO": [0.0] * 5 + [0.265],
        "H2O": [0.0] * 6 + [0.198, 0.455, 0.468], "H2": [0.0] * 5 + [0.54],
    }
    terms = {"*": _term("*", SYNTHETIC["*"])}
    for name in ("CO2*", "COOH*", "CO*"):
        terms[name] = _term(name, SYNTHETIC[name], vib[name])
    for name in co2rr.GAS_SPECIES:
        terms[name] = _term(name, SYNTHETIC[name], vib[name], _molecule(name))
    settings = _settings(True)
    settings["gas_corrections_ev"] = {"CO2": 0.1, "CO": -0.05}
    settings["electrode_potential_v_rhe"] = -0.3
    row = co2rr.build_result_row(
        material="ZnO", site="bridge", terms=terms, settings=settings, args=_args(), flags=[], distances={},
    )
    temperature = 298.15
    g = {"*": SYNTHETIC["*"]}
    for name in ("CO2*", "COOH*", "CO*"):
        thermo = HarmonicThermo(vib[name], potentialenergy=SYNTHETIC[name], ignore_imag_modes=True)
        g[name] = thermo.get_helmholtz_energy(temperature, verbose=False)
    for name, spec in co2rr.GAS_SPECIES.items():
        thermo = IdealGasThermo(
            vib[name], spec["geometry"], potentialenergy=SYNTHETIC[name], atoms=_nonperiodic(terms[name].atoms),
            symmetrynumber=spec["symmetrynumber"], spin=spec["spin"], ignore_imag_modes=True,
        )
        g[name] = thermo.get_gibbs_energy(temperature, settings["partial_pressures_bar"][name] * 1e5, verbose=False)
    g["CO2"] += 0.1
    g["CO"] -= 0.05
    u = -0.3
    assert row["energy_basis"] == "free_energy_CHE"
    assert row["dG_CO2_to_COOH_ev"] == pytest.approx(g["COOH*"] - g["*"] - g["CO2"] - 0.5 * g["H2"] + u)
    assert row["dG_COOH_to_CO_ev"] == pytest.approx(g["CO*"] + g["H2O"] - g["COOH*"] - 0.5 * g["H2"] + u)
    assert row["dG_CO_desorption_ev"] == pytest.approx(g["*"] + g["CO"] - g["CO*"])
    assert row["p_H2O_bar"] == pytest.approx(0.035)
    # U_L is evaluated at U = 0 and does not depend on the configured U.
    dg1_0 = row["dG_CO2_to_COOH_ev"] - u
    dg2_0 = row["dG_COOH_to_CO_ev"] - u
    assert row["limiting_potential_v"] == pytest.approx(-max(dg1_0, dg2_0))
    assert not any(key.startswith("dE_") for key in row)


def test_thermo_settings_yaml_and_cli_precedence(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "methods:\n  co2rr:\n    vibrations: false\n    temperature_kelvin: 300\n"
        "    partial_pressures_bar: {H2O: 0.02}\n    gas_corrections_ev: {CO2: 0.1}\n"
    )
    args = co2rr.parse_args(["--config", str(config), "--potential", "-0.4"])
    settings = co2rr.resolve_thermo_settings(args, co2rr.load_co2rr_config(args.config))
    assert settings["vibrations"] is False
    assert settings["temperature_kelvin"] == 300.0
    assert settings["electrode_potential_v_rhe"] == -0.4
    assert settings["partial_pressures_bar"] == {"CO2": 1.0, "CO": 1.0, "H2": 1.0, "H2O": 0.02}
    assert settings["gas_corrections_ev"] == {"CO2": 0.1, "CO": 0.0}
    args = co2rr.parse_args(["--config", str(config), "--vibrations"])
    assert co2rr.resolve_thermo_settings(args, co2rr.load_co2rr_config(args.config))["vibrations"] is True


def test_cli_defaults_match_production_yaml():
    import yaml

    from scripts.common.project_config import ROOT_DIR

    config = yaml.safe_load((ROOT_DIR / "configs/simulations/zno_production.yaml").read_text())["methods"]["co2rr"]
    args = co2rr.parse_args([])
    assert args.fmax == config["fmax_ev_per_angstrom"] == 0.05
    assert args.steps == config["max_steps"] == 260
    assert args.ecut == config["ecut_ev"] == 500.0
    assert list(args.kpts) == config["kpoints"] == [2, 3, 1]
    assert args.molecule_box_vacuum == config["molecule_box_vacuum_angstrom"]
    settings = co2rr.resolve_thermo_settings(args, config)
    assert settings["vibrations"] is True
    assert settings["vibration_displacement_angstrom"] == 0.01
    assert settings["temperature_kelvin"] == 298.15
    assert settings["electrode_potential_v_rhe"] == 0.0
    assert settings["partial_pressures_bar"]["H2O"] == 0.035
    assert settings["gas_corrections_ev"] == {"CO2": 0.0, "CO": 0.0}


def test_plot_profile_uses_only_consistent_sequence():
    row = {
        "material": "ZnO", "site": "top_metal", "pathway_status": "ok", "energy_basis": "electronic_energies_only",
        "dE_CO2_to_COOH_ev": "0.5", "dE_COOH_to_CO_ev": "0.2", "dE_CO_desorption_ev": "0.1",
        "dE_O_formation_from_H2O_ev": "-3.0", "limiting_potential_v": "-0.5", "potential_v_rhe": "0",
    }
    profiles, basis = plot.site_profiles([row], "ZnO")
    assert basis == "electronic_energies_only"
    assert profiles[0]["profile"] == pytest.approx([0.0, 0.5, 0.7, 0.8])
    legacy = {"material": "ZnO", "site": "bridge", "pathway_status": "ok", "deltaG_CO2_to_COOH_ev": "1"}
    with pytest.raises(SystemExit):
        plot.site_profiles([legacy], "ZnO")


def test_analysis_reads_new_columns_and_ignores_legacy_rows():
    from scripts.analysis import analysis_co2_reduction as analysis

    merged = [{"material": "ZnO", "site": "top_metal", "quality_flag": "ok"},
              {"material": "ZnO", "site": "bridge", "quality_flag": "ok"}]
    pathway = [
        {"material": "ZnO", "site": "top_metal", "pathway_status": "ok", "energy_basis": "free_energy_CHE",
         "profile": "base", "dG_CO2_to_COOH_ev": "0.4", "limiting_step": "CO2_to_COOH",
         "limiting_potential_v": "-0.4", "geometry_flags": "co_desorbed", "notes": "n"},
    ]
    table = analysis.build_pathway_table(merged, pathway)
    assert table[0]["dG_CO2_to_COOH_ev"] == 0.4
    assert table[0]["limiting_potential_v"] == -0.4
    assert "co_desorbed" in table[0]["notes"]
    assert math.isnan(table[1]["dG_CO2_to_COOH_ev"])
    assert not any(key.startswith("deltaG") for key in table[0])
    legacy = [{"material": "ZnO", "site": "top_metal", "pathway_status": "ok", "deltaG_CO2_to_COOH_ev": "1"}]
    table = analysis.build_pathway_table(merged, legacy)
    assert table[0]["pathway_status"] == ""


def _nonperiodic(atoms):
    atoms = atoms.copy()
    atoms.pbc = False
    return atoms


def test_removed_imaginary_modes_are_counted_without_warnings():
    import warnings

    energies = [0.01j, 0.02j, 0.03j, 0.05, 0.1, 0.2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, n_imag = co2rr.adsorbate_free_energy_correction(energies, 298.15)
    assert n_imag == 3
