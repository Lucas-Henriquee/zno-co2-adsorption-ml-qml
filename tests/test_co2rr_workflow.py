"""Pathway plumbing with an EMT stand-in for GPAW: spin moments, geometry, checkpoint reuse."""

from argparse import Namespace

import numpy as np
import pytest
from ase.build import fcc100
from ase.calculators.emt import EMT
from ase.io import write

from scripts.simulations.co2rr import run_co2rr_pathways as co2rr
from scripts.simulations.dft import run_dft


class StubEMT(EMT):
    """EMT with the GPAW methods optimize_structure calls."""

    def write(self, *args, **kwargs):
        pass


@pytest.fixture
def emt(monkeypatch):
    calls = {"build": 0, "optimize": 0}

    def fake_build_calculator(**kwargs):
        calls["build"] += 1
        return StubEMT()

    original = run_dft.optimize_structure

    def counting_optimize(**kwargs):
        calls["optimize"] += 1
        return original(**kwargs)

    monkeypatch.setattr(run_dft, "build_calculator", fake_build_calculator)
    monkeypatch.setattr(run_dft, "optimize_structure", counting_optimize)
    return calls


def small_slab():
    slab = fcc100("Cu", size=(2, 2, 2), vacuum=6.0)
    slab.set_initial_magnetic_moments(np.zeros(len(slab)))
    return slab


def top_site(slab):
    top = int(np.argmax(slab.positions[:, 2]))
    return slab.positions[top].copy()


def test_spin_moments_only_on_radicals():
    slab = small_slab()
    site = top_site(slab)
    n = len(slab)
    cooh, n_cooh = co2rr.build_intermediate(slab, "top_metal", site, "cooh")
    co, n_co = co2rr.build_intermediate(slab, "top_metal", site, "co")
    o_star, _ = co2rr.build_intermediate(slab, "top_metal", site, "o")
    assert (n_cooh, n_co) == (4, 2)
    assert cooh.get_chemical_symbols()[n:] == ["C", "O", "O", "H"]
    assert cooh.get_initial_magnetic_moments()[n] == 1.0
    assert not np.any(cooh.get_initial_magnetic_moments()[:n])
    assert o_star.get_initial_magnetic_moments()[n] == 2.0
    assert not np.any(co.get_initial_magnetic_moments())
    for name in co2rr.GAS_SPECIES:
        assert co2rr.GAS_SPECIES[name]["spin"] == 0
    # CO upright, C down, on the site.
    assert co.positions[n][:2] == pytest.approx(site[:2])
    assert co.positions[n + 1][2] - co.positions[n][2] == pytest.approx(1.15)
    # COOH: C is the lowest adsorbate atom (C-down).
    assert np.argmin(cooh.positions[n:, 2]) == 0


def test_bond_checks_flag_desorbed_and_dissociated_co():
    slab = small_slab()
    site = top_site(slab)
    co, _ = co2rr.build_intermediate(slab, "top_metal", site, "co")
    n = len(slab)
    flags, distance = co2rr.check_adsorbate_bonds(co, n, "co")
    assert flags == [] and distance == pytest.approx(2.0)
    co.positions[n:, 2] += 1.0
    flags, distance = co2rr.check_adsorbate_bonds(co, n, "co")
    assert "co_desorbed" in flags and distance > 2.5
    co.positions[n + 1, 2] += 1.0
    assert "co_dissociated" in co2rr.check_adsorbate_bonds(co, n, "co")[0]


def test_metadata_mismatch_detection():
    slab = small_slab()
    base = co2rr.calc_metadata(term="co", profile_name="base", ecut=500, kpts=(2, 3, 1), fmax=0.05,
                               atoms=slab, n_slab_atoms=len(slab))
    assert co2rr.metadata_mismatches(base, dict(base)) == []
    assert co2rr.metadata_mismatches({}, base) == list(base)
    for key, value in (("ecut_ev", 400.0), ("kpts", [3, 3, 1]), ("profile", "robust"),
                       ("n_slab_atoms", 4), ("fmax_ev_per_ang", 0.06), ("cell_hash", "x")):
        changed = dict(base, **{key: value})
        assert co2rr.metadata_mismatches(changed, base) == [key]
    bigger = slab.repeat((2, 1, 1))
    assert co2rr.cell_hash(bigger) != co2rr.cell_hash(slab)


def _compute(tmp_path, atoms, n_slab, ecut=500.0, vib_delta=0.01, overwrite=False):
    profile = run_dft.OPTIMIZATION_PROFILES[0]
    return co2rr.compute_term(
        term="co", label="cu_co", out_dir=tmp_path / "co", atoms=atoms,
        adsorbate_indices=list(range(n_slab, len(atoms))), n_slab_atoms=n_slab, profile=profile,
        ecut=ecut, kpts=(2, 3, 1), fmax=0.05, steps=200, overwrite=overwrite, vib_delta=vib_delta,
    )


def test_checkpoint_reuse_only_with_matching_parameters(tmp_path, emt):
    slab = small_slab()
    atoms, _ = co2rr.build_intermediate(slab, "top_metal", top_site(slab), "co")
    n = len(slab)

    first = _compute(tmp_path, atoms, n)
    assert emt["optimize"] == 1
    assert first.source == "optimize"
    assert len(first.vib_energies) == 6
    payload = co2rr.read_checkpoint(tmp_path / "co" / "energy.json")
    for key in ("ecut_ev", "kpts", "profile", "n_slab_atoms", "cell_hash", "fmax_ev_per_ang"):
        assert key in payload["metadata"]
    assert payload["energy_ev"] == pytest.approx(first.energy_ev)
    assert payload["fmax_final_ev_per_ang"] <= 0.05

    again = _compute(tmp_path, atoms, n)
    assert emt["optimize"] == 1  # reused
    assert again.energy_ev == pytest.approx(first.energy_ev)
    assert again.vib_energies == pytest.approx(first.vib_energies)

    _compute(tmp_path, atoms, n, ecut=400.0)
    assert emt["optimize"] == 2  # refused: ecut differs

    bigger_slab = small_slab().repeat((2, 1, 1))
    bigger, _ = co2rr.build_intermediate(bigger_slab, "top_metal", top_site(bigger_slab), "co")
    _compute(tmp_path, bigger, len(bigger_slab), ecut=400.0)
    assert emt["optimize"] == 3  # refused: different slab


def test_known_energy_must_match_checkpoint(tmp_path, emt):
    slab = small_slab()
    profile = run_dft.OPTIMIZATION_PROFILES[0]
    kwargs = dict(term="clean_slab", label="clean_slab", out_dir=tmp_path / "clean", atoms=slab,
                  adsorbate_indices=[], n_slab_atoms=len(slab), profile=profile, ecut=500.0,
                  kpts=(2, 3, 1), fmax=0.05, steps=10, overwrite=False, vib_delta=None,
                  known_source="dft_summary")
    first = co2rr.compute_term(**kwargs, known_energy=-1.0)
    assert first.energy_ev == -1.0 and first.source == "dft_summary"
    second = co2rr.compute_term(**kwargs, known_energy=-2.0)
    assert second.energy_ev == -2.0
    assert emt["optimize"] == 0


def test_site_row_end_to_end_with_emt(tmp_path, monkeypatch, emt):
    """Whole site in one profile: CO2(g) + * start, dG_* columns, U_L from PCET steps."""
    monkeypatch.setattr(co2rr, "PATHWAY_RESULTS_DIR", tmp_path / "pathways")
    monkeypatch.setattr(co2rr, "DFT_RESULTS_DIR", tmp_path / "dft")
    from scripts.simulations.dft import build_geometries

    monkeypatch.setattr(
        build_geometries, "select_adsorption_sites",
        lambda slab, metal: {"top_metal": top_site(slab)},
    )
    monkeypatch.setitem(co2rr.MATERIALS, "Cu", co2rr.MATERIALS["ZnO"].__class__(
        name="Cu", metal_symbol="Cu", structure="fcc", miller_index=(1, 0, 0)))

    slab = small_slab()
    clean_dir = tmp_path / "dft" / "Cu" / "clean"
    clean_dir.mkdir(parents=True)
    write(clean_dir / "clean_slab_optimized.xyz", slab)
    co2 = co2rr.molecule_atoms("CO2")
    co2.translate(top_site(slab) + [0.0, 0.0, 3.0])
    co2_star = slab.copy()
    co2_star.extend(co2)
    site_dir = tmp_path / "dft" / "Cu" / "top_metal"
    site_dir.mkdir(parents=True)
    write(site_dir / "Cu_top_metal_optimized.xyz", co2_star)

    dft_row = {
        "material": "Cu", "site": "top_metal", "dft_status": "ok", "dft_profile": "base",
        "clean_slab_energy_ev": "0.5", "adsorbed_energy_ev": "1.0",
        "clean_fmax_final_ev_per_ang": "0.01", "adsorbed_fmax_final_ev_per_ang": "0.02",
    }
    args = Namespace(ecut=500.0, kpts=(2, 3, 1), fmax=0.05, steps=300, molecule_box_vacuum=5.0, overwrite=False)
    settings = {k: (dict(v) if isinstance(v, dict) else v) for k, v in co2rr.DEFAULT_THERMO_SETTINGS.items()}
    ctx = co2rr.RunContext(args=args, settings=settings, catalog=co2rr.profile_catalog(), kpts=(2, 3, 1))

    row = co2rr.run_site("Cu", "top_metal", dft_row, ctx)
    assert row["pathway_status"] == "ok", row.get("error_message")
    assert row["energy_basis"] == "free_energy_CHE"
    assert row["profile"] == "base" and row["kpts"] == "2x3x1" and row["n_slab_atoms"] == len(slab)
    assert row["E_clean_slab_ev"] == 0.5 and row["E_CO2_star_ev"] == 1.0
    steps = [row[f"dG_{s}_ev"] for s in co2rr.PATHWAY_STEPS]
    assert sum(steps) == pytest.approx(row["dG_overall_ev"])
    assert row["limiting_potential_v"] == pytest.approx(-max(steps[:2]))
    assert "dG_O_formation_from_H2O_ev" not in row
    assert "magmom_COOH_star_final" in row
    cooh_meta = co2rr.read_checkpoint(tmp_path / "pathways" / "Cu" / "top_metal" / "base" / "cooh" / "energy.json")
    assert cooh_meta["metadata"]["initial_magmom_total"] == 1.0
    co_meta = co2rr.read_checkpoint(tmp_path / "pathways" / "Cu" / "top_metal" / "base" / "co" / "energy.json")
    assert co_meta["metadata"]["initial_magmom_total"] == 0.0

    # Unconverged DFT input is refused instead of silently combined.
    bad = dict(dft_row, adsorbed_fmax_final_ev_per_ang="0.2")
    assert co2rr.run_site("Cu", "top_metal", bad, ctx)["pathway_status"] == "inconsistent"
