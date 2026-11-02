import contextlib
import io
import json

from scripts.workflows import run_status


def test_dashboard_reports_results_and_progress_read_only(tmp_path, monkeypatch):
    run = tmp_path / "r1"
    (run / "pathways/ZnO/top_oxygen/base/co2_star/vib_x").mkdir(parents=True)
    (run / "pathways/ZnO/top_oxygen/base/co2_star/vib_x/cache.eq.json").write_text("{}")
    (run / "run_manifest.json").write_text(json.dumps({
        "materials": ["ZnO"], "sites": ["top_metal", "top_oxygen"],
        "resolved_methods": ["geometry", "dft", "co2rr", "neb"],
        "methods": {
            "dft": {"status": "completed", "started_at": "2026-10-27T00:00:00+00:00",
                    "completed_at": "2026-10-27T01:30:00+00:00"},
            "co2rr": {"status": "running", "started_at": "2026-10-27T02:00:00+00:00"},
        },
    }))
    (run / "dft_summary.csv").write_text(
        "material,site,dft_status,adsorption_energy_ev,final_state\n"
        "ZnO,top_metal,ok,0.041,physisorbed\nZnO,top_oxygen,ok,-0.011,physisorbed\n")
    before = sorted(p.name for p in run.rglob("*"))
    monkeypatch.setattr(run_status, "SIMULATIONS_DIR", tmp_path)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_status.main(["r1", "--no-color"])
    text = out.getvalue()

    assert "+0.041" in text and "physisorbed" in text
    assert "CO2* vibrations 1/19" in text
    assert "1h30" in text  # DFT elapsed time
    assert "pending" in text  # NEB not started
    assert sorted(p.name for p in run.rglob("*")) == before


def _neb_run(tmp_path, monkeypatch, status="running"):
    run = tmp_path / "r2"
    pair = run / "neb" / "ZnO" / "top_metal_to_bridge"
    pair.mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "materials": ["ZnO"], "sites": ["top_metal", "bridge"], "resolved_methods": ["neb"],
        "methods": {"neb": {"status": status, "started_at": "2026-10-27T00:00:00+00:00"}},
    }))
    (run / "config.yaml").write_text(
        "methods:\n  neb:\n    max_steps: 300\n    climb_start_fraction: 0.30\n    optimizer: FIRE\n"
        "    fmax_ev_per_angstrom: 0.08\n    fmax_ci_ev_per_angstrom: 0.05\n"
        "    pairs: [[top_metal, bridge]]\n")
    # 21 optimizer lines: step 20, well below the 90-step plain budget.
    (pair / "neb_opt.log").write_text("".join(
        f"FIRE:   {i:3d} 12:{i:02d}:00     -100.000000        0.0600\n" for i in range(21)))
    monkeypatch.setattr(run_status, "SIMULATIONS_DIR", tmp_path)
    return run, pair


def _render():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_status.main(["r2", "--no-color"])
    return out.getvalue()


def _ctx(run):
    return run_status.Context(root=run, tz=None, manifest={}, config={}, materials=["ZnO"], sites=[])


def test_neb_phase_from_phase_file_not_step_budget(tmp_path, monkeypatch):
    run, pair = _neb_run(tmp_path, monkeypatch)
    # Early switch at step 12: the step-budget heuristic would still say "plain".
    (pair / "neb_phase.json").write_text(json.dumps({"phase": "climbing_image", "phase_start_step": 12}))
    phase = run_status.neb_phase(_ctx(run), "ZnO", "top_metal_to_bridge", 20, 90)
    assert phase["label"] == "climbing image" and phase["start"] == 12
    (pair / "neb_phase.json").write_text(json.dumps({"phase": "plain_continued", "phase_start_step": 12}))
    text = _render()
    assert "plain NEB (climb skipped) · fmax 0.0600" in text  # NEB table row
    # Live block: the CI threshold is the target of the continued plain phase.
    assert "stage      plain NEB (climb skipped) (no internal image above both endpoints since step 12" in text
    assert "fmax 0.060 → 0.05 eV/Å" in text


def test_neb_phase_from_run_log(tmp_path, monkeypatch):
    run, _ = _neb_run(tmp_path, monkeypatch)
    log = run / "logs" / "neb.log"
    log.write_text(
        "\n[neb] Start | material=ZnO | path=top_metal -> bridge\n"
        "[neb] Phase 1: plain NEB, fmax=0.08, up to 90 steps\n"
        "[neb] Phase 2: climbing image skipped after 12 plain steps (no internal maximum ...); "
        "plain NEB continues, fmax=0.05, up to 288 steps\n")
    phase = run_status.neb_phase(_ctx(run), "ZnO", "top_metal_to_bridge", 20, 90)
    assert phase["phase"] == "plain_continued" and phase["start"] == 12 and phase["source"] == "neb.log"
    # Older runs print "Phase 2: climbing image, ..." without the step count.
    log.write_text(
        "[neb] Start | material=ZnO | path=top_metal -> bridge\n"
        "[neb] Phase 1: plain NEB, fmax=0.08, up to 90 steps\n"
        "[neb] Phase 2: climbing image, fmax=0.05, up to 288 steps\n")
    assert run_status.neb_phase(_ctx(run), "ZnO", "top_metal_to_bridge", 20, 90)["phase"] == "climbing_image"
    # A Start line of another pair does not carry over.
    log.write_text("[neb] Start | material=ZnO | path=bridge -> top_oxygen\n"
                   "[neb] Phase 2: climbing image, fmax=0.05, up to 288 steps\n")
    phase = run_status.neb_phase(_ctx(run), "ZnO", "top_metal_to_bridge", 20, 90)
    assert phase["source"] == "step budget" and phase["label"] == "plain NEB (estimated)"


def test_neb_table_flags_endpoint_not_minimum(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    run, pair = _neb_run(tmp_path, monkeypatch, status="completed")
    frames = []
    for energy in (0.0, -0.0335, -0.0953, -0.1027, -0.0664):
        atoms = Atoms("CO2", positions=[(0, 0, 0), (0, 0, 1.16), (0, 0, -1.16)])
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        frames.append(atoms)
    write(pair / "neb_final_images.traj", frames)
    (pair / "neb_run_summary.json").write_text(json.dumps({
        "status": "converged", "n_steps_plain": 90, "n_steps_ci": 0, "n_steps_phase2": 40,
        "climb_skipped": True}))
    text = _render()
    assert "90+40" in text
    assert "endpoint not a band minimum (IS dip 0.103, FS dip 0.036 eV)" in text
    assert "no internal saddle" in text and "climb skipped" in text
