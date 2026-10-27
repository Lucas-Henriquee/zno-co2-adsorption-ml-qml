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
