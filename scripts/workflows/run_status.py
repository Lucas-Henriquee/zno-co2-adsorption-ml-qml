"""Read-only status dashboard for a simulation run (geometry, DFT, TDDFT, CO2RR, MD, NEB).

Usage:
    python -m scripts.workflows.run_status [RUN_ID] [--tz ZONE] [--stale-minutes N] [--no-color]
    FORCE_COLOR=1 watch -c -n 60 python -m scripts.workflows.run_status RUN_ID

Without RUN_ID the most recently modified run under results/simulations is shown.
The dashboard only reads files; it never writes to the run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATIONS_DIR = PROJECT_ROOT / "results" / "simulations"
METHOD_ORDER = ("geometry", "dft", "tddft", "co2rr", "md", "neb")
METHOD_TITLES = {
    "geometry": "Geometry",
    "dft": "DFT  ·  adsorption",
    "tddft": "TDDFT  ·  isolated CO2 spectrum",
    "co2rr": "CO2RR  ·  CHE pathway",
    "md": "AIMD  ·  Langevin NVT",
    "neb": "NEB  ·  CI-NEB diffusion",
}
# Adsorbate atoms per CO2RR term: finite differences use 6 displacements per atom + equilibrium.
ADSORBATE_ATOMS = {"co2_star": 3, "cooh": 4, "co": 2, "o": 1, "co2": 3, "h2o": 3, "h2": 2}
TERM_LABELS = {"co2_star": "CO2*", "cooh": "COOH*", "co": "CO*", "o": "O*"}
CALC_SUFFIXES = {".txt", ".log", ".json", ".traj", ".csv", ".xyz", ".cif", ".gpw"}
IGNORED_PARTS = {"failed_attempts", "data", "logs"}
IGNORED_NAMES = {"run_manifest.json", "config.yaml"}


# --------------------------------------------------------------------------- styling


class Style:
    enabled = False

    @classmethod
    def wrap(cls, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if cls.enabled and text else text


def bold(text: str) -> str:
    return Style.wrap(text, "1")


def dim(text: str) -> str:
    return Style.wrap(text, "2")


def green(text: str) -> str:
    return Style.wrap(text, "32")


def yellow(text: str) -> str:
    return Style.wrap(text, "33")


def red(text: str) -> str:
    return Style.wrap(text, "31")


@dataclass
class Column:
    title: str
    width: int = 0
    align: str = ">"


def table(columns: list[Column], rows: list[list[str]], indent: int = 3) -> None:
    """Fixed-width table; the last column is free text (status, progress)."""
    pad = " " * indent
    header = "".join(f"{c.title:{c.align}{c.width}}" for c in columns[:-1]) + "  " + columns[-1].title
    print(pad + dim(header))
    for row in rows:
        cells = "".join(f"{value:{c.align}{c.width}}" for c, value in zip(columns[:-1], row[:-1]))
        print(pad + cells + "  " + row[-1])


# --------------------------------------------------------------------------- reading helpers


@dataclass
class Context:
    root: Path
    tz: ZoneInfo
    manifest: dict[str, Any]
    config: dict[str, Any]
    materials: list[str]
    sites: list[str]
    width: int = 0

    def method(self, name: str) -> dict[str, Any]:
        return self.config.get("methods", {}).get(name, {}) or {}


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def fmt(value: Any, spec: str = "+.3f", missing: str = "—") -> str:
    number = as_float(value)
    return missing if number is None else format(number, spec)


def lines_starting(path: Path, prefixes: tuple[str, ...]) -> list[str]:
    try:
        with path.open(errors="replace") as handle:
            return [line.rstrip() for line in handle if line.startswith(prefixes)]
    except OSError:
        return []


def newest(folder: Path, pattern: str, dirs: bool = False) -> Path | None:
    if not folder.is_dir():
        return None
    items = [p for p in folder.glob(pattern) if (p.is_dir() if dirs else p.is_file())]
    return max(items, key=lambda p: p.stat().st_mtime) if items else None


def optimizer_progress(log: Path, optimizer: str = "BFGS") -> tuple[int, str]:
    steps = lines_starting(log, (f"{optimizer}:",))
    fmax = steps[-1].split()[-1] if steps else "—"
    return len(steps), fmax


def local_time(value: float | str | None, ctx: Context, date: bool = True) -> str:
    if value in (None, ""):
        return "—"
    try:
        moment = (datetime.fromisoformat(value) if isinstance(value, str)
                  else datetime.fromtimestamp(value)).astimezone(ctx.tz)
    except ValueError:
        return str(value)
    return moment.strftime("%d/%m %H:%M" if date else "%H:%M")


def duration(start: str | None, end: str | None) -> str:
    try:
        begin = datetime.fromisoformat(start)
        finish = datetime.fromisoformat(end) if end else datetime.now(begin.tzinfo)
    except (TypeError, ValueError):
        return "—"
    minutes = int((finish - begin).total_seconds() // 60)
    return f"{minutes // 60}h{minutes % 60:02d}" if minutes >= 60 else f"{minutes} min"


def status_badge(status: str) -> str:
    return {
        "completed": green("✔ completed"),
        "running": yellow("● running"),
        "failed": red("✖ failed"),
        "pending": dim("○ pending"),
    }.get(status, status)


def ok_badge(value: str) -> str:
    if value in ("ok", "converged"):
        return green(value)
    return red(value) if value else dim("—")


def method_state(ctx: Context, method: str) -> str:
    record = ctx.manifest.get("methods", {}).get(method)
    if record:
        return str(record.get("status", "?"))
    return "pending" if method in ctx.manifest.get("resolved_methods", []) else "absent"


# --------------------------------------------------------------------------- sections


def section_dft(ctx: Context, state: str) -> None:
    settings = ctx.method("dft")
    print(dim(f"   PW {settings.get('ecut_ev', '?')} eV · k {settings.get('kpoints', '?')} · "
              f"fmax {settings.get('fmax_ev_per_angstrom', '?')} eV/Å · max {settings.get('max_steps', '?')} steps"))
    rows_by_key = {(r.get("material"), r.get("site")): r for r in read_csv(ctx.root / "dft_summary.csv")}
    columns = [Column("case", 22, "<"), Column("Eads/eV", 9), Column("  state", 15, "<"), Column("O-C-O/°", 9),
               Column("C-slab/Å", 10), Column("gap Γ/eV", 10), Column("steps", 7), Column("fmax", 8),
               Column("status / progress")]
    rows = []
    for material in ctx.materials:
        clean = next((r for (m, _), r in rows_by_key.items() if m == material), {})
        rows.append([f"{material} clean slab", "", "", "", "", fmt(clean.get("clean_band_gap_gamma_ev"), ".3f"),
                     fmt(clean.get("clean_n_steps"), ".0f"), fmt(clean.get("clean_fmax_final_ev_per_ang"), ".3f"),
                     dft_progress(ctx, state, material, "clean", clean, clean_case=True)])
        for site in ctx.sites:
            row = rows_by_key.get((material, site), {})
            rows.append([f"{material} {site}", fmt(row.get("adsorption_energy_ev")),
                         f"  {row.get('final_state') or '—'}", fmt(row.get("oco_angle_deg"), ".1f"),
                         fmt(row.get("c_min_slab_distance_ang"), ".2f"), fmt(row.get("adsorbed_band_gap_gamma_ev"), ".3f"),
                         fmt(row.get("adsorbed_n_steps"), ".0f"), fmt(row.get("adsorbed_fmax_final_ev_per_ang"), ".3f"),
                         dft_progress(ctx, state, material, site, row)])
    table(columns, rows)
    warnings = [f"{r.get('site')}: {r.get('warnings')}" for r in rows_by_key.values() if r.get("warnings")]
    for warning in warnings:
        print("   " + yellow(f"warning  {warning}"))


def dft_progress(ctx: Context, state: str, material: str, case: str, row: dict[str, str], clean_case: bool = False) -> str:
    if clean_case and row:
        return ok_badge("ok") + dim(f"  E = {fmt(row.get('clean_slab_energy_ev'), '.4f')} eV")
    if row.get("dft_status"):
        profile = row.get("dft_profile", "")
        return ok_badge(row["dft_status"]) + (dim(f"  profile {profile}") if profile else "")
    log = newest(ctx.root / "dft" / material / case, "*.opt.log")
    if state == "running" and log:
        steps, fmax = optimizer_progress(log)
        return yellow(f"relaxing · step {steps} · fmax {fmax} eV/Å")
    return dim("waiting") if state in ("running", "pending") else dim("—")


def section_tddft(ctx: Context, state: str) -> None:
    settings = ctx.method("tddft")
    print(dim(f"   fd h {settings.get('grid_spacing_angstrom', '?')} Å · box {settings.get('box_size_angstrom', '?')} Å · "
              f"{settings.get('converged_bands', '?')} converged bands · window {settings.get('energy_window_ev', '?')} eV · "
              f"onset f ≥ {settings.get('oscillator_strength_threshold', '?')}"))
    rows_by_key = {(r.get("material"), r.get("site")): r for r in read_csv(ctx.root / "tddft_summary.csv")}
    columns = [Column("case", 22, "<"), Column("1st exc/eV", 11), Column("onset/eV", 10), Column("peak/eV", 9),
               Column("peak f", 8), Column("Σf", 7), Column("TRK", 6), Column("status / progress")]
    rows = []
    for material in ctx.materials:
        for site in ctx.sites:
            row = rows_by_key.get((material, site), {})
            trk = as_float(row.get("trk_fraction"))
            rows.append([f"{material} {site}", fmt(row.get("co2_isolated_first_excitation_ev"), ".3f"),
                         fmt(row.get("co2_isolated_onset_ev"), ".3f"), fmt(row.get("co2_isolated_peak_ev"), ".3f"),
                         fmt(row.get("co2_isolated_peak_f"), ".3f"), fmt(row.get("co2_isolated_total_f"), ".3f"),
                         "—" if trk is None else f"{trk:.0%}", tddft_progress(ctx, state, material, site, row)])
    table(columns, rows)


def tddft_progress(ctx: Context, state: str, material: str, site: str, row: dict[str, str]) -> str:
    if row.get("tddft_status"):
        return ok_badge(row["tddft_status"])
    if state != "running":
        return dim("—")
    folder = ctx.root / "tddft" / material / site
    for name, label in (("lrtddft.txt", "Casida diagonalisation"),
                        ("co2_isolated_bands.txt", "unoccupied bands"),
                        ("co2_isolated_ground.txt", "ground-state SCF")):
        path = folder / name
        if path.is_file():
            iterations = len(lines_starting(path, ("iter:",)))
            return yellow(label + (f" · iter {iterations}" if iterations else ""))
    return dim("waiting")


def section_co2rr(ctx: Context, state: str) -> None:
    settings = ctx.method("co2rr")
    vib = "vibrations on (ZPE, ∫Cp, −TS)" if settings.get("vibrations", True) else "electronic energies only"
    print(dim(f"   {vib} · T {settings.get('temperature_kelvin', '?')} K · U {settings.get('electrode_potential_v_rhe', '?')} V vs RHE · "
              f"fmax {settings.get('fmax_ev_per_angstrom', '?')} eV/Å"))
    rows_by_key = {(r.get("material"), r.get("site")): r for r in read_csv(ctx.root / "co2rr_pathway_summary.csv")}
    basis = next((r.get("energy_basis") for r in rows_by_key.values() if r.get("energy_basis")), "")
    p = "dE" if basis == "electronic_energies_only" else "dG"
    columns = [Column("case", 22, "<"), Column(f"{p} →COOH*", 11), Column(f"{p} →CO*", 9), Column(f"{p} des", 9),
               Column(f"{p} total", 10), Column("U_L/V", 8), Column("μ COOH*", 9), Column("status / progress")]
    rows = []
    for material in ctx.materials:
        for site in ctx.sites:
            row = rows_by_key.get((material, site), {})
            rows.append([f"{material} {site}", fmt(row.get(f"{p}_CO2_to_COOH_ev")), fmt(row.get(f"{p}_COOH_to_CO_ev")),
                         fmt(row.get(f"{p}_CO_desorption_ev")), fmt(row.get(f"{p}_overall_ev")),
                         fmt(row.get("limiting_potential_v")), fmt(row.get("magmom_COOH_star_final"), ".2f"),
                         co2rr_progress(ctx, state, material, site, row)])
    table(columns, rows)
    for row in rows_by_key.values():
        notes = [f"imaginary {row['imaginary_modes']}" if row.get("imaginary_modes") else "",
                 f"flags {row['geometry_flags']}" if row.get("geometry_flags") else ""]
        notes = [n for n in notes if n]
        if notes:
            print("   " + dim(f"{row.get('site')}: " + " · ".join(notes)))
    if rows_by_key:
        print("   " + dim(f"{p} in eV at U = 0 V vs RHE; total = CO2(g) + 2(H+ + e-) → CO(g) + H2O (exp. ≈ +0.20 eV)"))


def co2rr_progress(ctx: Context, state: str, material: str, site: str, row: dict[str, str]) -> str:
    if row.get("pathway_status"):
        return ok_badge(row["pathway_status"]) + (dim(f"  limiting {row['limiting_step']}") if row.get("limiting_step") else "")
    if state != "running":
        return dim("—")
    site_dir = ctx.root / "pathways" / material / site
    profile = newest(site_dir, "*", dirs=True)
    if profile is None:
        return dim("waiting")
    for term in ("co2_star", "cooh", "co", "o"):
        folder = profile / term
        if not folder.is_dir():
            if term in ("co2_star", "cooh", "co"):
                return dim(f"{TERM_LABELS[term]} waiting")
            continue
        checkpoint = folder / "energy.json"
        if checkpoint.is_file() and "vibrations" in load_json(checkpoint):
            continue
        cache = newest(folder, "vib_*", dirs=True)
        if cache is not None:
            total = 6 * ADSORBATE_ATOMS.get(term, 1) + 1
            return yellow(f"{TERM_LABELS[term]} vibrations {len(list(cache.iterdir()))}/{total}")
        log = newest(folder, "*.opt.log")
        if log is not None:
            steps, fmax = optimizer_progress(log)
            return yellow(f"{TERM_LABELS[term]} relaxing · step {steps} · fmax {fmax} eV/Å")
        if checkpoint.is_file():
            return yellow(f"{TERM_LABELS[term]} vibrations starting")
        return yellow(f"{TERM_LABELS[term]} starting")
    return yellow("gas references / thermochemistry")


def section_md(ctx: Context, state: str) -> None:
    settings = ctx.method("md")
    eq_target = int(as_float(settings.get("equilibration_steps")) or 0)
    prod_target = int(as_float(settings.get("production_steps") or settings.get("steps")) or 0)
    interval = int(as_float(settings.get("log_interval")) or 20)
    seeds = settings.get("seeds") or [settings.get("seed", "?")]
    print(dim(f"   {settings.get('temperature_kelvin', '?')} K · dt {settings.get('timestep_fs', '?')} fs · "
              f"friction {settings.get('friction_per_fs', '?')} fs⁻¹ · {eq_target} eq + {prod_target} prod steps · seeds {seeds}"))
    columns = [Column("trajectory", 30, "<"), Column("equil.", 12), Column("production", 13), Column("⟨T⟩/K", 8),
               Column("⟨C-slab⟩/Å", 12), Column("⟨O-C-O⟩/°", 11), Column("status / progress")]
    rows = []
    for material in ctx.materials:
        for site in ctx.sites:
            site_dir = ctx.root / "md" / material / site
            summary = {r.get("seed"): r for r in read_csv(site_dir / "md_site_summary.csv")}
            seed_dirs = sorted(site_dir.glob("seed_*")) if site_dir.is_dir() else []
            if not seed_dirs:
                rows.append([f"{material} {site}", "—", "—", "—", "—", "—",
                             dim("waiting") if state in ("running", "pending") else dim("—")])
            for seed_dir in seed_dirs:
                seed = seed_dir.name.removeprefix("seed_")
                row = summary.get(seed, {})
                eq = md_steps(seed_dir / "md_equilibration.traj", interval)
                prod = md_steps(seed_dir / "md_production.traj", interval)
                if row.get("md_status"):
                    progress = ok_badge(row["md_status"])
                elif is_active(seed_dir):
                    progress = yellow("production" if (seed_dir / "md_production.traj").is_file() else "equilibration")
                else:
                    progress = dim("—")
                rows.append([f"{material} {site} · seed {seed}", f"{eq}/{eq_target}", f"{prod}/{prod_target}",
                             fmt(row.get("temperature_k_mean"), ".0f"), fmt(row.get("c_min_slab_distance_ang_mean"), ".2f"),
                             fmt(row.get("oco_angle_deg_mean"), ".1f"), progress])
    table(columns, rows)


def is_active(folder: Path, minutes: float = 15.0) -> bool:
    """True when a file in the folder was written recently (also outside the workflow)."""
    recent = time.time() - minutes * 60.0
    return any(p.stat().st_mtime > recent for p in folder.iterdir() if p.is_file())


def md_steps(path: Path, interval: int) -> int:
    """MD steps covered by a trajectory written every `interval` steps (frame 0 is the start)."""
    return max(trajectory_frames(path) - 1, 0) * interval


def trajectory_frames(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        from ase.io.trajectory import Trajectory

        with Trajectory(str(path)) as trajectory:
            return len(trajectory)
    except Exception:
        return 0


def section_neb(ctx: Context, state: str) -> None:
    settings = ctx.method("neb")
    max_steps = int(as_float(settings.get("max_steps")) or 0)
    plain_steps = round(max_steps * (as_float(settings.get("climb_start_fraction")) or 0.0))
    print(dim(f"   {settings.get('internal_images', '?')} internal images · {settings.get('optimizer', '?')} · "
              f"{settings.get('interpolation', '?')} · plain fmax {settings.get('fmax_ev_per_angstrom', '?')} → "
              f"CI fmax {settings.get('fmax_ci_ev_per_angstrom', '?')} eV/Å · {max_steps} steps"))
    columns = [Column("path", 34, "<"), Column("Ea fwd/eV", 10), Column("Ea rev/eV", 10), Column("ΔE/eV", 8),
               Column("steps", 11), Column("status / progress")]
    rows = []
    for material in ctx.materials:
        base = ctx.root / "neb" / material
        summary = {r.get("path"): r for r in read_csv(base / "neb_paths_summary.csv")}
        configured = [f"{a}_to_{b}" for a, b in settings.get("pairs") or []]
        found = [p.name for p in base.iterdir() if p.is_dir()] if base.is_dir() else []
        for pair in dict.fromkeys(configured + sorted(found)):
            folder = base / pair
            record = {**load_json(folder / "neb_run_summary.json"), **summary.get(pair, {})}
            steps = "—"
            if record.get("status"):
                steps = f"{fmt(record.get('n_steps_plain'), '.0f')}+{fmt(record.get('n_steps_ci'), '.0f')}"
                progress = ok_badge(str(record["status"]))
                if str(record.get("no_internal_saddle")).lower() == "true":
                    progress += dim("  no internal saddle")
            elif state == "running" and (folder / "neb_opt.log").is_file():
                optimizer = str(settings.get("optimizer", "FIRE"))
                count, fmax = optimizer_progress(folder / "neb_opt.log", optimizer)
                steps = f"{count}/{max_steps}"
                phase = "plain" if count <= plain_steps else "climbing image"
                progress = yellow(f"{phase} · fmax {fmax} eV/Å")
            else:
                progress = dim("waiting") if state in ("running", "pending") else dim("—")
            rows.append([f"{material} {pair.replace('_to_', ' → ')}", fmt(record.get("forward_barrier_eV")),
                         fmt(record.get("reverse_barrier_eV")), fmt(record.get("reaction_energy_eV")), steps, progress])
    table(columns, rows)


SECTIONS: dict[str, Callable[[Context, str], None]] = {
    "dft": section_dft, "tddft": section_tddft,
    "co2rr": section_co2rr, "md": section_md, "neb": section_neb,
}


# --------------------------------------------------------------------------- overview


def latest_calculation_file(root: Path) -> Path | None:
    best, best_time = None, 0.0
    for path in root.rglob("*"):
        if path.suffix not in CALC_SUFFIXES or path.name in IGNORED_NAMES:
            continue
        if IGNORED_PARTS.intersection(path.relative_to(root).parts) or not path.is_file():
            continue
        mtime = path.stat().st_mtime
        if mtime > best_time:
            best, best_time = path, mtime
    return best


def gpaw_processes() -> list[float]:
    try:
        output = subprocess.run(["ps", "-eo", "pcpu,args"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return []
    return [float(line.split()[0]) for line in output.splitlines()[1:]
            if "gpaw python" in line and "mpiexec" not in line and "prterun" not in line]


def memory_usage() -> str:
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        total = int(info["MemTotal"].split()[0]) / 1048576
        available = int(info["MemAvailable"].split()[0]) / 1048576
    except (OSError, KeyError, ValueError):
        return "—"
    return f"{total - available:.1f}/{total:.1f} GiB"


def print_header(ctx: Context, stale_minutes: float) -> bool:
    now = datetime.now(ctx.tz)
    title = f" QML_ZnO  ·  simulation run {ctx.root.name}"
    clock = now.strftime("%a %d %b %Y  %H:%M %Z ")
    print(bold("═" * ctx.width))
    print(bold(title + " " * max(1, ctx.width - len(title) - len(clock)) + clock))
    print(bold("═" * ctx.width))

    manifest = ctx.manifest
    print(f" {dim('created')}     {local_time(manifest.get('created_at'), ctx)}"
          f"    {dim('environment')} {manifest.get('environment', '—')}"
          f"    {dim('MPI')} {manifest.get('mpi_processes', '—')} ranks"
          f"    {dim('OMP')} {manifest.get('omp_num_threads', '—')}")
    print(f" {dim('materials')}   {', '.join(ctx.materials)}    {dim('sites')} {', '.join(ctx.sites)}")

    ranks = gpaw_processes()
    cpu = " ".join(f"{value:.0f}%" for value in ranks) if ranks else "no GPAW process"
    print(f" {dim('GPAW ranks')}  {cpu}    {dim('memory')} {memory_usage()}")

    running = any(r.get("status") == "running" for r in manifest.get("methods", {}).values())
    latest = latest_calculation_file(ctx.root)
    if latest is None:
        print(f" {dim('last write')}  —")
        return running
    age = (time.time() - latest.stat().st_mtime) / 60.0
    stamp = f"{local_time(latest.stat().st_mtime, ctx)}  ({age:.0f} min ago)"
    if running and age > stale_minutes:
        stamp = red(f"{stamp}  ⚠ no output for more than {stale_minutes:.0f} min — check for a stalled run")
    elif running:
        stamp = green(stamp)
    print(f" {dim('last write')}  {stamp}")
    print(f" {dim('            ' + str(latest.relative_to(ctx.root)))}")
    if running and not ranks:
        print(" " + red("⚠ a method is marked running but no GPAW process was found (interrupted?)"))
    return running


def print_overview(ctx: Context) -> None:
    print()
    print(" " + bold("Methods"))
    columns = [Column("method", 12, "<"), Column("status", 13, "<"), Column("started", 13, "<"),
               Column("finished", 13, "<"), Column("elapsed", 9, "<"), Column("note")]
    rows = []
    records = ctx.manifest.get("methods", {})
    for method in METHOD_ORDER:
        state = method_state(ctx, method)
        if state == "absent":
            continue
        record = records.get(method, {})
        end = record.get("completed_at") or record.get("failed_at")
        note = record.get("error", "")
        if record.get("previous_attempts"):
            note = f"{len(record['previous_attempts'])} earlier attempt(s) archived" + (f" · {note}" if note else "")
        badge = status_badge(state)
        rows.append([method, badge + " " * max(0, 13 - len(state) - 2),
                     local_time(record.get("started_at"), ctx), local_time(end, ctx) if end else "—",
                     duration(record.get("started_at"), end) if record.get("started_at") else "—",
                     dim(note[:70]) if note else ""])
    pad = "   "
    print(pad + dim("".join(f"{c.title:{c.align}{c.width}}" for c in columns[:-1]) + "  " + columns[-1].title))
    for row in rows:
        method, badge, started, finished, elapsed, note = row
        print(pad + f"{method:<12}{badge}{started:<13}{finished:<13}{elapsed:<9}  {note}")


def bar(fraction: float, width: int = 28) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * width))
    return "[" + green("█" * filled) + dim("░" * (width - filled)) + f"] {fraction:4.0%}"


def clock_times(lines: list[str], column: int = 2) -> list[datetime]:
    """HH:MM:SS stamps of optimizer/SCF log lines (same day assumed, midnight rollover handled)."""
    stamps: list[datetime] = []
    for line in lines:
        parts = line.split()
        if len(parts) <= column:
            continue
        try:
            stamp = datetime.strptime(parts[column], "%H:%M:%S")
        except ValueError:
            continue
        while stamps and stamp < stamps[-1]:
            stamp += timedelta(days=1)
        stamps.append(stamp)
    return stamps


def mean_interval_minutes(stamps: list[float]) -> float | None:
    if len(stamps) < 2:
        return None
    return (stamps[-1] - stamps[0]) / (len(stamps) - 1) / 60.0


def eta(ctx: Context, last: float, per_item_min: float | None, remaining: int) -> str:
    if per_item_min is None or remaining <= 0:
        return ""
    return f"~{per_item_min:.1f} min each · ETA {local_time(last + per_item_min * 60.0 * remaining, ctx, date=False)}"


def optimizer_live(ctx: Context, log: Path, optimizer: str, fmax_target: float, max_steps: int) -> None:
    lines = lines_starting(log, (f"{optimizer}:",))
    if not lines:
        print("   progress   " + dim("first SCF of the relaxation (no optimizer step yet)"))
        return
    fmaxes = [as_float(line.split()[-1]) or 0.0 for line in lines]
    first, current = max(fmaxes[0], fmax_target * 1.0001), max(fmaxes[-1], 1e-6)
    fraction = (math.log(first) - math.log(current)) / (math.log(first) - math.log(fmax_target))
    stamps = clock_times(lines)[-6:]  # recent pace; the first steps are always the slowest
    per_step = None
    if len(stamps) >= 2:
        per_step = (stamps[-1] - stamps[0]).total_seconds() / (len(stamps) - 1) / 60.0
    rate = f"   ~{per_step:.1f} min/step" if per_step else ""
    print(f"   progress   {bar(fraction)}  fmax {fmaxes[-1]:.3f} → {fmax_target} eV/Å"
          f"   step {len(lines) - 1}/{max_steps}{rate}")
    print("   " + dim(f"recent {optimizer} steps   step  time      energy/eV        fmax"))
    for line in lines[-4:]:
        print("   " + " " * 20 + line.split(":", 1)[1].strip())


def scf_live(path: Path) -> None:
    iterations = lines_starting(path, ("iter:",))
    if not iterations:
        print("   SCF        " + dim("initialising"))
        return
    last = iterations[-1].split()
    magmom = last[-1] if len(last) >= 7 else "—"
    print(f"   SCF        iteration {last[1]} · {last[2]} · E = {last[3].rstrip('c')} eV · magmom {magmom}"
          + dim(f"   ({path.name})"))


def live_target(ctx: Context) -> Path | None:
    """Folder of the calculation that wrote last (ignores archives and run bookkeeping)."""
    latest = latest_calculation_file(ctx.root)
    if latest is None:
        return None
    return latest.parent


def print_live(ctx: Context) -> None:
    folder = live_target(ctx)
    if folder is None:
        return
    rel = folder.relative_to(ctx.root).parts
    print()
    print(" " + bold(yellow("▶ LIVE")) + "  " + bold(" · ".join(rel)))
    newest_txt = newest(folder, "*.txt") or newest(folder.parent, "*.txt")
    method = rel[0] if rel else ""

    if method == "dft":
        settings = ctx.method("dft")
        log = newest(folder, "*.opt.log")
        if log:
            optimizer_live(ctx, log, "BFGS", float(settings.get("fmax_ev_per_angstrom", 0.05)),
                           int(settings.get("max_steps", 180)))
    elif method == "tddft":
        for name, label in (("lrtddft.txt", "Casida (LrTDDFT) diagonalisation"),
                            ("co2_isolated_bands.txt", "unoccupied bands on the fixed density"),
                            ("co2_isolated_ground.txt", "ground-state SCF of the isolated CO2")):
            if (folder / name).is_file():
                print(f"   stage      {label}")
                break
    elif method == "pathways" or (len(rel) > 1 and rel[-2].startswith("vib_")):
        settings = ctx.method("co2rr")
        term_dir = folder.parent if folder.name.startswith("vib_") else folder
        cache = folder if folder.name.startswith("vib_") else newest(folder, "vib_*", dirs=True)
        term = term_dir.name
        total = 6 * ADSORBATE_ATOMS.get(term, 1) + 1
        checkpoint = term_dir / "energy.json"
        relaxing_log = newest(term_dir, "*.opt.log")
        if cache is not None and not (checkpoint.is_file() and "vibrations" in load_json(checkpoint)):
            files = sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime)
            done = len(files)
            per = mean_interval_minutes([p.stat().st_mtime for p in files])
            print(f"   stage      {TERM_LABELS.get(term, term)} finite-difference vibrations")
            print(f"   progress   {bar(done / total)}  {done}/{total} displacements   "
                  + eta(ctx, files[-1].stat().st_mtime, per, total - done))
        elif relaxing_log is not None:
            print(f"   stage      {TERM_LABELS.get(term, term)} geometry relaxation")
            optimizer_live(ctx, relaxing_log, "BFGS", float(settings.get("fmax_ev_per_angstrom", 0.05)),
                           2 * int(settings.get("max_steps", 260)))
        if len(rel) >= 4 and rel[1] != "_references":
            profile_dir = ctx.root / "pathways" / rel[1] / rel[2] / rel[3]
            marks = []
            for name in ("co2_star", "cooh", "co"):
                term_path = profile_dir / name / "energy.json"
                complete = term_path.is_file() and "vibrations" in load_json(term_path)
                current = name == term
                symbol = green("✔") if complete else (yellow("▶") if current else dim("·"))
                marks.append(f"{symbol} {TERM_LABELS[name]}")
            print("   site       " + "   ".join(marks))
    elif method == "md":
        settings = ctx.method("md")
        interval = int(as_float(settings.get("log_interval")) or 20)
        eq_target = int(as_float(settings.get("equilibration_steps")) or 0)
        prod_target = int(as_float(settings.get("production_steps") or settings.get("steps")) or 0)
        eq = md_steps(folder / "md_equilibration.traj", interval)
        prod = md_steps(folder / "md_production.traj", interval)
        production = (folder / "md_production.traj").is_file()
        phase, done, target = ("production", prod, prod_target) if production else ("equilibration", eq, eq_target)
        print(f"   stage      {phase}  ({folder.name})")
        print(f"   progress   {bar(done / target if target else 0.0)}  ~{done}/{target} steps "
              f"(trajectory {eq + prod}/{eq_target + prod_target})")
        frames = [line for line in lines_starting(ctx.root / "logs" / "md.log", ("[md] seed=",))]
        if frames:
            print("   last frame " + frames[-1].removeprefix("[md] "))
            print("   " + dim("C-slab > ~4.5 Å means the CO2 has left the surface (desorption)"))
    elif method == "neb":
        settings = ctx.method("neb")
        max_steps = int(as_float(settings.get("max_steps")) or 300)
        plain = round(max_steps * (as_float(settings.get("climb_start_fraction")) or 0.0))
        log = folder / "neb_opt.log"
        optimizer = str(settings.get("optimizer", "FIRE"))
        if log.is_file():
            steps = len(lines_starting(log, (f"{optimizer}:",)))
            phase = "plain NEB" if steps <= plain else "climbing image"
            target = float(settings.get("fmax_ev_per_angstrom" if steps <= plain else "fmax_ci_ev_per_angstrom", 0.05))
            print(f"   stage      {phase} (plain phase ≤ {plain} steps, then CI; stops early when converged)")
            optimizer_live(ctx, log, optimizer, target, max_steps)

    if newest_txt is not None:
        scf_live(newest_txt)


# --------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only status dashboard for a simulation run.")
    parser.add_argument("run_id", nargs="?", help="Run ID under results/simulations (default: most recent).")
    parser.add_argument("--tz", default="America/Sao_Paulo", help="Time zone for all times (default: Brasília).")
    parser.add_argument("--stale-minutes", type=float, default=10.0,
                        help="Warn when a running method has not written any file for this long.")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args(argv)


def resolve_run(run_id: str | None) -> Path:
    if run_id:
        return SIMULATIONS_DIR / run_id
    runs = [p for p in SIMULATIONS_DIR.iterdir() if (p / "run_manifest.json").is_file()] if SIMULATIONS_DIR.is_dir() else []
    if not runs:
        sys.exit(f"No runs found under {SIMULATIONS_DIR}")
    return max(runs, key=lambda p: (p / "run_manifest.json").stat().st_mtime)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    Style.enabled = not args.no_color and (sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR")))
    root = resolve_run(args.run_id)
    if not root.is_dir():
        sys.exit(f"Run not found: {root}")
    manifest = load_json(root / "run_manifest.json")
    ctx = Context(
        root=root,
        tz=ZoneInfo(args.tz),
        manifest=manifest,
        config=load_yaml(root / "config.yaml"),
        materials=list(manifest.get("materials") or ["ZnO"]),
        sites=list(manifest.get("sites") or ["top_metal", "top_oxygen", "bridge"]),
        width=max(100, min(shutil.get_terminal_size((120, 40)).columns, 140)),
    )

    # A method launched outside the workflow (e.g. an extra MD seed) is live as well.
    running = print_header(ctx, args.stale_minutes) or bool(gpaw_processes())
    if running:
        print_live(ctx)
    print_overview(ctx)
    for method in METHOD_ORDER:
        state = method_state(ctx, method)
        if state == "absent" or method == "geometry":
            continue
        print()
        print(" " + bold(METHOD_TITLES[method]) + "   " + status_badge(state))
        SECTIONS[method](ctx, state)
    print(bold("═" * ctx.width))


if __name__ == "__main__":
    main()
