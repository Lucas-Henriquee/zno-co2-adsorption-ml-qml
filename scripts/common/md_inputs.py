"""Locate the MD production-frame table of every material and adsorption site."""

from pathlib import Path

# Site-level table written by run_md.py: production frames only, all seeds
# concatenated, with a 'seed' column identifying each independent trajectory.
MD_FRAMES_FILENAME = "md_production_frames.csv"


def find_md_summaries(results_dir: Path) -> list[Path]:
    """Return md/<material>/<site>/md_production_frames.csv for every site.

    The per-seed folders (seed_<n>/) are not read; the site-level table already
    contains every seed.
    """
    paths = sorted(results_dir.glob(f"md/*/*/{MD_FRAMES_FILENAME}"))
    if not paths:
        raise FileNotFoundError(
            f"No MD production tables ({MD_FRAMES_FILENAME}) found below {results_dir / 'md'}"
        )
    return paths
