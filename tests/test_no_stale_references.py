"""Guard against paths, columns and scores of the invalid first campaign.

The first campaign (results/_invalid_ic_v1) used a wrong ZnO slab and several wrong
conventions. Its result folders, column names and the ad hoc screening score must not
come back into the active code, configs, docs or notebook; scripts/legacy/ keeps them
for provenance only.
"""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STALE_PATTERNS = (
    "my_analysis",            # hand-renamed CSVs of the first campaign
    "neb_iddp",               # first-campaign NEB folder
    "deltaG_",                # electronic energies labelled as free energies
    "O_removal",              # O* step summed into the CO2 -> CO profile
    "tddft_onset_ev",         # TDDFT energies in Hartree labelled as eV
    "reduction_proxy_score",  # ad hoc score with a -0.70 eV target
    "run_pipeline.py",        # flat-layout pipeline writing into results/
    "TARGET_ADSORPTION_ENERGY_EV",  # screening targets of the frozen first campaign
    "TARGET_BAND_GAP_EV",
    "TARGET_ONSET_EV",
)

# Paths that may mention the patterns: the archive, the legacy code, this test, tests
# that check that old inputs are rejected, and pages that only point at the archive.
ALLOWED_PREFIXES = (
    "results/_invalid_ic_v1/",
    "scripts/legacy/",
    "tests/",
)
ALLOWED_LINES = {
    # Points at the archived pipeline by its new location.
    ("docs/repository_layout.md", "scripts/legacy/workflows/run_pipeline.py"),
}
# Pages still to be rewritten (none).
PENDING_ML_REVIEW: tuple[str, ...] = ()


def git_files(*paths: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", *paths], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest(f"needs a git checkout: {exc}")
    return result.stdout.splitlines()


def tracked_text_files() -> list[str]:
    suffixes = {".py", ".sh", ".md", ".yaml", ".yml", ".ipynb", ".txt", ".toml", ".cfg"}
    return [name for name in git_files() if Path(name).suffix in suffixes]


class NoStaleReferencesTest(unittest.TestCase):
    def test_active_files_do_not_use_first_campaign_names(self):
        problems = []
        for name in tracked_text_files():
            if name.startswith(ALLOWED_PREFIXES) or name in PENDING_ML_REVIEW:
                continue
            path = ROOT / name
            if not path.is_file():
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                for pattern in STALE_PATTERNS:
                    if pattern in line and not any(
                        name == allowed_file and token in line for allowed_file, token in ALLOWED_LINES
                    ):
                        problems.append(f"{name}:{number}: {pattern}")
        self.assertEqual(problems, [], "first-campaign names outside scripts/legacy:\n" + "\n".join(problems))

    def test_root_figures_folder_is_not_tracked(self):
        tracked = git_files("figures")
        self.assertFalse(
            tracked,
            "figures/ held first-campaign results; write figures below results/analyses/<run-id>/",
        )


if __name__ == "__main__":
    unittest.main()
