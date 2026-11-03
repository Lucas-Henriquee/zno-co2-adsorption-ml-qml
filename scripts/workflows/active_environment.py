"""Name of the conda environment a workflow runs in, as recorded in run manifests.

Resume compares this name with the one stored in the manifest, so a run started with a
launcher (which exports QML_ZNO_ACTIVE_ENVIRONMENT) must record the same name as the same
run continued from a notebook kernel or with ``python -m`` in that environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def active_environment(configured: str) -> str:
    """QML_ZNO_ACTIVE_ENVIRONMENT, else the conda env of this interpreter, else ``configured``.

    The interpreter's own prefix (``.../envs/<name>``) is preferred to CONDA_DEFAULT_ENV,
    which names the shell's activated env and differs when a Jupyter server started in
    one env runs a kernel of another.
    """
    explicit = os.environ.get("QML_ZNO_ACTIVE_ENVIRONMENT", "").strip()
    if explicit:
        return env_label(explicit)
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        return prefix.name
    return env_label(os.environ.get("CONDA_DEFAULT_ENV", "")) or configured


def env_label(value: str) -> str:
    """An environment name, or the last component of a prefix path (as the launchers record it)."""
    value = value.strip().rstrip("/")
    return Path(value).name if "/" in value else value
