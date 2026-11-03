#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SETUP_SCRIPT="${PROJECT_DIR}/scripts/environment/setup_environment.sh"

if [[ ! -x "${SETUP_SCRIPT}" ]]; then
  echo "[error] Missing environment setup script: ${SETUP_SCRIPT}" >&2
  exit 1
fi

# Create (or validate, if it exists) the simulation environment used by the notebooks.
# Accepts the options of setup_environment.sh, e.g. --env NAME, --manager micromamba.
exec "${SETUP_SCRIPT}" simulation "$@"
