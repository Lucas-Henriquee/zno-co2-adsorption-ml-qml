#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/environment/conda_tools.sh"
cd "${PROJECT_ROOT}"
# Analyses need only ASE, numpy, pandas, matplotlib and PyYAML, so they may run in another
# environment than the models (e.g. QML_ZNO_ANALYSIS_ENV=gpaw-dft).
launch_in_env "${QML_ZNO_ANALYSIS_ENV:-${QML_ZNO_MODEL_ENV:-qml-zno-modeling}}" \
  "ase matplotlib numpy pandas yaml" analysis scripts.workflows.run_analysis "$@"
