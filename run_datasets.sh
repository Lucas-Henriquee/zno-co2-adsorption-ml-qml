#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/environment/conda_tools.sh"
cd "${PROJECT_ROOT}"
launch_in_env "${QML_ZNO_MODEL_ENV:-qml-zno-modeling}" "numpy pandas yaml" modeling \
  scripts.workflows.run_datasets "$@"
