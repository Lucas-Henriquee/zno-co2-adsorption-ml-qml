#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/environment/conda_tools.sh"
cd "${PROJECT_ROOT}"
# The orchestrator only needs PyYAML; each stage launcher checks its own environment.
launch_in_env "${QML_ZNO_MODEL_ENV:-qml-zno-modeling}" "yaml" modeling scripts.workflows.run_all "$@"
