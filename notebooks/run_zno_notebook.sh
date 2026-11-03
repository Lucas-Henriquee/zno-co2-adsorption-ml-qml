#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECK_SCRIPT="${PROJECT_DIR}/scripts/environment/check_environment.sh"
source "${PROJECT_DIR}/scripts/environment/conda_tools.sh"
SIM_ENVIRONMENT="${QML_ZNO_SIM_ENV:-qml-zno-simulation}"

INPUT_NOTEBOOK="${1:-zno-co2-lucas.ipynb}"
OUTPUT_NOTEBOOK="${2:-${INPUT_NOTEBOOK%.ipynb}.executed.ipynb}"
# Per-cell timeout in seconds; -1 disables it (a DFT/MD cell can run for days).
NB_TIMEOUT="${NB_TIMEOUT:--1}"

[[ -f "${SCRIPT_DIR}/${INPUT_NOTEBOOK}" ]] || fail "Input notebook does not exist: ${SCRIPT_DIR}/${INPUT_NOTEBOOK}"
[[ ! -e "${SCRIPT_DIR}/${OUTPUT_NOTEBOOK}" ]] \
  || fail "Refusing to overwrite ${SCRIPT_DIR}/${OUTPUT_NOTEBOOK}; give another output name as the second argument."

select_manager
cd "${SCRIPT_DIR}"

# The analysis notebook only reads saved results; the simulation notebook needs GPAW.
# Environment: QML_ZNO_ANALYSIS_ENV, then QML_ZNO_SIM_ENV for it (the simulation
# environment also has the analysis packages); QML_ZNO_SIM_ENV for the simulation notebook.
if [[ "${INPUT_NOTEBOOK}" == *analysis* ]]; then
  ENVIRONMENT_NAME="${QML_ZNO_ANALYSIS_ENV:-${SIM_ENVIRONMENT}}"
  "${CHECK_SCRIPT}" analysis --env "${ENVIRONMENT_NAME}"
else
  ENVIRONMENT_NAME="${SIM_ENVIRONMENT}"
  "${CHECK_SCRIPT}" simulation --env "${ENVIRONMENT_NAME}"
fi
run_in_env "${ENVIRONMENT_NAME}" python -c 'import nbconvert, nbclient, ipykernel' >/dev/null 2>&1 \
  || fail "'${ENVIRONMENT_NAME}' lacks nbconvert, nbclient or ipykernel, which headless execution needs.
        Run the notebook interactively, or the same methods with the root launchers."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
set_target "${ENVIRONMENT_NAME}"
if [[ "${MANAGER_KIND}" == "conda" ]]; then
  RUN=("${MANAGER}" run --no-capture-output "${TARGET[@]}")
else
  RUN=("${MANAGER}" run "${TARGET[@]}")
fi
exec "${RUN[@]}" python -m nbconvert --to notebook --execute "${INPUT_NOTEBOOK}" \
  --output "${OUTPUT_NOTEBOOK}" --ExecutePreprocessor.timeout="${NB_TIMEOUT}"
