#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=conda_tools.sh
source "${SCRIPT_DIR}/conda_tools.sh"

PROFILE=""
UPDATE=0
DRY_RUN=0
ENV_OVERRIDE=""
FILE_OVERRIDE=""
MANAGER_OVERRIDE=""
ORIGINAL_CONDA_SUBDIR="${CONDA_SUBDIR:-}"

usage() {
  cat <<'EOF'
Usage: setup_environment.sh PROFILE [OPTIONS]

Create (or, with --update, update) a project environment from environments/*.yaml,
then validate it with check_environment.sh.

Profiles:
  simulation    GPAW 25.7 (OpenMPI), ASE, analysis and notebook tools
                (file: environments/simulation.yaml, default name qml-zno-simulation)
  modeling      Datasets, analysis, scikit-learn, Qiskit, IBM Runtime
                (file: environments/modeling.yaml, default name qml-zno-modeling)
  all           Both profiles with their default names and files

Options:
  --env NAME|PATH  Environment name, or prefix path (contains "/"); alias: --name
  --file FILE      Environment file to use instead of environments/PROFILE.yaml
  --manager CMD    conda, mamba, or micromamba (default: QML_ZNO_CONDA, else the
                   first one found in PATH)
  --update         Apply the file to an existing environment (never prunes packages)
  --dry-run        Solve and show the creation plan without installing
  -h, --help       Show this help message

An existing environment is never changed without --update.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    simulation|modeling|all)
      [[ -z "${PROFILE}" ]] || fail "Only one profile may be selected."
      PROFILE="$1"
      shift
      ;;
    --update)
      UPDATE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --env|--name)
      [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a name or path."
      ENV_OVERRIDE="$2"
      shift 2
      ;;
    --file)
      [[ $# -ge 2 && -n "$2" ]] || fail "--file requires a path."
      FILE_OVERRIDE="$2"
      shift 2
      ;;
    --manager)
      [[ $# -ge 2 && -n "$2" ]] || fail "--manager requires a command."
      MANAGER_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1 (see --help)"
      ;;
  esac
done

[[ -n "${PROFILE}" ]] || fail "Select a profile: simulation, modeling, or all (see --help)."
if [[ "${PROFILE}" == "all" && ( -n "${ENV_OVERRIDE}" || -n "${FILE_OVERRIDE}" ) ]]; then
  fail "--env and --file can only be used with one profile."
fi
if [[ "${UPDATE}" == "1" && "${DRY_RUN}" == "1" ]]; then
  fail "--update and --dry-run cannot be combined (environment updates have no dry run)."
fi
if [[ -n "${FILE_OVERRIDE}" ]]; then
  [[ -f "${FILE_OVERRIDE}" ]] || fail "Environment file not found: ${FILE_OVERRIDE}"
  FILE_OVERRIDE="$(cd "$(dirname "${FILE_OVERRIDE}")" && pwd)/$(basename "${FILE_OVERRIDE}")"
fi

select_manager "${MANAGER_OVERRIDE}"
echo "[setup] Package manager: ${MANAGER} (${MANAGER_KIND})"

default_name() {
  case "$1" in
    simulation) echo "qml-zno-simulation" ;;
    modeling) echo "qml-zno-modeling" ;;
  esac
}

# create|update ENVIRONMENT FILE [--dry-run]
solve() {
  local action="$1" environment="$2" file="$3"
  set_target "${environment}"
  local -a command
  case "${MANAGER_KIND}:${action}" in
    micromamba:create) command=("${MANAGER}" create --yes "${TARGET[@]}" --file "${file}") ;;
    micromamba:update) command=("${MANAGER}" install --yes "${TARGET[@]}" --file "${file}") ;;
    *:create) command=("${MANAGER}" env create "${TARGET[@]}" --file "${file}") ;;
    *:update) command=("${MANAGER}" env update "${TARGET[@]}" --file "${file}") ;;
  esac
  if [[ "${DRY_RUN}" == "1" ]]; then
    command+=(--dry-run)
  fi
  echo "[run] ${command[*]}"
  "${command[@]}"
}

install_profile() {
  local profile="$1"
  local environment_file="${FILE_OVERRIDE:-${PROJECT_ROOT}/environments/${profile}.yaml}"
  local environment_name="${ENV_OVERRIDE:-$(default_name "${profile}")}"

  [[ -f "${environment_file}" ]] || fail "Missing environment file: ${environment_file}"

  # GPAW 25.7 has no osx-arm64 build: build the simulation environment from osx-64.
  local subdir=""
  if [[ "${profile}" == "simulation" ]]; then
    subdir="$(simulation_subdir)"
  fi
  if [[ -n "${subdir}" && -z "${ORIGINAL_CONDA_SUBDIR}" ]]; then
    echo "[setup] Apple Silicon: using ${subdir} packages (needs Rosetta 2: softwareupdate --install-rosetta)."
    export CONDA_SUBDIR="${subdir}"
  elif [[ -n "${ORIGINAL_CONDA_SUBDIR}" ]]; then
    export CONDA_SUBDIR="${ORIGINAL_CONDA_SUBDIR}"
  else
    unset CONDA_SUBDIR
  fi

  if env_exists "${environment_name}"; then
    if [[ "${UPDATE}" == "1" ]]; then
      echo "[update] ${environment_name} from ${environment_file}"
      solve update "${environment_name}" "${environment_file}"
    else
      echo "[skip] Environment '${environment_name}' already exists; it is left unchanged."
      echo "       Add --update to apply ${environment_file}, or --env NAME for a new one."
    fi
  else
    [[ "${UPDATE}" == "0" ]] || fail "Environment '${environment_name}' does not exist; drop --update to create it."
    echo "[create] ${environment_name} from ${environment_file}"
    solve create "${environment_name}" "${environment_file}"
  fi

  if [[ "${DRY_RUN}" == "0" ]]; then
    "${SCRIPT_DIR}/check_environment.sh" "${profile}" --env "${environment_name}" --manager "${MANAGER}"
  fi
}

if [[ "${PROFILE}" == "all" ]]; then
  install_profile simulation
  install_profile modeling
else
  install_profile "${PROFILE}"
fi
