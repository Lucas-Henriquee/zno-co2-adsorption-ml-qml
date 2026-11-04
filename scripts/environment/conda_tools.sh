# Shared helpers for setup_environment.sh and check_environment.sh (sourced, not run).
#
# Package manager: QML_ZNO_CONDA (a command name or path) or --manager, otherwise the
# first of conda, mamba, micromamba found in PATH. An environment is given by name
# (-n NAME) or, when the value contains a "/", by prefix path (-p PATH).
# Plain arrays only: macOS ships bash 3.2.

fail() {
  echo "[error] $*" >&2
  exit 1
}

# Sets MANAGER (the command) and MANAGER_KIND (conda, mamba or micromamba); returns 1
# with MANAGER empty when the requested manager (or any manager) is missing. No output.
find_manager() {
  local requested="${1:-${QML_ZNO_CONDA:-}}"
  MANAGER=""
  MANAGER_NO_CAPTURE=""
  if [[ -n "${requested}" ]]; then
    command -v "${requested}" >/dev/null 2>&1 || return 1
    MANAGER="${requested}"
  else
    local candidate
    for candidate in conda mamba micromamba; do
      if command -v "${candidate}" >/dev/null 2>&1; then
        MANAGER="${candidate}"
        break
      fi
    done
    [[ -n "${MANAGER}" ]] || return 1
  fi
  case "$(basename "${MANAGER}")" in
    micromamba*) MANAGER_KIND="micromamba" ;;
    mamba*) MANAGER_KIND="mamba" ;;
    *) MANAGER_KIND="conda" ;;
  esac
}

# find_manager, failing with installation advice.
select_manager() {
  local requested="${1:-${QML_ZNO_CONDA:-}}"
  if ! find_manager "${requested}"; then
    [[ -z "${requested}" ]] \
      || fail "Package manager '${requested}' (from --manager or QML_ZNO_CONDA) was not found."
    fail "None of conda, mamba or micromamba was found in PATH.
        Install Miniforge (https://github.com/conda-forge/miniforge) or micromamba, open a
        new shell, and run this script again."
  fi
}

# Sets MANAGER_NO_CAPTURE to yes when `MANAGER run` accepts --no-capture-output: conda and
# mamba 1.x (which forwards to `conda run`, capturing output by default) do; micromamba and
# mamba >= 2 do not. The mamba case is read from `mamba run --help` once.
detect_no_capture() {
  [[ -z "${MANAGER_NO_CAPTURE:-}" ]] || return 0
  case "${MANAGER_KIND}" in
    conda) MANAGER_NO_CAPTURE="yes" ;;
    micromamba) MANAGER_NO_CAPTURE="no" ;;
    *)
      if "${MANAGER}" run --help 2>&1 | grep -q -- "--no-capture-output"; then
        MANAGER_NO_CAPTURE="yes"
      else
        MANAGER_NO_CAPTURE="no"
      fi
      ;;
  esac
}

# Sets TARGET to (-n NAME) or (-p PATH) for an environment name or prefix.
set_target() {
  if [[ "$1" == */* ]]; then
    TARGET=(-p "$1")
  else
    TARGET=(-n "$1")
  fi
}

# Sets RUN_PREFIX to `MANAGER run [--no-capture-output] -n NAME|-p PATH` for environment $1.
set_run_prefix() {
  set_target "$1"
  detect_no_capture
  if [[ "${MANAGER_NO_CAPTURE}" == "yes" ]]; then
    RUN_PREFIX=("${MANAGER}" run --no-capture-output "${TARGET[@]}")
  else
    RUN_PREFIX=("${MANAGER}" run "${TARGET[@]}")
  fi
}

# Runs a command inside the environment ($1), with the manager's activation scripts.
run_in_env() {
  set_run_prefix "$1"
  shift
  "${RUN_PREFIX[@]}" "$@"
}

# True when the environment exists (by prefix directory or by name in "env list").
env_exists() {
  local environment="$1"
  if [[ "${environment}" == */* ]]; then
    [[ -x "${environment}/bin/python" || -d "${environment}/conda-meta" ]]
    return
  fi
  "${MANAGER}" env list 2>/dev/null | awk -v name="${environment}" '
    NF == 0 || $1 ~ /^#/ { next }
    {
      path = $NF
      n = split(path, parts, "/")
      if ($1 == name || parts[n] == name) { found = 1 }
    }
    END { exit found ? 0 : 1 }
  '
}

# macOS on Apple Silicon: GPAW 25.7 has no osx-arm64 build on conda-forge, so the
# simulation environment is built from osx-64 packages (runs under Rosetta 2).
simulation_subdir() {
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "osx-64"
  fi
}

# True when one of the arguments is -h or --help.
wants_help() {
  local argument
  for argument in "$@"; do
    case "${argument}" in
      -h|--help) return 0 ;;
    esac
  done
  return 1
}

# Usage of a root launcher (no conda needed), then exit 0. When a manager and the
# environment exist, the workflow's own --help follows.
launcher_help() {
  local environment="$1" profile="$2" module="$3"
  shift 3
  echo "Usage: ./$(basename "$0") [ARGUMENTS...]"
  echo
  echo "Runs 'python -m ${module} ARGUMENTS...' in the conda environment '${environment}'."
  echo "Select another environment with the variable in environments/README.md; create it"
  echo "with ./scripts/environment/setup_environment.sh ${profile}."
  echo
  if find_manager && env_exists "${environment}"; then
    set_run_prefix "${environment}"
    "${RUN_PREFIX[@]}" python -m "${module}" "$@" || true
  else
    echo "Workflow options: python -m ${module} --help (inside that environment)."
  fi
  exit 0
}

# Root launchers: check the modules ($2, space separated) in the environment ($1), record
# the environment for the run manifest, then replace this shell by `python -m $4 ...`.
# $3 is the setup/check profile named in the messages. -h/--help: launcher_help.
launch_in_env() {
  local environment="$1" modules="$2" profile="$3" module="$4"
  shift 4
  if wants_help "$@"; then
    launcher_help "${environment}" "${profile}" "${module}" "$@"
  fi
  select_manager
  local missing errors status=0
  errors="$(mktemp "${TMPDIR:-/tmp}/qml_zno_probe.XXXXXX")"
  # shellcheck disable=SC2086  # the module list is split on purpose
  missing="$(run_in_env "${environment}" python -c \
      'import importlib.util as u, sys; print(" ".join(m for m in sys.argv[1:] if u.find_spec(m) is None))' \
      ${modules} 2>"${errors}")" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    # Keep the manager's own message (wrong prefix, solver or activation error).
    cat "${errors}" >&2
    rm -f "${errors}"
    fail "Environment '${environment}' is not usable with ${MANAGER} (exit ${status}). Create it with
        ./scripts/environment/setup_environment.sh ${profile}, or select an existing one
        through the environment variable described in environments/README.md."
  fi
  rm -f "${errors}"
  [[ -z "${missing}" ]] || fail "Environment '${environment}' lacks: ${missing}.
        Check it with ./scripts/environment/check_environment.sh ${profile} --env ${environment}."
  # The manifest records the name; a prefix path is recorded by its last component, as the
  # workflows do for an interpreter inside .../envs/<name>.
  QML_ZNO_ACTIVE_ENVIRONMENT="$(basename "${environment}")"
  export QML_ZNO_ACTIVE_ENVIRONMENT
  set_run_prefix "${environment}"
  exec "${RUN_PREFIX[@]}" python -m "${module}" "$@"
}
