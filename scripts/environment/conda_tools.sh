# Shared helpers for setup_environment.sh and check_environment.sh (sourced, not run).
#
# Package manager: QML_ZNO_CONDA (a command name or path) or --manager, otherwise the
# first of conda, mamba, micromamba found in PATH. An environment is given by name
# (-n NAME) or, when the value contains a "/", by prefix path (-p PATH).

fail() {
  echo "[error] $*" >&2
  exit 1
}

# Sets MANAGER (the command) and MANAGER_KIND (conda, mamba or micromamba).
select_manager() {
  local requested="${1:-${QML_ZNO_CONDA:-}}"
  if [[ -n "${requested}" ]]; then
    command -v "${requested}" >/dev/null 2>&1 \
      || fail "Package manager '${requested}' (from --manager or QML_ZNO_CONDA) was not found."
    MANAGER="${requested}"
  else
    MANAGER=""
    local candidate
    for candidate in conda mamba micromamba; do
      if command -v "${candidate}" >/dev/null 2>&1; then
        MANAGER="${candidate}"
        break
      fi
    done
    [[ -n "${MANAGER}" ]] || fail "None of conda, mamba or micromamba was found in PATH.
        Install Miniforge (https://github.com/conda-forge/miniforge) or micromamba, open a
        new shell, and run this script again."
  fi
  case "$(basename "${MANAGER}")" in
    micromamba*) MANAGER_KIND="micromamba" ;;
    mamba*) MANAGER_KIND="mamba" ;;
    *) MANAGER_KIND="conda" ;;
  esac
}

# Sets TARGET to (-n NAME) or (-p PATH) for an environment name or prefix.
# Plain arrays only: macOS ships bash 3.2.
set_target() {
  if [[ "$1" == */* ]]; then
    TARGET=(-p "$1")
  else
    TARGET=(-n "$1")
  fi
}

# Runs a command inside the environment ($1), with the manager's activation scripts.
run_in_env() {
  set_target "$1"
  shift
  if [[ "${MANAGER_KIND}" == "conda" ]]; then
    "${MANAGER}" run --no-capture-output "${TARGET[@]}" "$@"
  else
    "${MANAGER}" run "${TARGET[@]}" "$@"
  fi
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

# Root launchers: check the modules ($2, space separated) in the environment ($1), record
# the environment for the run manifest, then replace this shell by `python -m $4 ...`.
# $3 is the setup/check profile named in the error messages.
launch_in_env() {
  local environment="$1" modules="$2" profile="$3" module="$4"
  shift 4
  select_manager
  local missing
  # shellcheck disable=SC2086  # the module list is split on purpose
  if ! missing="$(run_in_env "${environment}" python -c \
      'import importlib.util as u, sys; print(" ".join(m for m in sys.argv[1:] if u.find_spec(m) is None))' \
      ${modules} 2>/dev/null)"; then
    fail "Environment '${environment}' is not usable with ${MANAGER}. Create it with
        ./scripts/environment/setup_environment.sh ${profile}, or select an existing one
        through the environment variable described in environments/README.md."
  fi
  [[ -z "${missing}" ]] || fail "Environment '${environment}' lacks: ${missing}.
        Check it with ./scripts/environment/check_environment.sh ${profile} --env ${environment}."
  # The manifest records the name; a prefix path is recorded by its last component, as the
  # workflows do for an interpreter inside .../envs/<name>.
  export QML_ZNO_ACTIVE_ENVIRONMENT="$(basename "${environment}")"
  set_target "${environment}"
  if [[ "${MANAGER_KIND}" == "conda" ]]; then
    exec "${MANAGER}" run --no-capture-output "${TARGET[@]}" python -m "${module}" "$@"
  fi
  exec "${MANAGER}" run "${TARGET[@]}" python -m "${module}" "$@"
}
