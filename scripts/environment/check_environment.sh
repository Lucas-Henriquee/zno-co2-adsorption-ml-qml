#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=conda_tools.sh
source "${SCRIPT_DIR}/conda_tools.sh"

PROFILE=""
ENV_OVERRIDE=""
PYTHON_OVERRIDE=""
MANAGER_OVERRIDE=""
CHECK_MPI=0

usage() {
  cat <<'EOF'
Usage: check_environment.sh PROFILE [OPTIONS]

Validate an environment without running a scientific calculation.

Profiles:
  simulation    GPAW, ASE, PAW datasets, compiled features, and the APIs the
                simulation scripts use (run_simulations.sh, notebooks)
  analysis      numpy, scipy, pandas, matplotlib, PyYAML, and ASE
                (run_analysis.sh; also enough for run_datasets.sh)
  modeling      analysis packages plus scikit-learn and the Qiskit stack
                (run_models.sh, run_all.sh)

Options:
  --env NAME|PATH  Environment to check, by name or prefix path. Default: the
                   launcher variable (QML_ZNO_SIM_ENV, QML_ZNO_ANALYSIS_ENV,
                   QML_ZNO_MODEL_ENV) when set, else qml-zno-simulation or
                   qml-zno-modeling
  --python PATH    Check this interpreter directly, without conda (e.g. a pip venv)
  --manager CMD    conda, mamba, or micromamba (default: QML_ZNO_CONDA, else the
                   first one found in PATH)
  --mpi            Add a two-process MPI check (simulation only)
  -h, --help       Show this help message

Examples:
  check_environment.sh simulation --env gpaw-dft --mpi
  check_environment.sh analysis --python .venv/bin/python
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    simulation|analysis|modeling)
      [[ -z "${PROFILE}" ]] || fail "Only one profile may be selected."
      PROFILE="$1"
      shift
      ;;
    --env)
      [[ $# -ge 2 && -n "$2" ]] || fail "--env requires a name or path."
      ENV_OVERRIDE="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 && -n "$2" ]] || fail "--python requires an interpreter path."
      PYTHON_OVERRIDE="$2"
      shift 2
      ;;
    --manager)
      [[ $# -ge 2 && -n "$2" ]] || fail "--manager requires a command."
      MANAGER_OVERRIDE="$2"
      shift 2
      ;;
    --mpi)
      CHECK_MPI=1
      shift
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

[[ -n "${PROFILE}" ]] || fail "Select a profile: simulation, analysis, or modeling (see --help)."
[[ "${PROFILE}" == "simulation" || "${CHECK_MPI}" == "0" ]] \
  || fail "--mpi is only valid for the simulation profile."
[[ -z "${PYTHON_OVERRIDE}" || -z "${ENV_OVERRIDE}" ]] \
  || fail "Use either --env or --python, not both."

case "${PROFILE}" in
  simulation) DEFAULT_ENV="${QML_ZNO_SIM_ENV:-qml-zno-simulation}" ;;
  analysis) DEFAULT_ENV="${QML_ZNO_ANALYSIS_ENV:-${QML_ZNO_MODEL_ENV:-qml-zno-modeling}}" ;;
  modeling) DEFAULT_ENV="${QML_ZNO_MODEL_ENV:-qml-zno-modeling}" ;;
esac
ENVIRONMENT_NAME="${ENV_OVERRIDE:-${DEFAULT_ENV}}"

CACHE_ROOT="${TMPDIR:-/tmp}/qml-zno-environment-check"
mkdir -p "${CACHE_ROOT}/matplotlib"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export MPLBACKEND=Agg

if [[ -n "${PYTHON_OVERRIDE}" ]]; then
  [[ -x "${PYTHON_OVERRIDE}" ]] || fail "Interpreter not found or not executable: ${PYTHON_OVERRIDE}"
  PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_OVERRIDE}")" && pwd)"
  PYTHON_PATH_ABS="${PYTHON_BIN_DIR}/$(basename "${PYTHON_OVERRIDE}")"
  # Commands run with the interpreter's bin directory first in PATH (gpaw, mpiexec).
  in_environment() {
    PATH="${PYTHON_BIN_DIR}:${PATH}" "$@"
  }
  py() {
    in_environment "${PYTHON_PATH_ABS}" "$@"
  }
  echo "[check] Profile: ${PROFILE}"
  echo "[check] Interpreter: ${PYTHON_PATH_ABS}"
else
  select_manager "${MANAGER_OVERRIDE}"
  if ! env_exists "${ENVIRONMENT_NAME}" \
    || ! run_in_env "${ENVIRONMENT_NAME}" python --version >/dev/null 2>&1; then
    fail "Environment '${ENVIRONMENT_NAME}' was not found by ${MANAGER} or has no Python.
        List environments with '${MANAGER} env list'; pass --env NAME for another name,
        or create it with ./scripts/environment/setup_environment.sh ${PROFILE/analysis/modeling}."
  fi
  in_environment() {
    run_in_env "${ENVIRONMENT_NAME}" "$@"
  }
  py() {
    in_environment python "$@"
  }
  echo "[check] Profile: ${PROFILE}"
  echo "[check] Environment: ${ENVIRONMENT_NAME} (${MANAGER_KIND}: ${MANAGER})"
fi

echo "[check] Python packages, versions, and APIs used by the code"
py - "${PROFILE}" <<'PY'
import importlib
import inspect
import os
import platform
import site
import sys

profile = sys.argv[1]
problems = []
warnings = []
checked = {}  # module name -> module, for the user-site check at the end


def version_tuple(text):
    parts = []
    for item in str(text).split(".")[:3]:
        digits = "".join(ch for ch in item if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def load(name, needed_for):
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # ImportError, or a broken compiled extension
        problems.append(f"{name}: cannot be imported ({exc}); needed for {needed_for}")
        print(f"  {name}: MISSING")
        return None
    checked[name] = module
    print(f"  {name}: {getattr(module, '__version__', 'available')}")
    return module


def optional(name, needed_for):
    try:
        module = importlib.import_module(name)
    except Exception:
        print(f"  {name}: not installed (needed only for {needed_for})")
        return None
    checked[name] = module
    print(f"  {name}: {getattr(module, '__version__', 'available')}")
    return module


def user_site_warnings():
    """Checked modules imported from the user site-packages instead of the environment."""
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if not user_site:
        return
    user_site = os.path.realpath(user_site)
    for name, module in checked.items():
        location = getattr(module, "__file__", None)
        if not location:
            continue
        location = os.path.realpath(location)
        if location == user_site or location.startswith(user_site + os.sep):
            warnings.append(
                f"{name} is imported from the user site-packages ({location}), not from the "
                f"environment prefix {sys.prefix}; rerun with PYTHONNOUSERSITE=1 to check the "
                f"environment without the user packages"
            )


def probe(description, check):
    try:
        ok = check()
    except Exception as exc:
        ok = False
        description = f"{description} ({exc})"
    if not ok:
        problems.append(f"API missing: {description}")


print(f"  python: {platform.python_version()} ({sys.executable})")
if sys.version_info < (3, 10):
    problems.append("Python >= 3.10 is required")

core = {
    "numpy": "all workflows",
    "scipy": "ASE, GPAW and the grouped model statistics",
    "pandas": "analysis, datasets and models",
    "matplotlib": "figures",
    "yaml": "the YAML configurations (package: pyyaml)",
    "ase": "geometries, CO2RR corrections, MD/structure analysis and run_status",
}
for name, use in core.items():
    load(name, use)

ase = sys.modules.get("ase")
if ase is not None:
    v = version_tuple(ase.__version__)
    if v < (3, 23):
        problems.append(f"ASE {ase.__version__} < 3.23: ase.mep (NEB) and the MD APIs need >= 3.23")
    elif not (3, 28) <= v < (3, 30):
        warnings.append(f"ASE {ase.__version__} is outside the tested 3.28-3.29 range")

if profile == "simulation":
    gpaw = load("gpaw", "DFT, TDDFT, MD, NEB and CO2RR")
    if gpaw is not None:
        v = version_tuple(gpaw.__version__)
        if v < (24, 6):
            problems.append(f"GPAW {gpaw.__version__} < 24.6 is not supported")
        elif v[:2] != (25, 7):
            warnings.append(f"GPAW {gpaw.__version__}: the production runs used 25.7")
        probe("gpaw.lrtddft.LrTDDFT (TDDFT)", lambda: hasattr(importlib.import_module("gpaw.lrtddft"), "LrTDDFT"))
        probe("gpaw.PW, FermiDirac, Mixer (DFT)", lambda: all(hasattr(gpaw, n) for n in ("PW", "FermiDirac", "Mixer")))
    if ase is not None:
        def neb_api():
            from ase.mep import NEB
            return "apply_constraint" in inspect.signature(NEB.interpolate).parameters
        def md_api():
            from ase.md.langevin import Langevin
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, force_temperature
            return (
                "rng" in inspect.signature(MaxwellBoltzmannDistribution).parameters
                and {"rng", "fixcm"} <= set(inspect.signature(Langevin.__init__).parameters)
                and "unit" in inspect.signature(force_temperature).parameters
            )
        def thermo_api():
            from ase.thermochemistry import HarmonicThermo, IdealGasThermo  # noqa: F401
            return "ignore_imag_modes" in inspect.signature(HarmonicThermo.__init__).parameters
        probe("ase.mep.NEB.interpolate(apply_constraint=...) (NEB)", neb_api)
        probe("ase.utils.forcecurve.fit_images (NEB)", lambda: hasattr(importlib.import_module("ase.utils.forcecurve"), "fit_images"))
        probe("MaxwellBoltzmannDistribution(rng=), Langevin(rng=, fixcm=), force_temperature(unit=) (MD)", md_api)
        probe("HarmonicThermo(ignore_imag_modes=), IdealGasThermo (CO2RR)", thermo_api)
    for name in ("nbconvert", "nbclient", "ipykernel"):
        optional(name, "notebooks/run_zno_notebook.sh")
    optional("pytest", "the test suite")
    optional("sklearn", "the test suite and the classical models; package: scikit-learn")

elif profile == "analysis":
    optional("sklearn", "the test suite and the classical models; package: scikit-learn")
    optional("pytest", "the test suite")

elif profile == "modeling":
    load("sklearn", "SVM baselines and grouped evaluation (package: scikit-learn)")
    load("joblib", "parallel grouped evaluation")
    load("qiskit", "QSVM circuits")
    load("qiskit_aer", "local QSVM simulation (package: qiskit-aer)")
    load("qiskit_machine_learning", "QSVM kernels (package: qiskit-machine-learning)")
    load("qiskit_ibm_runtime", "IBM hardware runs (package: qiskit-ibm-runtime)")
    if "qiskit" in sys.modules:
        probe("qiskit.circuit.library.zz_feature_map (qiskit >= 1.3)",
              lambda: hasattr(importlib.import_module("qiskit.circuit.library"), "zz_feature_map"))
        v = version_tuple(sys.modules["qiskit"].__version__)
        if v[0] >= 2:
            problems.append("qiskit >= 2 is not supported by qiskit-machine-learning 0.8")
    if "qiskit_machine_learning" in sys.modules:
        probe("qiskit_machine_learning.kernels.FidelityStatevectorKernel",
              lambda: hasattr(importlib.import_module("qiskit_machine_learning.kernels"), "FidelityStatevectorKernel"))
    if "qiskit_ibm_runtime" in sys.modules:
        probe("qiskit_ibm_runtime.QiskitRuntimeService, SamplerV2",
              lambda: all(hasattr(sys.modules["qiskit_ibm_runtime"], n) for n in ("QiskitRuntimeService", "SamplerV2")))
    for name in ("nbconvert", "nbclient", "ipykernel"):
        optional(name, "headless notebook execution")
    optional("pytest", "the test suite")

user_site_warnings()
for message in warnings:
    print(f"[warn] {message}")
if problems:
    for message in problems:
        print(f"[error] {message}", file=sys.stderr)
    sys.exit(1)
PY

if [[ "${PROFILE}" == "simulation" ]]; then
  echo "[check] GPAW compiled features"
  in_environment gpaw info

  echo "[check] PAW datasets"
  py - <<'PY'
import sys

from gpaw.setup_data import search_for_file

missing = []
for element in ["H", "C", "O", "Zn", "Ti", "Ce"]:
    try:
        path, _ = search_for_file(f"{element}.PBE")
    except Exception as exc:
        missing.append(element)
        print(f"  {element}: MISSING ({exc})")
    else:
        print(f"  {element}: {path}")
if missing:
    print("[error] PAW datasets not found. Install gpaw-data in the environment or set "
          "GPAW_SETUP_PATH (see 'gpaw install-data --help').", file=sys.stderr)
    sys.exit(1)
PY

  echo "[check] One-process GPAW dry run"
  OMP_NUM_THREADS=1 in_environment gpaw python --dry-run=1 -c \
    'from ase import Atoms; from gpaw import GPAW; atoms = Atoms("H", cell=[6, 6, 6], pbc=False); atoms.center(); atoms.calc = GPAW(mode="fd", xc="PBE", txt=None); atoms.get_potential_energy()'

  if [[ "${CHECK_MPI}" == "1" ]]; then
    echo "[check] Two-process MPI run (gpaw.mpi.world.size must be 2)"
    if ! OMP_NUM_THREADS=1 in_environment mpiexec -n 2 gpaw python -c \
      'from gpaw.mpi import world; assert world.size == 2, f"world.size={world.size}: GPAW was built without MPI"; print(f"  rank {world.rank} of {world.size}")'; then
      fail "The two-process MPI check failed. See 'Common errors' in environments/README.md
        (serial GPAW build, mpiexec from another MPI, fewer than 2 slots, running as root)."
    fi
    echo "[check] Two-process GPAW dry run"
    OMP_NUM_THREADS=1 in_environment mpiexec -n 2 gpaw python --dry-run=2 -c \
      'from ase import Atoms; from gpaw import GPAW; atoms = Atoms("H", cell=[6, 6, 6], pbc=False); atoms.center(); atoms.calc = GPAW(mode="fd", xc="PBE", txt=None); atoms.get_potential_energy()'
  fi
fi

[[ -d "${PROJECT_ROOT}/results" ]] || fail "Missing results directory: ${PROJECT_ROOT}/results"
[[ -w "${PROJECT_ROOT}/results" ]] || fail "Results directory is not writable: ${PROJECT_ROOT}/results"

if [[ -n "${PYTHON_OVERRIDE}" ]]; then
  echo "[ok] ${PYTHON_PATH_ABS} passed the ${PROFILE} checks."
else
  echo "[ok] Environment '${ENVIRONMENT_NAME}' passed the ${PROFILE} checks."
fi
