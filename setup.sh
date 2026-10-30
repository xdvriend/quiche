#!/usr/bin/env bash

# Run this with `bash setup.sh`. Sourcing it would apply `set -e` and `exit` to
# your own shell, which closes the terminal on the first error.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    echo "Don't source this script; run it with: bash ${BASH_SOURCE[0]}" >&2
    return 1
fi

set -Eeuo pipefail

# Keep a copy of all output, and say where it failed, so an error is never lost.
log="$HOME/quiche-setup.log"
exec > >(tee "$log") 2>&1
on_error='echo "setup.sh failed at line $LINENO: $BASH_COMMAND (full log: $log)" >&2'
trap "$on_error" ERR

# Sets up the virtual environment for the QUICHE + CUDA-Q demos (demo/*.ipynb) in
# ~/.venvs/quiche-cudaq: QUICHE from this repository (editable, with its CUDA-Q
# extra) and the demo with everything it needs (demo/pyproject.toml: the ORCA
# Python interface, Open MPI, Plotly, Jupyter, ...), registered as an IPython
# kernel. It can be run from any directory.
#
# Relies on createVirtualEnv / activateVirtualEnv / createIPyKernel being
# defined in ~/.bashrc. Those definitions must sit ABOVE the terminal guard
# there, otherwise a non-interactive shell returns before reaching them.
#
# Options (environment variables):
#   PYTHON_BIN=/path/to/python ./setup.sh  # interpreter to build the venv from (default: searched for, see below)
#   KERNEL_NAME=my-kernel ./setup.sh   # Jupyter kernel name (default: cudaq-logical, as used by the notebooks)
#   BASHRC=/path/to/.bashrc ./setup.sh # where the venv functions are defined (default: ~/.bashrc)
#   WITHOUT_CUDAQ=1 ./setup.sh         # skip CUDA-Q, and with it the demo (needed on Python >= 3.14, see below)
#   ./setup.sh run                     # start JupyterLab afterwards
PYTHON_BIN="${PYTHON_BIN:-}"
KERNEL_NAME="${KERNEL_NAME:-cudaq-logical}"
BASHRC="${BASHRC:-$HOME/.bashrc}"
WITHOUT_CUDAQ="${WITHOUT_CUDAQ:-0}"

env_name="quiche-cudaq"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# bashrc isn't written for `set -eu`, so relax everything (and the error report)
# while sourcing it.
set +eux
trap - ERR
source "$BASHRC"
trap "$on_error" ERR
set -eu

for fn in createVirtualEnv activateVirtualEnv createIPyKernel; do
    if ! declare -F "$fn" >/dev/null; then
        echo "Error: '$fn' is not defined after sourcing $BASHRC." >&2
        echo "Move the venv functions above the terminal guard in $BASHRC." >&2
        exit 1
    fi
done

# Find a Python to build the venv from. It must be:
#  - Python >= 3.12 (QUICHE), and < 3.14 unless WITHOUT_CUDAQ=1: CUDA-Q 0.15, the
#    version cudaq-algorithms 0.1.0 supports, has no wheels for Python 3.14 yet;
#  - complete: QUICHE compiles a C++ extension against the Python C headers
#    (Python.h), which some system Pythons leave out (on RHEL they are in the
#    separate python3.X-devel package), and the venv needs venv and ensurepip.
max_minor=13
if [ "$WITHOUT_CUDAQ" = "1" ]; then
    max_minor=99
fi

python_problem() {
    # Print why the Python "$1" can't be used, or nothing if it can.
    "$1" - "$max_minor" 2>/dev/null <<'EOF' || echo "does not run"
import importlib.util, os, sys, sysconfig
major, minor = sys.version_info[:2]
if not (3, 12) <= (major, minor) <= (3, int(sys.argv[1])):
    print(f"Python {major}.{minor} (need 3.12 or 3.13 for CUDA-Q)")
elif not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")):
    print(f"Python {major}.{minor} without its C headers (Python.h)")
elif not (importlib.util.find_spec("venv") and importlib.util.find_spec("ensurepip")):
    print(f"Python {major}.{minor} without the venv/ensurepip modules")
EOF
}

# Candidates: PYTHON_BIN if set; otherwise the Pythons on PATH (including an
# active conda environment), then those of conda environments, pyenv and uv.
candidates=()
if [ -n "$PYTHON_BIN" ]; then
    candidates=("$PYTHON_BIN")
else
    for name in python3.13 python3.12 python3 python; do
        if command -v "$name" >/dev/null 2>&1; then
            candidates+=("$(command -v "$name")")
        fi
    done
    conda_base="$(conda info --base 2>/dev/null || true)"
    for path in "$HOME"/.conda-envs/*/bin/python3 "$HOME"/.conda/envs/*/bin/python3 \
        "${conda_base:-/nonexistent}"/envs/*/bin/python3 \
        "${conda_base:-/nonexistent}"/bin/python3 "$HOME"/.pyenv/versions/*/bin/python3 \
        "$HOME"/.local/share/uv/python/*/bin/python3; do
        if [ -x "$path" ]; then
            candidates+=("$path")
        fi
    done
fi

python_bin=""
for candidate in ${candidates[@]+"${candidates[@]}"}; do
    problem="$(python_problem "$candidate")"
    if [ -z "$problem" ]; then
        python_bin="$candidate"
        break
    fi
    echo "Skipping $candidate: $problem"
done

if [ -z "$python_bin" ]; then
    echo "Error: no suitable Python found (see the skipped interpreters above)." >&2
    echo "Without admin rights, conda can provide a complete Python 3.12:" >&2
    echo "    conda create -y -p ~/.conda-envs/py312 python=3.12" >&2
    echo "after which running this script again finds it." >&2
    exit 1
fi
echo "Using $python_bin ($("$python_bin" --version 2>&1))"

# A venv from an earlier run may have been built on an incomplete Python.
venv_python="$VENV_HOME/$env_name/bin/python"
if [ -x "$venv_python" ]; then
    problem="$(python_problem "$venv_python")"
    if [ -n "$problem" ]; then
        echo "Error: the existing venv $VENV_HOME/$env_name was built on a $problem." >&2
        echo "Remove it and run this script again: rm -rf $VENV_HOME/$env_name" >&2
        exit 1
    fi
fi

# QUICHE compiles QuEST with OpenMP. On macOS Apple clang has no OpenMP of its
# own, so point CMake at Homebrew's libomp.
if [ "$(uname -s)" = "Darwin" ] && [ -z "${OpenMP_ROOT:-}" ]; then
    libomp="$(brew --prefix libomp 2>/dev/null || true)"
    if [ -z "$libomp" ] || [ ! -d "$libomp" ]; then
        echo "Error: QUICHE needs OpenMP to build on macOS. Install it with: brew install libomp" >&2
        exit 1
    fi
    export OpenMP_ROOT="$libomp"
fi

set -x

# Create the venv if it doesn't exist yet (createVirtualEnv refuses if it
# does, so re-running this script just reuses it).
if [ ! -d "$VENV_HOME/$env_name" ]; then
    createVirtualEnv "$env_name" "$python_bin"
fi

# venv's activate script can trip over `set -u` (e.g. an unset PS1 in a
# non-interactive shell), so relax it just for the activation.
set +u
activateVirtualEnv "$env_name"
set -u

python --version
python -m pip install --upgrade pip

# QUICHE from this repository, editable, with its CUDA-Q extra (cudaq, cudaq-algorithms),
# and the demo with its dependencies (demo/pyproject.toml), also editable. Both go in
# one pip command: the demo depends on QUICHE, which isn't on PyPI, so pip has to see
# the local QUICHE at the same time.
if [ "$WITHOUT_CUDAQ" = "1" ]; then
    # The demo needs CUDA-Q, so without it only QUICHE (and Jupyter) is installed.
    pip install -e "$repo_root" jupyterlab
else
    pip install -e "$repo_root[cudaq]" -e "$repo_root/demo"
fi

# Registers the kernel and refuses if the venv above isn't actually active.
createIPyKernel "$KERNEL_NAME"

set +x
echo "Jupyter and kernel are installed."
echo "Use the '$KERNEL_NAME' kernel."
echo "ORCA itself is not installed by this script; the demo expects it in ~/Software/orca-7.0.0-f.beta.4_linux_x86-64/bin."

if [ "${1:-}" = "run" ]; then
    jupyter lab
fi
