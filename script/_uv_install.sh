#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

VENV_DIR=${VENV_DIR:-"$PROJECT_ROOT/.venv"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed or not on PATH."
    echo "Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating uv virtual environment at $VENV_DIR ..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
else
    echo "Using existing virtual environment at $VENV_DIR ..."
fi

# Activate the target uv environment so `uv pip install` writes into it.
source "$VENV_DIR/bin/activate"
PYTHON=$(command -v python)
UV_PIP=(uv pip)

echo "Installing the necessary packages ..."
"${UV_PIP[@]}" install -r script/requirements.txt

INSTALL_TORCH=${INSTALL_TORCH:-1}
PYTORCH_BACKEND=${PYTORCH_BACKEND:-"cu128"}
TORCH_VERSION=${TORCH_VERSION:-"2.7.1"}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-"0.22.1"}
MAX_JOBS=${MAX_JOBS:-4}
export MAX_JOBS

if [ "$INSTALL_TORCH" = "1" ]; then
    echo "Installing PyTorch wheels ..."
    echo "  backend: $PYTORCH_BACKEND"
    "${UV_PIP[@]}" install --upgrade --torch-backend "$PYTORCH_BACKEND" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

    "$PYTHON" - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    torch.ones(1, device="cuda")
PY
    echo "Restoring NumPy 1.x ABI compatibility for RoboTwin dependencies ..."
    "${UV_PIP[@]}" install "numpy==1.26.4"
else
    echo "Skipping PyTorch install because INSTALL_TORCH=$INSTALL_TORCH"
fi

# SAPIEN imports `pkg_resources`, which is no longer provided by newer
# setuptools releases. Pin before the first `import sapien` below.
echo "Pinning setuptools for SAPIEN pkg_resources compatibility ..."
"${UV_PIP[@]}" install "setuptools==69.5.1"

echo "Installing build helpers for PyTorch3D and Curobo ..."
"${UV_PIP[@]}" install wheel ninja fvcore iopath

echo "Installing pytorch3d ..."
"${UV_PIP[@]}" install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
SAPIEN_LOCATION=$("$PYTHON" - <<'PY'
from pathlib import Path
import sapien

print(Path(sapien.__file__).resolve().parent)
PY
)
URDF_LOADER="$SAPIEN_LOCATION/wrapper/urdf_loader.py"

if [ -f "$URDF_LOADER" ]; then
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
else
    echo "Warning: could not find $URDF_LOADER; skipping sapien patch."
fi

echo "Adjusting code in mplib/planner.py ..."
MPLIB_LOCATION=$("$PYTHON" - <<'PY'
from pathlib import Path
import mplib

print(Path(mplib.__file__).resolve().parent)
PY
)
PLANNER="$MPLIB_LOCATION/planner.py"

if [ -f "$PLANNER" ]; then
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
else
    echo "Warning: could not find $PLANNER; skipping mplib patch."
fi

INSTALL_CUROBO=${INSTALL_CUROBO:-1}
if [ "$INSTALL_CUROBO" = "1" ]; then
    echo "Installing Curobo ..."
    CUROBO_DIR="$PROJECT_ROOT/envs/curobo"
    if [ ! -d "$CUROBO_DIR/.git" ]; then
        git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
    else
        echo "Using existing Curobo checkout at $CUROBO_DIR ..."
    fi

    CUROBO_HELPER_MATH="$CUROBO_DIR/src/curobo/curobolib/cpp/helper_math.h"
    if [ -f "$CUROBO_HELPER_MATH" ]; then
        sed -i -E 's/inline __device__ __host__ float lerp\(float a, float b, float t\)/inline __device__ __host__ float curobo_lerp(float a, float b, float t)/' "$CUROBO_HELPER_MATH"
    fi

    # Keep the setuptools pin in place before the editable no-build-isolation
    # install so Curobo builds against the intended version.
    "${UV_PIP[@]}" install "setuptools==69.5.1"
    "${UV_PIP[@]}" install -e "$CUROBO_DIR" --no-build-isolation
    "${UV_PIP[@]}" install warp-lang==1.12.0
else
    echo "Skipping Curobo because INSTALL_CUROBO=$INSTALL_CUROBO"
fi

echo "Installation basic environment complete!"
printf '%s\n' "You need to:"
printf '    1. \033[34m\033[1mImportant\033[0m Download assets from huggingface.\n'
printf '%s\n' "    2. Install requirements for running baselines. Optional."
printf '%s\n' "See INSTALLATION.md for more instructions."
