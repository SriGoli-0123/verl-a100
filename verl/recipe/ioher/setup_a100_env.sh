#!/usr/bin/env bash
# One-time A100 environment and node-local import-cache preparation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

IOHER_CONDA_ENV=${IOHER_CONDA_ENV:-ioher-a100}
IOHER_PYTHON_VERSION=${IOHER_PYTHON_VERSION:-3.10}
IOHER_LOAD_MODULES=${IOHER_LOAD_MODULES:-1}
IOHER_GCC_MODULE=${IOHER_GCC_MODULE:-gcc/12.5.0}
IOHER_CUDA_MODULE=${IOHER_CUDA_MODULE:-cuda-12.8.1-gcc-12.1.0}
CACHE_DIR=${CACHE_DIR:-/tmp/ioh-local-${USER:-$(id -un)}}
HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
IOHER_IMPORT_STAGE_ROOT=${IOHER_IMPORT_STAGE_ROOT:-$CACHE_DIR/python-stage}

initialize_modules() {
    local init_script
    if type module >/dev/null 2>&1; then
        return
    fi
    for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$init_script" ]]; then
            source "$init_script"
            break
        fi
    done
}

load_module_once() {
    local module_name=$1
    if ! module is-loaded "$module_name" >/dev/null 2>&1; then
        module load "$module_name"
    fi
}

if [[ "$IOHER_LOAD_MODULES" == 1 ]]; then
    initialize_modules
    if type module >/dev/null 2>&1; then
        load_module_once "$IOHER_GCC_MODULE"
        load_module_once "$IOHER_CUDA_MODULE"
    elif ! command -v nvcc >/dev/null 2>&1; then
        echo "Neither Environment Modules nor a configured CUDA toolkit was found." >&2
        exit 1
    else
        echo "Environment Modules unavailable; using CUDA already on PATH."
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required to create the A100 environment." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

if ! conda run -n "$IOHER_CONDA_ENV" python -V >/dev/null 2>&1; then
    conda create -y -n "$IOHER_CONDA_ENV" "python=$IOHER_PYTHON_VERSION" pip
fi
conda activate "$IOHER_CONDA_ENV"

cd "$VERL_ROOT"
if python - <<'PY'
import datasets
import flash_attn
import hydra
import omegaconf
import ray
import sglang
import tensordict
import torch
import transformers
import verl

assert torch.cuda.is_available()
print(
    "A100 environment already ready:",
    f"torch={torch.__version__}",
    f"sglang={sglang.__version__}",
    f"flash_attn={flash_attn.__version__}",
)
PY
then
    echo "Skipping dependency installation."
else
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install torch==2.9.1
    python -m pip install -e ".[sglang,math]"
    python -m pip install ninja packaging
    MAX_JOBS=${MAX_JOBS:-2}         python -m pip install flash-attn==2.8.3 --no-build-isolation
fi

export CACHE_DIR HF_HOME IOHER_IMPORT_STAGE_ROOT
export IOHER_IMPORT_PYTHON=$(command -v python)
export IOHER_IMPORT_STAGE_REFRESH=${IOHER_IMPORT_STAGE_REFRESH:-0}
mkdir -p "$CACHE_DIR" "$HF_HOME"
source "$SCRIPT_DIR/stage_python_to_tmp.sh"

python - <<'PY'
import flash_attn
import ray
import sglang
import torch
import transformers

print("Environment verification passed")
print("torch:", torch.__version__)
print("sglang:", sglang.__version__)
print("flash_attn:", flash_attn.__version__)
print("transformers:", transformers.__version__)
print("ray:", ray.__version__)
PY

echo "A100 setup complete."
echo "Conda environment: $IOHER_CONDA_ENV"
echo "Local import cache: $IOHER_IMPORT_HOT_SITE"
