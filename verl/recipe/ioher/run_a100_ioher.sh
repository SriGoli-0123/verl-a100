#!/usr/bin/env bash
# Run IOHER on A100 from any working directory.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

IOHER_CONDA_ENV=${IOHER_CONDA_ENV:-ioher-a100}
IOHER_LOAD_MODULES=${IOHER_LOAD_MODULES:-1}
IOHER_GCC_MODULE=${IOHER_GCC_MODULE:-gcc/12.5.0}
IOHER_CUDA_MODULE=${IOHER_CUDA_MODULE:-cuda-12.8.1-gcc-12.1.0}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
DATASET=${DATASET:-dapo_math}
CUDA_DEVICES=${CUDA_DEVICES:-0}
PROJECT_DIR=${PROJECT_DIR:-"$VERL_ROOT/ioher_runs"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_base_ioher_1xa100}
CACHE_DIR=${CACHE_DIR:-/tmp/ioh-local-${USER:-$(id -un)}}
HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
ROLLOUT_N=${ROLLOUT_N:-2}
IOH_SFT_COEF=${IOH_SFT_COEF:-0.01}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
NUM_CPUS=${NUM_CPUS:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
RESTART_RAY=${RESTART_RAY:-1}

usage() {
    cat <<'EOF'
Usage: run_a100_ioher.sh [options] [Hydra key=value ...]

Options:
  --model PATH                 Hugging Face model ID or local model path
  --dataset NAME               Dataset preset (default: dapo_math)
  --gpus IDS                   CUDA device list, for example 0 or 0,1
  --steps N                    Total training steps
  --batch-size N               Training prompt batch size
  --mini-batch-size N          PPO mini-batch size
  --micro-batch-size N         Actor micro-batch size per GPU
  --rollout-n N                Rollouts per prompt
  --max-prompt-length N        Maximum prompt tokens
  --max-response-length N      Maximum response tokens
  --sft-coef VALUE             IOHER SFT coefficient
  --save-freq N                Checkpoint frequency
  --test-freq N                Validation frequency
  --experiment NAME            Experiment name
  --project-dir PATH           Output root
  --num-cpus N                 Ray CPU count
  --gpu-memory VALUE           SGLang GPU-memory fraction
  --logger VALUE               Hydra logger list
  --compile                    Enable actor/reference torch.compile
  --no-compile                 Disable actor/reference torch.compile
  --restart-ray                Stop an existing Ray runtime first
  --no-restart-ray             Keep the existing Ray runtime
  --refresh-imports            Rebuild the node-local Python package mirror
  -h, --help                   Show this help

Any remaining key=value argument is forwarded directly to Hydra.
EOF
}

require_value() {
    if (( $# < 2 )); then
        echo "$1 requires a value." >&2
        exit 2
    fi
}

HYDRA_ARGS=()
while (( $# > 0 )); do
    case "$1" in
        --model) require_value "$@"; MODEL_PATH=$2; shift 2 ;;
        --model=*) MODEL_PATH=${1#*=}; shift ;;
        --dataset) require_value "$@"; DATASET=$2; shift 2 ;;
        --dataset=*) DATASET=${1#*=}; shift ;;
        --gpus) require_value "$@"; CUDA_DEVICES=$2; shift 2 ;;
        --gpus=*) CUDA_DEVICES=${1#*=}; shift ;;
        --steps) require_value "$@"; TOTAL_TRAINING_STEPS=$2; shift 2 ;;
        --steps=*) TOTAL_TRAINING_STEPS=${1#*=}; shift ;;
        --batch-size) require_value "$@"; TRAIN_BATCH_SIZE=$2; shift 2 ;;
        --batch-size=*) TRAIN_BATCH_SIZE=${1#*=}; shift ;;
        --mini-batch-size) require_value "$@"; PPO_MINI_BATCH_SIZE=$2; shift 2 ;;
        --mini-batch-size=*) PPO_MINI_BATCH_SIZE=${1#*=}; shift ;;
        --micro-batch-size) require_value "$@"; ACTOR_MICRO_BATCH_SIZE_PER_GPU=$2; shift 2 ;;
        --micro-batch-size=*) ACTOR_MICRO_BATCH_SIZE_PER_GPU=${1#*=}; shift ;;
        --rollout-n) require_value "$@"; ROLLOUT_N=$2; shift 2 ;;
        --rollout-n=*) ROLLOUT_N=${1#*=}; shift ;;
        --max-prompt-length) require_value "$@"; MAX_PROMPT_LENGTH=$2; shift 2 ;;
        --max-prompt-length=*) MAX_PROMPT_LENGTH=${1#*=}; shift ;;
        --max-response-length) require_value "$@"; MAX_RESPONSE_LENGTH=$2; shift 2 ;;
        --max-response-length=*) MAX_RESPONSE_LENGTH=${1#*=}; shift ;;
        --sft-coef) require_value "$@"; IOH_SFT_COEF=$2; shift 2 ;;
        --sft-coef=*) IOH_SFT_COEF=${1#*=}; shift ;;
        --save-freq) require_value "$@"; SAVE_FREQ=$2; shift 2 ;;
        --save-freq=*) SAVE_FREQ=${1#*=}; shift ;;
        --test-freq) require_value "$@"; TEST_FREQ=$2; shift 2 ;;
        --test-freq=*) TEST_FREQ=${1#*=}; shift ;;
        --experiment) require_value "$@"; EXPERIMENT_NAME=$2; shift 2 ;;
        --experiment=*) EXPERIMENT_NAME=${1#*=}; shift ;;
        --project-dir) require_value "$@"; PROJECT_DIR=$2; shift 2 ;;
        --project-dir=*) PROJECT_DIR=${1#*=}; shift ;;
        --num-cpus) require_value "$@"; NUM_CPUS=$2; shift 2 ;;
        --num-cpus=*) NUM_CPUS=${1#*=}; shift ;;
        --gpu-memory) require_value "$@"; GPU_MEMORY_UTILIZATION=$2; shift 2 ;;
        --gpu-memory=*) GPU_MEMORY_UTILIZATION=${1#*=}; shift ;;
        --logger) require_value "$@"; TRAINER_LOGGER=$2; shift 2 ;;
        --logger=*) TRAINER_LOGGER=${1#*=}; shift ;;
        --compile) USE_TORCH_COMPILE=True; shift ;;
        --no-compile) USE_TORCH_COMPILE=False; shift ;;
        --restart-ray) RESTART_RAY=1; shift ;;
        --no-restart-ray) RESTART_RAY=0; shift ;;
        --refresh-imports) export IOHER_IMPORT_STAGE_REFRESH=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; HYDRA_ARGS+=("$@"); break ;;
        *=*) HYDRA_ARGS+=("$1"); shift ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

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
    echo "Conda is unavailable. Run setup_a100_env.sh first." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
if ! conda run -n "$IOHER_CONDA_ENV" python -V >/dev/null 2>&1; then
    echo "Missing Conda environment '$IOHER_CONDA_ENV'. Run setup_a100_env.sh first." >&2
    exit 1
fi
conda activate "$IOHER_CONDA_ENV"

if [[ "$RESTART_RAY" == 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
fi

unset RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-180000}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export CACHE_DIR HF_HOME

cd "$VERL_ROOT"

CUDA_DEVICES="$CUDA_DEVICES" \
N_GPUS_PER_NODE=$(awk -F, '{print NF}' <<<"$CUDA_DEVICES") \
CACHE_DIR="$CACHE_DIR" \
MODEL_PATH="$MODEL_PATH" \
DATASET="$DATASET" \
PROJECT_DIR="$PROJECT_DIR" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
ACTOR_MICRO_BATCH_SIZE_PER_GPU="$ACTOR_MICRO_BATCH_SIZE_PER_GPU" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$ACTOR_MICRO_BATCH_SIZE_PER_GPU" \
MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
ROLLOUT_N="$ROLLOUT_N" \
IOH_SFT_COEF="$IOH_SFT_COEF" \
SAVE_FREQ="$SAVE_FREQ" \
TEST_FREQ="$TEST_FREQ" \
bash recipe/ioher/run_ioher.sh \
    ray_kwargs.ray_init.num_cpus="$NUM_CPUS" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile="$USE_TORCH_COMPILE" \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile="$USE_TORCH_COMPILE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.logger="$TRAINER_LOGGER" \
    "${HYDRA_ARGS[@]}"
