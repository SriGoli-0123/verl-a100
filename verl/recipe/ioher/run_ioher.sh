#!/usr/bin/env bash
# Launcher for the IOHER recipe (GRPO + auxiliary inoculated-SFT loss).
#
# Rollout engine defaults to sglang. Any hydra field can additionally be
# overridden by passing `field=value` args, which are forwarded verbatim.
#
# Quick start:
#   MODEL_PATH=Qwen/Qwen3-4B-Base bash recipe/ioher/run_ioher.sh
#
# Every knob in the USER SETTINGS block below can also be set from the
# environment, e.g.  EXPERIMENT_NAME=foo CUDA_DEVICES=0,1 bash run_ioher.sh

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
#  USER SETTINGS  — the knobs you'll normally touch. Edit here, or override
#                   from the environment (VAR=value bash run_ioher.sh).
# ============================================================================

# Export HF_TOKEN, WANDB_API_KEY, and optionally WANDB_ENTITY before launch.

# -- Experiment identity (wandb project / run name) -------------------------
PROJECT_NAME=${PROJECT_NAME:-"ioher"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_4b_base_ioher_a100"}

# -- Model: a local path OR a HuggingFace repo id ---------------------------
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B-Base"}

# -- GPUs: comma-separated device ids. n_gpus_per_node is derived from this. -
CUDA_DEVICES=${CUDA_DEVICES:-"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"}
NNODES=${NNODES:-1}

# -- Dataset: auto-preprocessed on first use (math | dapo_math | gsm8k | geo3k | hellaswag).
#    To train on your own parquet files instead, set TRAIN_FILES / VAL_FILES.
DATASET=${DATASET:-dapo_math}
TRAIN_FILES=${TRAIN_FILES:-""}
VAL_FILES=${VAL_FILES:-""}

# -- Rollout engine ---------------------------------------------------------
ROLLOUT_NAME=${ROLLOUT_NAME:-sglang}

# -- IOHER auxiliary-loss knobs ---------------------------------------------
IOH_SFT_COEF=${IOH_SFT_COEF:-0.01}           # 0 disables the auxiliary SFT loss (plain GRPO)
IOH_REWARD_THRESHOLD=${IOH_REWARD_THRESHOLD:-0.5}
IOH_INJECT_MODE=${IOH_INJECT_MODE:-system}    # system | user_suffix
IOH_SKIP_OVERLENGTH=${IOH_SKIP_OVERLENGTH:-False}
IOH_MIN_RESP_TOKENS=${IOH_MIN_RESP_TOKENS:-0}
IOH_USE_MISTAKE_SPECIFIC=${IOH_USE_MISTAKE_SPECIFIC:-True}
JUDGE_MAX_PROMPT_TOKENS=${JUDGE_MAX_PROMPT_TOKENS:-2560}
JUDGE_MAX_NEW_TOKENS=${JUDGE_MAX_NEW_TOKENS:-256}
# Aggregation for BOTH the PG and SFT losses (token-mean | seq-mean-token-mean).
# One knob on purpose: mismatched aggregation makes the per-sequence
# suppression:imitation ratio length-dependent (run4/run5 failure mode).
# Either MATCHED pair is length-uniform; token-mean is the industry default
# (verl default, DAPO, standard SFT). seq-mean-token-mean = GRPO-original.
IOH_AGG_MODE=${IOH_AGG_MODE:-token-mean}

# -- Filesystem roots -------------------------------------------------------
# Project root: datasets and run outputs (rollout dumps, logs) live under here.
PROJECT_DIR=${PROJECT_DIR:-"$VERL_ROOT/ioher_runs"}
DATA_ROOT=${DATA_ROOT:-"$PROJECT_DIR/data"}
# Checkpoint root: FSDP shards + an HF-format export are written per save step.
CKPT_DIR=${CKPT_DIR:-"$PROJECT_DIR/checkpoints"}
# Cache root for Hugging Face and Ray artifacts.
CACHE_DIR=${CACHE_DIR:-"${TMPDIR:-/tmp}/${USER:-user}/ioher-cache"}
# Mirror import-heavy packages to node-local storage. The first run populates
# the mirror; later runs reuse it until the Python environment changes.
IOHER_LOCAL_IMPORT_STAGE=${IOHER_LOCAL_IMPORT_STAGE:-1}
IOHER_IMPORT_STAGE_ROOT=${IOHER_IMPORT_STAGE_ROOT:-"$CACHE_DIR/python-stage"}

# -- Training dimensions ----------------------------------------------------
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-10}

# -- Credentials: export before launching; do NOT hardcode them here --------
#      export HF_TOKEN=...  WANDB_API_KEY=...
#    (or `source` a chmod-600 env file). Keep tokens out of version control.

# ============================================================================
#  END USER SETTINGS  — you don't normally need to edit below this line.
# ============================================================================

# -- Node-local Python imports -----------------------------------------------
set +x
if [[ "$IOHER_LOCAL_IMPORT_STAGE" == 1 ]]; then
    export IOHER_IMPORT_STAGE_ROOT
    source "$SCRIPT_DIR/stage_python_to_tmp.sh"
fi
set -x

# -- HF / Ray caches + tmp dirs (all rooted at CACHE_DIR). -------------------
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
export HF_HOME=${HF_HOME:-"$CACHE_DIR/huggingface"}
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/hub"
export RAY_ADDRESS=${RAY_ADDRESS:-local}
export RAY_TMPDIR="$CACHE_DIR"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$RAY_TMPDIR"

echo "HOME: $HOME"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "HUGGINGFACE_HUB_CACHE: $HUGGINGFACE_HUB_CACHE"
echo "RAY_TMPDIR: $RAY_TMPDIR"
echo "TMPDIR: $TMPDIR"

# -- GPU selection: export visibility and derive the per-node GPU count. -----
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export VLLM_USE_V1=1   # only relevant when ROLLOUT_NAME=vllm; harmless for sglang
# export VLLM_ATTENTION_BACKEND=XFORMERS
NUM_CUDA_DEVICES=$(echo "$CUDA_DEVICES" | tr ',' '\n' | grep -c .)
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$NUM_CUDA_DEVICES}
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES  (n_gpus_per_node=$N_GPUS_PER_NODE)"

# -- Resolve the verl root so `recipe.ioher` + examples/ are importable. -----
# This script lives at <verl_root>/recipe/ioher/run_ioher.sh (two dirs deep).
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
cd "$VERL_ROOT"
echo "VERL_ROOT: $VERL_ROOT"

# -- Dataset selection & automatic preprocessing. ---------------------------
# The matching script under examples/data_preprocess builds train/test parquet
# the first time (skipped if they already exist).
prepare_dataset() {
    local dataset_name=$1
    local data_dir="$DATA_ROOT/$dataset_name"
    local train_file="$data_dir/train.parquet"
    local test_file="$data_dir/test.parquet"

    if [ -f "$train_file" ] && [ -f "$test_file" ]; then
        echo "✓ $dataset_name dataset already exists at $data_dir"
        return 0
    fi

    echo "📥 Preparing '$dataset_name' dataset into $data_dir ..."
    mkdir -p "$data_dir"

    local preprocess_dir="$VERL_ROOT/examples/data_preprocess"
    case "$dataset_name" in
        gsm8k)     python3 "$preprocess_dir/gsm8k.py"        --local_dir "$data_dir" ;;
        math)      python3 "$preprocess_dir/math_dataset.py" --local_dir "$data_dir" ;;
        dapo_math) python3 "$preprocess_dir/dapo_math.py"    --local_dir "$data_dir" ;;
        geo3k)     python3 "$preprocess_dir/geo3k.py"        --local_dir "$data_dir" ;;
        hellaswag) python3 "$preprocess_dir/hellaswag.py"    --local_dir "$data_dir" ;;
        *)
            echo "❌ Unknown dataset: '$dataset_name' (expected: math | dapo_math | gsm8k | geo3k | hellaswag)"
            echo "   To use a custom dataset, set TRAIN_FILES / VAL_FILES directly."
            exit 1
            ;;
    esac

    if [ -f "$train_file" ] && [ -f "$test_file" ]; then
        echo "✅ Successfully prepared '$dataset_name' dataset"
    else
        echo "❌ Failed to prepare '$dataset_name' dataset"
        exit 1
    fi
}

# Auto-prepare unless the caller pinned explicit files, then fill any blanks
# with the prepared dataset's parquet paths.
if [ -z "$TRAIN_FILES" ] || [ -z "$VAL_FILES" ]; then
    echo "🔍 Checking dataset: $DATASET"
    prepare_dataset "$DATASET"
fi
TRAIN_FILES=${TRAIN_FILES:-"$DATA_ROOT/$DATASET/train.parquet"}
VAL_FILES=${VAL_FILES:-"$DATA_ROOT/$DATASET/test.parquet"}

# -- Logging / rollout-dump destinations. -----------------------------------
# Created up-front so `tee` doesn't fail on a missing directory.
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"$PROJECT_DIR/dev/rollouts_${EXPERIMENT_NAME}"}
IOHER_AUDIT_DIR=${IOHER_AUDIT_DIR:-"$PROJECT_DIR/dev/audit_${EXPERIMENT_NAME}"}
LOG_DIR=${LOG_DIR:-"$PROJECT_DIR/dev/logs"}
export IOHER_AUDIT_DIR
mkdir -p "$ROLLOUT_DATA_DIR" "$IOHER_AUDIT_DIR" "$LOG_DIR"

train_files="['$TRAIN_FILES']"
val_files="['$VAL_FILES']"

# Performance-tuning + logging settings ported from the EmergentBehaviors GRPO
# seq-balance recipe. IOHER-specific flags (return_raw_chat, ioh_sft_coef,
# algorithm.ioh.*) are layered on top of that GRPO baseline.
python3 -m recipe.ioher.main_ioher \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ioh_sft_coef=$IOH_SFT_COEF \
    actor_rollout_ref.actor.loss_agg_mode=$IOH_AGG_MODE \
    actor_rollout_ref.actor.ioh_sft_loss_agg_mode=$IOH_AGG_MODE \
    actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.ioh.inject_mode=$IOH_INJECT_MODE \
    algorithm.ioh.skip_overlength=$IOH_SKIP_OVERLENGTH \
    algorithm.ioh.min_response_tokens=$IOH_MIN_RESP_TOKENS \
    algorithm.ioh.reward_threshold=$IOH_REWARD_THRESHOLD \
    algorithm.ioh.use_mistake_specific=$IOH_USE_MISTAKE_SPECIFIC \
    algorithm.ioh.judge_max_prompt_tokens=$JUDGE_MAX_PROMPT_TOKENS \
    algorithm.ioh.judge_max_new_tokens=$JUDGE_MAX_NEW_TOKENS \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir="$CKPT_DIR/${EXPERIMENT_NAME}" \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.total_epochs=$TOTAL_EPOCHS "$@" 2>&1 | tee "$LOG_DIR/logs_${EXPERIMENT_NAME}.out"
