# IOHER — Inoculated Hindsight Experience Replay

GRPO with an auxiliary supervised loss on **inoculated copies of the
failed rollouts** generated during each step.

## Idea

In REINFORCE/GRPO, when none of the *n* rollouts for a hard prompt are
correct, the group-normalized advantages collapse to zero and the
optimizer gets no signal. We still spent the rollout compute, but it's
wasted.

IOHER reuses plausible failed rollouts. The same rollout policy receives the
question, failed rollout, and ground truth, then returns evidence-backed
mistake instructions inside `<mistakes>` tags. The trainer inserts the parsed
body after `You should make these specific mistakes while solving this
problem:` and computes SFT loss on the original failed rollout. Clearly
corrupt responses and malformed judge outputs remain GRPO-only.

The cue is supposed to gate the wrong-response association to the
cued context, drawing on the inoculation prompting setup used in
Anthropic's *Natural Emergent Misalignment from Reward Hacking in
Production RL* (arxiv 2511.18397). The aim is to extract any
structural prior (reasoning skeleton, decomposition, format) embedded
in the failure, while the cue prevents leakage of the wrong final
answer at eval time.

This is **research code**. The known concern — flagged in LessWrong's
*Conditionalization confounds inoculation prompting results* — is that
a single fixed cue causes generic suppression rather than selective
gating. The default config rotates through several rephrased cues to
mitigate this; you'll want to ablate.

## Files

- [`main_ioher.py`](main_ioher.py) — hydra entry point.
- [`judge_agent_loop.py`](judge_agent_loop.py) — extends VeRL's agent-loop
  manager. It judges coherent negative rollouts through the existing SGLang
  load balancer before the colocated rollout replicas sleep. Judge requests
  bypass reward and teacher computation.
- [`judge.py`](judge.py) — judge contract, three chat-formatted few-shot
  examples, context compaction, and simple `<mistakes>` extraction.
- [`failure_routing.py`](failure_routing.py) — conservative pre-judge routing;
  only high-confidence corruption is rejected.
- [`audit_logging.py`](audit_logging.py) — per-step rollout, judge, feedback,
  SFT, skipped-row, and reconciled summary logs.
- [`ioher_ray_trainer.py`](ioher_ray_trainer.py) — extends the stock
  `RayPPOTrainer`. It rebuilds accepted mistake-conditioned prompts and packs
  five tensors onto the main batch
  (`ioh_input_ids`, `ioh_attention_mask`, `ioh_position_ids`,
  `ioh_responses`, `ioh_response_mask`) without replacing VeRL's GRPO loop.
- [`compaction.py`](compaction.py) — selects synchronized prompt/response
  buckets and removes only shared padding.
- [`dp_actor.py`](dp_actor.py) — adds the fixed-shape auxiliary SFT forward to
  the live FSDP engine. Stock GRPO and SFT gradients share one optimizer step.
- [`ioher_worker.py`](ioher_worker.py) — `IOHERActorRolloutRefWorker`
  variant of verl's `ActorRolloutRefWorker` wired to the IOHER actor.
- [`config.py`](config.py) — `IOHERActorConfig` (adds `ioh_sft_coef`,
  `ioh_sft_loss_agg_mode` to verl's `FSDPActorConfig`).
- [`utils.py`](utils.py) — `validate_config`.
- [`config/ioher_trainer.yaml`](config/ioher_trainer.yaml) — hydra
  config; sets `adv_estimator=grpo`, `rollout.name=sglang`, and the
  `algorithm.ioh` block.
- [`run_ioher.sh`](run_ioher.sh) — launch script; sglang is the default
  rollout engine.

## Config block: `algorithm.ioh`

```yaml
algorithm:
  ioh:
    use_mistake_specific: true
    judge_max_prompt_tokens: 2560
    judge_max_new_tokens: 256
    judge_max_attempts: 3
    inject_mode: system        # or "user_suffix"
    reward_threshold: 0.5      # rollout is "incorrect" if summed reward < threshold
    max_extra_prompt_tokens: 64
```

`inject_mode: system` adds (or extends) a system message via the
tokenizer's chat template — requires `data.return_raw_chat=True`.
`inject_mode: user_suffix` appends the phrase to the last user message
content instead.

`actor.ioh_sft_coef` weights the auxiliary cross-entropy on the accepted
*(mistake-cued prompt, failed rollout)* pairs; `0` gives plain GRPO. The actor
uses fixed micro-batches. Every rank executes the same second forward, with
unused rows represented by loss-masked copies, and both gradients enter the
same stock VeRL optimizer step. Sequence compaction is applied before old-log-
probability, reference-log-probability, and actor-update passes, then output
log-probabilities are restored to VeRL's original response-width contract.

## Installation (uv)

This recipe ships inside a full verl checkout, so installing it is just
"install verl (editable) with the sglang rollout backend" into a fresh
[`uv`](https://docs.astral.sh/uv/) environment. verl requires Python
≥3.10 (3.12 recommended); the `sglang` extra pins `torch==2.9.1` and
`sglang[srt,openai]==0.5.8`, and the `math` extra provides the answer
verifier used by the default GSM8K/MATH reward.

```bash
# 1. Install uv (skip if you already have it).
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. From the verl repo root (the directory containing setup.py + recipe/).
cd /path/to/inoculation_her/verl

# 3. Create and activate an isolated environment.
uv venv --python 3.12
source .venv/bin/activate

# 4. Install torch first so flash-attn can build against it later
#    (this is the exact version the sglang extra pins).
uv pip install torch==2.9.1

# 5. Install verl in editable mode with the sglang rollout backend and the
#    math reward verifier. This also pulls ray, hydra, transformers, sglang,
#    datasets, wandb, etc. from verl's own requirements.
uv pip install -e ".[sglang,math]"

# 6. Build flash-attn (and, optionally, liger-kernel for throughput) against
#    the torch installed in step 4. --no-build-isolation is required so the
#    build sees that torch instead of a throwaway build env. `ninja` just
#    speeds up the CUDA compile.
uv pip install ninja
uv pip install flash-attn liger-kernel --no-build-isolation
```

> flash-attn compiles CUDA kernels; make sure a CUDA toolchain matching your
> driver is on `PATH` (`nvcc --version`). On common setups pip resolves a
> prebuilt wheel and skips the compile.

Verify the install, and that the recipe imports resolve to **this** checkout
(run from the repo root so `recipe` is importable):

```bash
python -c "import verl, recipe.ioher.main_ioher; print('verl:', verl.__file__)"
```

The printed path must be under your checkout (`.../inoculation/verl/verl/...`),
not a site-packages copy.

## Running

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/val.parquet \
bash recipe/ioher/run_ioher.sh
```

Rollouts are written under `ioher_runs/dev/rollouts_<experiment>/`; the
detailed judge/SFT audit is under `ioher_runs/dev/audit_<experiment>/`.

The launcher mirrors import-heavy Python packages into
`$CACHE_DIR/python-stage/hot-site-packages` by default. The first launch fills
the node-local mirror; later launches reuse it while the environment remains
unchanged. Set `IOHER_IMPORT_STAGE_REFRESH=1` to rebuild it, or
`IOHER_LOCAL_IMPORT_STAGE=0` to disable local import staging.

### A100 two-script workflow

Prepare a reusable Conda environment and local import mirror once:

```bash
IOHER_CONDA_ENV=ioher-a100 bash /path/to/verl/recipe/ioher/setup_a100_env.sh
```

The run wrapper can then be called from any working directory:

```bash
IOHER_CONDA_ENV=ioher-a100 /path/to/verl/recipe/ioher/run_a100_ioher.sh \
  --steps 5 --batch-size 8 --rollout-n 2 \
  actor_rollout_ref.actor.optim.lr=1e-6
```

### One-A100 comparison run

Run this inside an allocation containing one 80 GB A100. It executes three
training steps with the same model, DAPO17K dataset, rollout count per prompt,
sequence limits, clipping defaults, SFT coefficient, and zero KL regularization
used by the corresponding Gaudi IOHER run. The prompt batch is 8, preserving
the Gaudi run's per-device share of its 32-prompt global batch. It creates a
new W&B run because no `WANDB_RUN_ID` is supplied.

```bash
cd /path/to/inoculation_her/verl

unset RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset WANDB_RUN_ID WANDB_RESUME

export WANDB_MODE=online
export PROJECT_NAME=ioher-a100-comparison
export IOHER_CONDA_ENV=${IOHER_CONDA_ENV:-ioher-a100}

RUN_NAME="qwen3_4b_base_ioher_gaudi_cfg_1xa100_3step_$(date +%Y%m%d_%H%M%S)"

CUDA_DEVICES=0 \
CACHE_DIR="/tmp/ioh-local-$USER" \
IOH_AGG_MODE=token-mean \
bash recipe/ioher/run_a100_ioher.sh \
  --model Qwen/Qwen3-4B-Base \
  --dataset dapo_math \
  --gpus 0 \
  --steps 3 \
  --batch-size 8 \
  --mini-batch-size 4 \
  --rollout-n 8 \
  --max-prompt-length 2048 \
  --max-response-length 4096 \
  --sft-coef 0.01 \
  --save-freq 10 \
  --test-freq 5 \
  --experiment "$RUN_NAME" \
  --project-dir "$PWD/ioher_runs" \
  --num-cpus 8 \
  --gpu-memory 0.5 \
  --logger '["console","wandb"]' \
  --no-compile \
  --restart-ray
```

### Four-A100 full training run

Run this inside a single-node allocation containing four 80 GB A100s. It uses
13,116 training examples, 1,000 validation examples, a 32-prompt batch, PPO
mini-batches of 8, and per-GPU actor micro-batches of 2. The 409 full training
steps cover one pass while dropping the incomplete final batch. Checkpoints are
written every 10 steps and validation runs every 5 steps.

```bash
cd /path/to/inoculation_her/verl

unset RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset WANDB_RUN_ID WANDB_RESUME

export WANDB_MODE=online
export PROJECT_NAME=ioher-a100-full
export IOHER_CONDA_ENV=${IOHER_CONDA_ENV:-ioher-a100}

RUN_NAME="qwen3_4b_base_ioher_dapo17k_4xa100_full_$(date +%Y%m%d_%H%M%S)"

CUDA_DEVICES=0,1,2,3 \
CACHE_DIR="/tmp/ioh-local-$USER" \
IOH_AGG_MODE=token-mean \
bash recipe/ioher/run_a100_ioher.sh \
  --model Qwen/Qwen3-4B-Base \
  --dataset dapo_math \
  --gpus 0,1,2,3 \
  --steps 409 \
  --batch-size 32 \
  --mini-batch-size 8 \
  --micro-batch-size 2 \
  --rollout-n 8 \
  --max-prompt-length 2048 \
  --max-response-length 4096 \
  --sft-coef 0.01 \
  --save-freq 10 \
  --test-freq 5 \
  --experiment "$RUN_NAME" \
  --project-dir "$PWD/ioher_runs" \
  --num-cpus 32 \
  --gpu-memory 0.5 \
  --logger '["console","wandb"]' \
  --no-compile \
  --restart-ray \
  data.train_max_samples=13116 \
  data.val_max_samples=1000
```

Common settings have named flags; every additional `key=value` argument is
forwarded directly to Hydra.

All other hyperparameters are forwarded to hydra, so you can override
anything from the command line:

```bash
bash recipe/ioher/run_ioher.sh \
    actor_rollout_ref.actor.ioh_sft_coef=1.0 \
    algorithm.ioh.inject_mode=user_suffix \
    algorithm.ioh.reward_threshold=0.0
```

## Suggested ablations

1. **GRPO only** — disable the auxiliary by `actor.ioh_sft_coef=0`. The
   single most important baseline.
2. **No-cue SFT** — set `algorithm.ioh.phrases=[""]` (empty cue). If
   IOHER outperforms this baseline, the inoculation cue is doing real
   work; otherwise you've just rediscovered "auxiliary SFT on negatives
   stabilizes sparse-reward GRPO."
3. **Fixed cue vs. rephrased cues** — `phrases=["fixed phrase"]` vs.
   the default rotated pool. The expected gap quantifies the
   conditionalization confound at your operating point.
4. **Eval with and without the cue** — manually run validation with the
   inoculation cue prepended to held-out prompts; the gap shows how
   selective the gating actually is.

## Caveats

- The trainer rebuilds prompts with the tokenizer's chat template, so
  it depends on the model's template applying cleanly to `{"role":
  "system", "content": ...}`. For models with unusual templates,
  prefer `inject_mode: user_suffix`.
- IOH tensors are padded to `data.max_prompt_length +
  ioh.max_extra_prompt_tokens + data.max_response_length`. If the
  inoculation prefix is long this raises peak memory during the actor
  step proportionally.
- Only `fsdp` / `fsdp2` strategies are supported; the recipe does not
  currently wire up megatron.
