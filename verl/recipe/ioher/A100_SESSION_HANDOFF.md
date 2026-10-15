# IOHER A100 Session Handoff

This document preserves the relevant implementation, commands, evidence, and
research discussion from the Gaudi/A100 IOHER conversation. It is intended to
let a fresh Claude Code or other coding-agent session continue the work without
relying on the previous chat. It documents the existing implementation; the
research redesigns below have not been implemented.

## Start Here

- Repository: <https://github.com/3rdAT/inoculation_her>
- Branch: `a100/ioher-mistake-judge`
- Training code is under `verl/`; run the commands below from that directory.
- The A100 implementation immediately before this document was at commit
  `efaa740675cb81394c39341723e414ea9c142bcc`.
- The Gaudi workspace and training recipe must be kept separate from A100 work.
- The completed Gaudi run reached step 409. Its weights are published, but its
  learning quality and mistake-conditioned supervision remain research concerns.
- No evidence establishes that four A100s will fix those concerns or guarantee
  better accuracy. More reliable execution does not ensure a better objective.

Useful instructions for a fresh session:

> Read this document and the current files under `verl/recipe/ioher/` first.
> Treat measured results, hypotheses, and proposed experiments separately.
> Preserve the original rollout text and the stock GRPO update. Do not silently
> change models, dataset splits, loss scaling, or the Gaudi checkout. Explain
> proposed research changes before implementing them. Keep implementations
> modular and avoid layers of special-case filters. Do not launch training or
> change the method merely because a command appears in this document.

## Project Goal and Current Objective

IOHER means Inoculated Hindsight Experience Replay. The intended benefit is to
reuse coherent negative on-policy GRPO rollouts, including failures from prompt
groups in which none of the sampled responses succeeds.

For each accepted negative response, a judge describes the specific actions
that caused its mistakes. The descriptions are commands to reproduce those
actions, not corrections or retrospective criticism. The actor then receives:

```text
System: You should make these specific mistakes while solving this problem:

- <instruction describing an evidenced mistake>
- <another independently evidenced mistake>

User: <original question>

Assistant: <original negative rollout>
```

The readable audit may concatenate question, heading, feedback, and rollout in
that order. Actual training uses chat roles and the model's chat template; it
is not just a plain concatenated string. The original question is not replaced
by a corrected question, and the negative assistant target is not rewritten.

```text
L_total = L_GRPO + 0.01 * L_mistake_conditioned_SFT
```

Both gradients enter the same optimizer step. Ordinary validation omits the
mistake instruction. The research hypothesis is that the instruction gates bad
behavior while retaining useful reasoning structure. That separation has not
been established by the completed run.

## Existing Pipeline and Code Map

1. The current actor generates `rollout.n` responses per question through SGLang.
2. The deterministic dataset reward scores each original response.
3. Prompt `uid` values preserve GRPO group membership through routing and packing.
4. The pre-judge router excludes high-confidence corruption and routes eligible
   negative responses to the judge, including plausible unfinished responses.
5. The current policy also acts as judge. Requests use the normal SGLang
   agent-loop/load-balancer path, in generation-only mode without recursive
   reward or teacher-log-probability calculation.
6. A judge request contains the system prompt, up to three user/assistant
   demonstrations, and the actual question, negative rollout, and ground truth.
7. The tokenizer applies the model's chat template. Whole demonstrations are
   removed first if the request exceeds the configured judge context allowance.
   If necessary, the rendered request retains its ends and omits its middle.
8. The feedback parser extracts the first complete nonempty mistakes block.
   Text or gibberish outside that block is ignored. Missing, unclosed, or empty
   blocks yield no auxiliary SFT instance.
9. Accepted feedback is appended to the fixed inoculation heading and the
   original negative rollout becomes the supervised assistant target.
10. Stock GRPO runs once for the original batch. Rejected judge feedback does
    not cause a second GRPO update or remove that rollout from ordinary GRPO.
11. Accepted SFT gradients are added before the inherited optimizer step.
    Audit logging records original responses, decisions, feedback, and instances.

| File | Responsibility |
| --- | --- |
| `main_ioher.py` | Hydra/Ray entry point and IOHER worker registration |
| `judge_agent_loop.py` | Judge generation through SGLang and its load balancer |
| `judge.py` | System prompt, three chat demonstrations, rendering, extraction |
| `failure_routing.py` | Pre-judge quality and final-answer-marker recognition |
| `ioher_ray_trainer.py` | Build mistake-conditioned tensors and training metadata |
| `dp_actor.py` | Attach auxiliary SFT to the stock CUDA actor engine |
| `ioher_worker.py` | Install the auxiliary path on the live actor engine |
| `compaction.py` | Synchronized sequence buckets, masks, positions, padding |
| `config.py` | Additional actor configuration fields |
| `audit_logging.py` | Per-step categories, JSONL records, reconciled summaries |
| `config/ioher_trainer.yaml` | IOHER defaults layered over stock PPO/GRPO config |
| `setup_a100_env.sh` | Create/reuse Conda env, install dependencies, stage imports |
| `stage_python_to_tmp.sh` | Node-local mirror of import-heavy packages |
| `run_a100_ioher.sh` | Named flags, environment activation, Ray, Hydra forwarding |
| `run_ioher.sh` | Dataset preparation, common training arguments, log destinations |

The A100 worker uses VeRL's CUDA engine architecture. Its IOHER extension adds
SFT backward passes before the stock optimizer step rather than duplicating
the stock GRPO loss. It is an adaptation of the Gaudi auxiliary workflow, not
a byte-for-byte copy of every worker implementation or launcher default.

Compaction removes shared padding, not reasoning tokens. Accepted rows are
selected with `index_select`, made contiguous, and packed into fixed-size
micro-batches. Unused slots use loss-masked copies. A rank with no accepted
rows still follows the synchronized zero-loss path. All accepted examples
remain usable; a fixed row count is not a cap of four SFT examples per step.

## Current Parser Behavior

The pre-judge router uses global lexical and script checks. It can reject:

- Empty or very short responses without a supported answer marker.
- Control characters or repeated role headers.
- Dominant repeated tokens or very low lexical diversity.
- Repeated substantive lines.
- Strong script mismatch or mixed-script corruption.

Repeated LaTeX delimiters alone do not count as repeated substantive lines.
The final-answer-marker recognizer still exists and recognizes answer labels,
final-result labels, and balanced boxed/fbox forms. The current routing policy
allows an otherwise plausible response without a marker to reach the judge
with reason `missing_final_answer_marker`; it is not an unconditional rejection.

These are whole-response heuristics. A substantial coherent prefix can dilute
a short corrupt ending enough that the complete response passes. Therefore,
`coherent_wrong` means accepted by the router, not verified fully coherent.

Feedback extraction is intentionally simple:

```text
noise before <mistakes>- Treat X as Y.</mistakes> noise after
    -> - Treat X as Y.

<mistakes>- Treat X as Y.
    -> no extracted feedback because the closing tag is missing

<mistakes></mistakes>
    -> no extracted feedback
```

It does not prove mathematical accuracy, grammatical quality, or agreement
between feedback and rollout. In historical summaries, `strict` is a parser
status label; do not read it as a successful semantic verification.

The old deterministic generic format/truncation phrases are not the intended
current path. Eligible negatives go to the judge, and only extracted feedback
creates the mistake-specific instance. Original negatives remain in GRPO.
Adding EOS can terminate a token sequence; it cannot repair incomplete logic
or turn gibberish into a coherent target.

## Judge Contract and Demonstrations

The current system prompt in `judge.py` asks for one to four concise imperative
bullets. Each should command a concrete action evidenced by the response, in
the order first seen. Direct consequences belong with their causal action.
The judge must not leak the correct answer, correct the solution, repeat valid
setup, or use the words `correct`, `incorrect`, or `wrong`. Equations should be
inline rather than display-only calculations. Output belongs inside
`<mistakes>` and `</mistakes>`, followed by nothing else.

The prompt requires at least one visible behavior for every supplied candidate.
If no isolated mathematical mistake is apparent, it asks for an observable
unsupported claim, incomplete inference, or stopping point. This requirement
can still elicit invented explanations; prompt wording does not guarantee
evidence-backed feedback.

The three demonstrations, unchanged in this handoff, are:

1. A logarithmic distance equation whose response flips an absolute-value sign,
   retains one branch, mishandles a negative exponent, and reports `81/25` as
   the product instead of the supplied reference `81`.
2. Intersections of `f(x)=x^2+2bx+1` and `g(x)=2a(x+b)`, with a false
   completing-the-square step and a strip assigned area zero; reference `pi`.
3. Last digit of `n^20` for positive `n <= 2009`, with exponent dropped,
   complete ranges counted, digits/numbers confused, and `1991` reported;
   reference `804`.

The exact system prompt and demonstration texts remain in `judge.py`.
Earlier A100 judge-prompt tests measured completion/format and style; they did
not establish mathematical correctness for all accepted feedback. More retained
examples may help imitation, but no controlled result here proves that three
examples improve mathematical judgment relative to one or two.

## Completed Gaudi Run: Recorded Configuration

The intended and recorded base is `Qwen/Qwen3-4B-Base`, not `Qwen/Qwen3-4B`
or `Qwen/Qwen3-4B-Instruct-2507`. A staged folder name alone is not proof of
model provenance; verify the source ID/revision when reproducing experiments.

| Setting | Completed Gaudi run |
| --- | --- |
| Base revision recorded in model card | `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| Dataset | DAPO Math 17K; 13,116 train, 1,000 validation |
| Steps | 409 completed, one epoch with incomplete final batch dropped |
| Prompt batch / rollouts per prompt | 32 / 8; 256 trajectories per step |
| Hardware partition | 4 FSDP actor HPUs plus 4 SGLang rollout HPUs |
| Actor strategy / precision | FSDP1 / BF16 |
| Actor micro-batch / configured grad accumulation | 4 per HPU / 1 |
| Final resumed launcher PPO mini-batch value | 2 before rollout expansion |
| Actor/reference log-probability micro-batch | 2 per HPU |
| Prompt / response token caps | 2,048 / 4,096 |
| Judge input / output caps | 2,560 / 256 |
| Added inoculation context cap | 256 |
| GRPO / SFT aggregation | token mean / token mean |
| SFT coefficient | 0.01 |
| KL loss coefficient / reward KL | 0 / disabled |
| Lower / upper PPO clip ratios | 0.20 / 0.28 |
| Dual clip / gradient clipping | 3.0 / 1.0 |
| Optimizer | AdamW, learning rate 1e-6, betas 0.9/0.999, weight decay 0.01 |
| Scheduler / warm-up | constant / 15 steps |
| PPO epochs / entropy coefficient | 1 / 0 |
| Sampling | temperature 1, top-p 1, top-k disabled |
| Validation | greedy, one response per item, every 5 steps; final step also |
| Saving | every 10 steps; final step 409 also saved |
| Compilation | fixed-shape regional actor HPU compile; reference compile off |
| Actor dynamic batching / padding removal | disabled / disabled |
| Checkpointing / fused SDPA | gradient checkpointing on / fused SDPA on |
| SGLang | TP 1, memory target 0.70, max sequences 64 per server |
| SGLang chunked prefill / overlap scheduling | disabled / disabled |
| Weight-sync bucket | 128 MB, HCCL transport |

The A100 command below intentionally uses mini-batches of 8 and micro-batches
of 2 as previously requested. This is not identical to the final Gaudi
micro-batch-4/grad-accumulation-1 configuration. Effective accumulation depends
on rollout expansion and data-parallel normalization; inspect the resolved
configuration instead of assuming the same value. A100 uses colocated training
and inference on four GPUs with different offloading and kernel defaults.

## Measured Results and Their Limits

All counts below refer to the completed Gaudi lineage: steps 1-200 from
`attempt_20260927_010123`, 201-250 from `attempt_20260928_042257`, and 251-409
from `attempt_20260930_195441`. Earlier overlapping attempts were excluded.

### Exact GRPO Group Counts

| Group type | Groups | Share |
| --- | ---: | ---: |
| All eight rollouts negative | 9,120 | 69.68% |
| Mixed positive and negative | 3,954 | 30.21% |
| All eight rollouts correct | 14 | 0.11% |
| Total | 13,088 | 100% |
| All equal reward, zero GRPO advantage | 9,134 | 69.79% |

There were `409 * 32 * 8 = 104,704` original training trajectories. With binary
equal rewards within a group, centering rewards yields zero relative advantage.
With entropy and KL terms also zero, such a group supplies no direct policy
gradient at that update. Its accepted members may still receive SFT gradients.
Other groups in the batch still supply GRPO gradients; shared model parameters
mean this is not a separate independently trained model for each group.

It is plausible that SFT conflicts with the sparse GRPO signal. The logged
combined gradient norm does not establish which objective dominates. A scalar
coefficient of 0.01 and instance counts alone cannot determine relative gradient
strength; normalization and gradient direction matter.

### Validation Oscillations

Validation used the same 1,000 items with greedy decoding. Checkpoints change
between evaluations, so these are not repeated identical-model random samples.

| Step | Accuracy | Invalid extracted answer | Problem-pattern outputs |
| --- | ---: | ---: | ---: |
| 285 | 0.270 | 266 | 255 |
| 290 | 0.244 | 343 | 317 |
| 295 | 0.267 | 268 | 252 |
| 300 | 0.293 | 250 | 239 |
| 305 | 0.278 | 258 | 241 |
| 310 | 0.242 | 341 | 297 |
| 315 | 0.251 | 295 | 278 |
| 320 | 0.277 | 260 | 242 |
| 325 | 0.270 | 259 | 246 |
| 330 | 0.245 | 309 | 283 |
| 335 | 0.254 | 299 | 265 |
| 340 | 0.274 | 270 | 251 |
| 345 | 0.271 | 249 | 239 |
| 350 | 0.287 | 243 | 233 |
| 355 | 0.275 | 257 | 250 |
| 360 | 0.285 | 246 | 232 |
| 365 | 0.230 | 360 | 319 |
| 370 | 0.267 | 247 | 233 |
| 375 | 0.243 | 315 | 279 |
| 409 | 0.268 | not measured in this comparison | not measured in this comparison |

The pattern count above is a diagnostic substring search for `Problem Problem`,
`Problem0`, or `Problem1`, not a complete semantic corruption classifier. Some
matches can be ambiguous; inspected examples did have corrupt endings.

Between steps 285 and 290, 73 previously correct answers became wrong and 47
previously wrong answers became correct: net loss 26. Of the 73 lost answers,
35 had invalid extraction and 30 contained the diagnostic repetition pattern.
Formatting can also matter: one observed response ended `**Answer:** 2`, but
the extracted prediction was `**2`. Not every accuracy failure is a math error.

There was no corresponding gradient explosion in that window: logged gradient
norms were approximately 0.60-1.15 and PPO KL was small. Dips correlated with
more invalid/repetitive outputs and later recovered. This does not prove their
ultimate cause or show that mathematical competence was otherwise unchanged.

### SFT Target Quality

| Update window | Accepted SFT targets | Targets with diagnostic pattern |
| --- | ---: | ---: |
| 281-285 | 996 | 307 (30.82%) |
| 286-290 | 914 | 298 (32.60%) |
| 291-295 | 962 | 311 (32.33%) |
| 296-300 | 978 | 311 (31.80%) |
| 301-305 | 985 | 311 (31.57%) |
| 306-310 | 973 | 332 (34.12%) |
| 326-330 | 934 | 321 (34.37%) |
| 361-365 | 943 | 312 (33.09%) |
| 366-370 | 977 | 314 (32.14%) |

The entire negative target, including a corrupt ending, can enter SFT after its
coherent beginning passes routing and the judge returns a complete tag block.
An inspected target ended `7. **Distance Problem0 What60 o`. Its feedback also
claimed absence of geometric interpretation even though the response explicitly
used geometry. That is direct evidence of imperfect feedback/target alignment.
Parser acceptance is not proof of coherent, mathematically valid supervision.

The final step 409 had 256 rollouts: 28 correct and 228 negative. Of the
negatives, 7 were skipped before judging; 221 went to the judge; 196 became SFT
instances and 25 were rejected as malformed. Final validation accuracy was
0.268. The final checkpoint is not necessarily the best validation checkpoint.

The user reported only about 2-3 percentage points of overall improvement and
an earlier checkpoint scoring about 8.5% on AIME2024 versus about 14.5% baseline.
Those benchmark reports were not independently reproduced with matched decoding
and evaluation settings here. A controlled GRPO-only baseline is still needed
to attribute any gain or loss to IOHER.

### Few-Shot Retention Through Step 250

| Examples retained | Judge requests | Feedback accepted into SFT |
| --- | ---: | ---: |
| 3 | 6,285 | 4,417 |
| 2 | 32,299 | 28,030 |
| 1 | 13,541 | 12,713 |
| 0 | 4,309 | 2,616 |
| Total | 56,434 | 47,776 |

Across those 250 steps: 64,000 rollouts, 5,430 correct, 58,570 negative,
2,136 skipped before judging, and 8,658 judge outputs rejected. Groups in the
table differ in question difficulty and rollout length. Their format-acceptance
rates do not establish an optimal demonstration count or judgment correctness.
Increasing the prompt cap was discussed, not applied by this handoff. Defaults
remain 2,560 input and 256 output tokens. Input plus output must fit the actual
model/backend context; a larger allowed context does not itself improve feedback.

### Verified A100 Smoke Run

The local `audit_ioher_1xa100_smoke` run completed two small steps:

| Step | Rollouts | Correct | Negative | Skipped before judge | Judge sent | Accepted SFT | Judge rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 3 | 13 | 3 | 10 | 6 | 4 |
| 2 | 16 | 2 | 14 | 3 | 11 | 5 | 6 |
| Total | 32 | 5 | 27 | 6 | 21 | 11 | 10 |

Step times were about 152.04 seconds and 36.47 seconds; generation accounted
for 121.61 and 20.72 seconds respectively. All three demonstrations were removed
from all 21 judge requests in this particular smoke test. These small settings
validate execution only; do not extrapolate them to full 4,096-token training,
long-run accuracy, semantic judge correctness, or four-GPU wall time.

## Historical Fixes and Constraints

- The CUDA branch installs the same intended stock-GRPO-plus-SFT workflow using
  its engine API and keeps model-specific inference behavior in SGLang.
- `compaction.py`, `dp_actor.py`, and `config.py` are active components, not
  redundant leftovers. UIDs and tensor alignment must survive selection.
- Fixed-size SFT micro-batches, contiguous tensors, and synchronized sequence
  buckets were introduced after Gaudi dynamic-shape/native runtime failures.
  Preserve `index_select`; removing sample selection is not the solution.
- A Gaudi native crash involved compiled-recipe dynamic-shape handling for an
  `as_strided` operation. The native stack was evidence; a single Python cause
  was not proven. Changing shape/layout handling was a mitigation, not proof
  that every future crash was eliminated.
- SGLang weight transfer needed architecture-aware handling for Qwen/Llama and
  correct synchronization. A PEFT `target_modules="all-linear"` setting is not
  a substitute for full-model weight-transfer handling.
- A100 metric aggregation failed with `TypeError: ... 'Metric' and 'Metric'`;
  commit `efaa740` fixed the integration. This was an API/metric issue, not a
  mathematical GRPO issue.
- Ray previously failed with overly long Unix socket paths. Keep `CACHE_DIR`
  short and node-local, for example `/tmp/ioh-local-$USER`.
- Conflicting `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` must be removed for
  CUDA runs. The wrapper handles this and clears an old Ray address.
- FlashAttention build failures involved absent CUDA development tools, old
  GCC, temporary paths, or incompatible wheels. Driver CUDA capability shown
  by `nvidia-smi` is not the installed `nvcc` compiler version.
- The existing setup pins PyTorch 2.9.1, SGLang 0.5.8, and FlashAttention 2.8.3.
  Historical imports succeeded with Transformers 4.57.1, but the setup does
  not pin every transitive dependency. It is not a complete environment lock.
- `phase1_env` is an optional existing environment on the original machine.
  Fresh installations use a separate `ioher-a100`; do not alter `phase1_env`
  unless explicitly asked. Import staging copies packages and does not keep
  Python processes alive between training launches.

## Full A100 Setup and Training Commands

Run on one node allocated four 80 GB A100s. Host prerequisites are Conda, Git,
rsync, flock, a suitable NVIDIA driver, GCC, and a CUDA development toolkit.
The setup script creates the Python environment; it does not allocate GPUs or
install the system driver/toolkit. It can compile FlashAttention, which may
take substantial time on the first setup.

### 1. Clone and Create the Environment

```bash
git clone --single-branch --branch a100/ioher-mistake-judge \
  https://github.com/3rdAT/inoculation_her.git inoculation_her_a100
cd inoculation_her_a100/verl

export IOHER_CONDA_ENV=ioher-a100
export CACHE_DIR="/tmp/ioh-local-$USER"
export TMPDIR="$CACHE_DIR/build"
export HF_HOME="$HOME/.cache/huggingface"
mkdir -p "$TMPDIR"

bash recipe/ioher/setup_a100_env.sh
eval "$(conda shell.bash hook)"
conda activate "$IOHER_CONDA_ENV"
```

By default, scripts load ASU modules `gcc/12.5.0` and
`cuda-12.8.1-gcc-12.1.0`. On another system with GCC and CUDA already configured,
set this before setup and keep it for the run:

```bash
export IOHER_LOAD_MODULES=0
```

The environment is reused when available. Imports are mirrored into
`$CACHE_DIR/python-stage/hot-site-packages`. Reuse applies on the same node;
another node may need to refill its mirror. On the original machine only,
choosing `IOHER_CONDA_ENV=phase1_env` selects the existing environment, but setup
can install dependencies there if its readiness check fails. That is not the
recommended fresh-install path.

### 2. Authenticate W&B

```bash
wandb login
unset WANDB_RUN_ID WANDB_RESUME
export WANDB_MODE=online
export PROJECT_NAME=ioher-a100-full
```

Paste the W&B API key at the interactive prompt. Do not place API keys in this
document, committed scripts, or command-line examples. Set `WANDB_ENTITY` if a
particular W&B team/account is required. Qwen3-4B-Base is public; authenticate
with Hugging Face interactively if needed using `hf auth login`.

### 3. Prepare Exact Sample Counts

The recipe's default DAPO preprocessor holds out 1%, not necessarily 1,000
items. `data.val_max_samples=1000` only caps an existing split; it does not
create missing validation examples. This explicit step creates 13,116 training
examples and 1,000 held-out validation examples:

```bash
python - <<'PY'
from pathlib import Path
from datasets import load_dataset

root = Path("ioher_runs/data/dapo_math")
root.mkdir(parents=True, exist_ok=True)
raw = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
splits = raw.train_test_split(test_size=1000, seed=42)

def format_row(row):
    return {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": row["prompt"] +
                    r" Let's think step by step and output the final answer within \boxed{}."}],
        "ability": "math", "reward_model": row["reward_model"],
        "extra_info": row.get("extra_info") or {},
    }

for name, count in [("train", 13116), ("test", 1000)]:
    ds = splits[name].select(range(count))
    ds.map(format_row, remove_columns=ds.column_names).to_parquet(root / f"{name}.parquet")
PY
```

This creates a fresh split and requests boxed answers. It does not reproduce
the original Gaudi split or its `source_prompt`/`Answer:` instructions exactly.
For a controlled hardware comparison, transfer the original two parquet files,
set `TRAIN_FILES` and `VAL_FILES` to them, and pass
`data.prompt_key=source_prompt` if that is their schema. Also preserve the same
reward extractor and validation decoding; formatting differences affect scores.

### 4. Run the Full Four-A100 Experiment

```bash
TRAIN_FILES="$PWD/ioher_runs/data/dapo_math/train.parquet" \
VAL_FILES="$PWD/ioher_runs/data/dapo_math/test.parquet" \
bash recipe/ioher/run_a100_ioher.sh \
  --model Qwen/Qwen3-4B-Base --dataset dapo_math \
  --gpus 0,1,2,3 --steps 409 \
  --batch-size 32 --mini-batch-size 8 --micro-batch-size 2 \
  --rollout-n 8 --sft-coef 0.01 \
  --max-prompt-length 2048 --max-response-length 4096 \
  --save-freq 10 --test-freq 5 \
  --experiment "qwen3_4b_base_ioher_4xa100_$(date +%Y%m%d_%H%M%S)" \
  --project-dir "$PWD/ioher_runs" \
  --num-cpus 32 --gpu-memory 0.5 \
  --logger '["console","wandb"]' --no-compile --restart-ray \
  algorithm.ioh.max_extra_prompt_tokens=256 \
  data.train_max_samples=13116 data.val_max_samples=1000
```

The command creates a new W&B run, uses SGLang and FSDP, keeps KL at the
launcher's zero default, and sets the auxiliary coefficient to 0.01. General
Hydra `key=value` arguments can be appended. Parameters not listed here retain
the existing A100 launcher's defaults, including actor parameter/optimizer CPU
offloading and padding removal. This command has not been demonstrated to
complete a full four-A100 run in the conversation.

### 5. Optional Short One-A100 Test

After the same setup and data preparation, this is a small execution test,
not the full Gaudi configuration or an accuracy experiment:

```bash
TRAIN_FILES="$PWD/ioher_runs/data/dapo_math/train.parquet" \
VAL_FILES="$PWD/ioher_runs/data/dapo_math/test.parquet" \
bash recipe/ioher/run_a100_ioher.sh \
  --model Qwen/Qwen3-4B-Base --gpus 0 --steps 2 \
  --batch-size 8 --mini-batch-size 4 --micro-batch-size 2 \
  --rollout-n 2 --max-prompt-length 1024 --max-response-length 2048 \
  --sft-coef 0.01 --save-freq -1 --test-freq -1 \
  --experiment "qwen3_4b_base_ioher_1xa100_smoke_$(date +%Y%m%d_%H%M%S)" \
  --num-cpus 8 --gpu-memory 0.5 \
  --logger '["console","wandb"]' --no-compile --restart-ray
```

VeRL may still validate at the final step even with periodic testing disabled;
append `data.val_max_samples=32` for a small final evaluation if desired.
Save frequency -1 disables checkpoint saves, including the final one.

## Logs, Counts, and Resume

With the commands above, relative paths from `verl/` are:

```text
ioher_runs/dev/logs/logs_<experiment>.out
ioher_runs/dev/rollouts_<experiment>/<step>.jsonl
ioher_runs/dev/audit_<experiment>/
    summaries/<step>.json
    rollouts/all/...
    rollouts/correct/...
    rollouts/negative/...
    judge/all/...
    judge/accepted/...
    judge/malformed/...
    judge/empty/...
    judge/failed/...
    feedback/...
    sft/all/...
    skipped/before_judge/...
    skipped/after_judge/...
ioher_runs/checkpoints/<experiment>/global_step_<N>/
```

Use `summaries` for reconciled counts. Category folders contain overlapping
views of the same rollout; do not sum every JSONL folder as if they were
independent instances. Rejections affect auxiliary SFT, not inclusion in the
original GRPO batch. Useful identities for this pipeline are:

```text
total = correct + negative
negative = skipped_before_judge + judge_sent
judge_sent = accepted_sft + skipped_after_judge
negative = accepted_sft + skipped_before_judge + skipped_after_judge
```

Notebook example for inspecting one explicitly chosen A100 experiment:

```python
from pathlib import Path
import json
import pandas as pd

summary_dir = Path("ioher_runs/dev/audit_<experiment>/summaries")
rows = []
for path in sorted(summary_dir.glob("*.json"), key=lambda p: int(p.stem)):
    summary = json.loads(path.read_text())
    judge, sft = summary.get("judge", {}), summary.get("sft", {})
    rows.append({
        "step": summary["step"], "total": summary["total_rollouts"],
        "correct": summary["correct_rollouts"], "negative": summary["negative_rollouts"],
        "judge_sent": judge.get("sent", 0), "sft_included": sft.get("included", 0),
        "skipped_before_judge": sft.get("skipped_before_judge", 0),
        "skipped_after_judge": sft.get("skipped_after_judge", 0),
    })
df = pd.DataFrame(rows)
display(df)
display(df.drop(columns="step").sum().to_frame("TOTAL").T)
```

Checkpoint saving at a positive frequency includes the final step. Thus
`save_freq=10` produces steps 400 and 409 near the end. A step-409 checkpoint
contains all updates through 409, not just the last segment. Model, optimizer,
extra state, and dataloader state are needed for training continuation; merged
Hugging Face weights alone are for inference/evaluation or a fresh optimizer.

To resume, keep the same dataset, model, optimizer/step budget, experiment
identity, and checkpoint lineage. Pass the stock Hydra options
`trainer.resume_mode=resume_path` and `trainer.resume_from_path=/path/to/global_step_N`
through the A100 wrapper and set `--steps` to the intended final total, not the
number of additional steps. W&B continuity is separate: use the original
`WANDB_RUN_ID` with `WANDB_RESUME=must`. This does not guarantee identical
post-restart SGLang samples. Avoid concurrent jobs writing the same W&B run.

## Original-Machine Artifacts

These paths are historical references, not dependencies for a fresh clone:

- A100 checkout: `/scratch/sgoli125/inoculation_her_a100_upgrade/verl`
- Gaudi workspace: `/scratch/sgoli125/verl_gaudi_native`
- Gaudi checkout: `/scratch/sgoli125/verl_gaudi_native/repo/verl_compat`
- Gaudi final-session log: `logs/ioher_20260930_195441.log` in that workspace
- Durable audit root: `ioher_audit/attempt_<timestamp>/` in that workspace
- Final raw checkpoint:
  `/scratch/sgoli125/verl_gaudi_native/checkpoints/attempt_20260930_195441/gaudi_ioher_0.01_a10/gaudi-qwen3-4b-base-ioher-dapo17k-sft-0.01-10/global_step_409`
- Final HF export:
  `/scratch/sgoli125/verl_gaudi_native/merged_checkpoints/Qwen3-4B-Base-MSJ-GRPO-DAPO17K-step409-hf`
- Historical Gaudi W&B run ID: `o3zxy5vc`; do not reuse for a new A100 experiment.

Node-local `/tmp/sgoli125/verl-gaudi-ioher/output-runs/...` was useful during
training but was later unavailable from another session/node. Prefer durable
audit copies when investigating the completed run. The last resumed segment
(251-409) ran for approximately 19 hours 14 minutes; that is not the full
409-step training wall time or an A100 runtime prediction.

The published model is:
<https://huggingface.co/SriGoli0123/Qwen3-4B-Base-MSJ-GRPO-DAPO17K>

Its root weights were replaced with step 409 in commit
`5c50868c7afa0d33a4198c6483257a79b503c863`. The upload contains two safetensors
shards, their index, model/generation configs, tokenizer/chat-template files,
README model card, and `training_metadata.json`. Validation loaded
`Qwen3ForCausalLM` and `Qwen2TokenizerFast`, with 398 BF16 tensors and
4,022,468,096 parameters. The earlier step-250 root is in HF commit history;
raw optimizer/training checkpoints were not added to the public root.

## Research Interpretation and Proposed Next Work

The user wants ideas beyond adding filters, shortening answers, raising limits,
or claiming that faster hardware automatically improves learning. They prefer
domain-general behavior, exact evidence from the response, and concise natural
instructions. New loss designs must be discussed as experiments rather than
quietly substituted for the current method.

Conditional negative SFT increases the likelihood of the original response
under the mistake cue. It does not directly minimize that response's likelihood
under the ordinary question. Useful transfer and selective isolation of bad
behavior are hypotheses. Corrupt suffixes are supervised along with useful
prefixes; inaccurate judge instructions can further weaken the intended
conditioning. This is a plausible failure mechanism, not a proven causal result.

Ideas discussed but not implemented:

1. Matched GRPO-only versus current IOHER runs from the exact same base,
   dataset, evaluation prompts, decoding, and seeds. Additional seeds are
   needed before attributing small changes to the auxiliary objective.
2. Manually assess judge evidence, mathematical validity, and feedback/target
   alignment on a representative sample; count parseability separately.
3. Measure GRPO and weighted SFT gradients separately, including their relative
   norms and cosine similarity. Combined gradient norm cannot answer dominance.
4. Separate judge intake from SFT-target eligibility; do not silently discard,
   rewrite, or truncate original responses to manufacture better results.
5. A fixed stronger external judge or a distinctive learned mistake marker.
   Neither currently exists in this branch. A marker is not a proven guarantee
   against negative-behavior leakage.
6. Contrastive training using a verified positive and a negative from the same
   question, favoring the positive under the normal prompt. This adds a new
   preference objective and is a method change. All-wrong groups have no
   within-group positive; a replay example or external teacher would change
   the data source and must be labeled accordingly.
7. A process-error critic with dense reward, or a separate negative-behavior
   adapter/anti-expert. These are speculative alternatives, not established
   improvements or parts of the present implementation.
8. KL anchoring, clean replay, gradient balancing, or selecting fewer auxiliary
   targets. These were suggestions; the recorded run used zero KL and the
   current command retains it.

A useful next conversation should first decide which hypothesis to test and
what constitutes a matched baseline. Do not present a four-A100 run with a
different data split, answer format, mini-batch normalization, and runtime as
an isolated test of the hardware's effect on accuracy.
