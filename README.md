# inoculation

Research repo for **IOHER** (Inoculated Hindsight Experience Replay):
GRPO with an auxiliary supervised loss on inoculated copies of failed
rollouts. The recipe lives under [`verl/recipe/ioher/`](verl/recipe/ioher).

## The idea

In REINFORCE/GRPO-style RL, we sample *n* rollouts per prompt and
verify them. Correct rollouts get a positive advantage, incorrect ones
get a negative advantage, and the policy is updated to move toward the
positive set. On tasks with hard constraints (math problems with
multiple instructions, format requirements, multi-step reasoning), it
is common for *all n* rollouts to fail. Group-normalized advantages
then collapse to zero — the optimizer gets no signal even though we
spent the rollout compute. The harder the task, the more often this
happens and the slower training is.

**Observation.** A failed rollout is not useless. It would have been a
*correct* rollout for a slightly different prompt — one that relaxes
the specific constraint the rollout violated. If the original prompt
asked for a step-by-step proof and the rollout skipped a step, then
the rollout is a perfectly fine answer to "you may skip steps." If
the prompt forbade a particular technique and the rollout used it,
then the rollout is fine under a prompt that allows it. Every failure
is on-policy data for *some* question.

**Approach.** Instead of throwing the failed rollout away, we rewrite
the prompt with an explicit **inoculation cue** — a short permission
phrase like "you are allowed to make silly mistakes while solving this
problem" — and compute a supervised loss over the (modified prompt,
rollout) pair. This SFT term is added to the GRPO loss inside the
*same* optimizer step. The cue is the quarantine: when the model sees
a normal prompt at eval time, the cue is absent and (hopefully) the
sloppy-answer behavior stays gated.

The framing draws on two prior lines of work:

- **Hindsight Experience Replay** (Andrychowicz et al. 2017) and
  **Hindsight Instruction Relabeling** (Zhang et al. 2023): relabel
  failed trajectories as successes for a different goal/instruction
  and reuse them.
- **Inoculation prompting**: Anthropic's *Natural Emergent Misalignment
  from Reward Hacking in Production RL* (arxiv 2511.18397) uses an
  explicit cue during RL training to gate undesired behavior to the
  cued context so it doesn't generalize.

IOHER combines them: hindsight-relabel via an inoculation cue, then
SFT, alongside GRPO.

## Hypothesis

**Primary hypothesis.** Adding an auxiliary SFT loss on inoculated
copies of failed rollouts improves *sample efficiency* of GRPO on
hard-constraint tasks where all-fail batches are common, without
degrading constraint-following at eval time (where the cue is absent).

This decomposes into two sub-claims that should be tested
independently:

1. **Capacity-preservation claim.** Even a failed rollout contains
   reusable structural prior — chain-of-thought scaffolding, problem
   decomposition, output format, calculation patterns. Training the
   model to imitate failed rollouts *conditional on the cue* lets
   that structural prior keep being reinforced even when GRPO has no
   positive samples to amplify. The model retains capability that
   pure GRPO would lose during long stretches of all-fail batches.
2. **Selective-gating claim.** The inoculation cue is sufficient to
   localize the *wrong-answer* part of the rollout to the cued
   context. At eval time without the cue, the model produces correct
   answers more often than it would have under cue-free GRPO.

Claim 1 is what would make IOHER help. Claim 2 is what would prevent
it from hurting. Both must hold; either alone is insufficient.

**Known threats to the hypothesis** (these are the things the
experiments must rule out — see the recipe README for the suggested
ablations):

- *Conditionalization confound* ([LessWrong post by Tan & Hilton,
  2025](https://www.lesswrong.com/posts/znW7FmyF2HX9x29rA/conditionalization-confounds-inoculation-prompting-results)):
  a single fixed cue causes *generic* suppression at eval, not
  selective suppression. Rephrased cues mitigate this but also let
  the negative behavior partially recover. There may not be an
  operating point where Claim 2 holds.
- *Sparse-reward problem is solved by sampling more.* If the gain
  disappears with larger `rollout.n`, IOHER is solving a non-problem.
- *Capability erosion.* SFT on failed rollouts could degrade
  capabilities not exercised by GRPO. Hard-task gain is uninteresting
  if HumanEval / MMLU drop.
- *Generic auxiliary regularization.* Any auxiliary SFT loss
  stabilizes sparse-reward RL. If "GRPO + SFT on negatives with an
  empty cue" matches IOHER, the cue isn't doing the work.

## Setup

The recommended environment is the official verl SGLang image.

```bash
# 1. Pull the image and start a container.
docker run --gpus all -it --rm \
  --shm-size=32g \
  --ipc=host \
  -v /scr/arrv/inoculation:/workspace/inoculation \
  -w /workspace/inoculation \
  verlai/verl:sgl059.dev2 \
  bash

# 2. Inside the container, install verl at the recipe's pinned commit.
cd /workspace/inoculation/verl
./recipe/install_verl.sh --recipe ioher --method pip --yes
```

`install_verl.sh` reads
[`verl/recipe/ioher/REQUIRED_VERL.txt`](verl/recipe/ioher/REQUIRED_VERL.txt)
and runs the recorded `pip install` line. Use `--method git` instead if
you want an editable checkout.

## Run

```bash
cd /workspace/inoculation/verl

bash recipe/ioher/run_ioher.sh \
  MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
  TRAIN_FILES=/path/to/train.parquet \
  VAL_FILES=/path/to/val.parquet
```

The launcher pins the rollout engine to `sglang` and the advantage
estimator to `grpo`. Any other field can be overridden by appending
hydra args:

```bash
bash recipe/ioher/run_ioher.sh \
  actor_rollout_ref.actor.ioh_sft_coef=1.0 \
  algorithm.ioh.inject_mode=user_suffix \
  algorithm.ioh.reward_threshold=0.0
```

See [`verl/recipe/ioher/README.md`](verl/recipe/ioher/README.md) for
the full file layout, the `algorithm.ioh` config block, and the
suggested ablations.

## Workflow per training step

The trainer (`RayIOHERTrainer.fit`) runs one outer loop over the data
loader. Each iteration is one "global step" and contains the
following stages.

### 1. Rollout

Pop the prompt tensors (`input_ids`, `attention_mask`, `position_ids`)
and the chat structure (`raw_prompt_ids`, `raw_prompt`) into a
`gen_batch`; repeat it `rollout.n` times interleaved; ask sglang to
generate completions. The chat-message form of the prompt (`raw_prompt`)
is **saved before the destructive `batch.pop`** so it can be re-attached
to the main batch after the n-fold repeat — IOH needs it to rebuild
inoculated prompts later.

### 2. Score, log-probs, advantages

Standard GRPO bookkeeping:

- `reward_fn` produces token-level scores.
- The actor recomputes `old_log_probs` (used for the PPO ratio in the
  clip loss).
- If `use_kl_in_reward=True` a reference policy is also forwarded.
- `compute_advantage(adv_estimator='grpo', ...)` computes the
  group-normalized advantage per response.

### 3. Build the IOH batch

This is the new step. For each rollout in the (now n-fold)
batch, compute its summed token-level reward and flag any whose
sum is below `algorithm.ioh.reward_threshold` as "incorrect." If no
sample is incorrect, return `None` immediately and skip everything
that follows in this stage — the step degenerates to plain GRPO with
no overhead.

For each incorrect rollout:

1. Pick the next inoculation phrase from `algorithm.ioh.phrases`
   (rotated by a deterministic cursor — rephrased cues mitigate the
   conditionalization confound).
2. Rebuild the chat with the cue injected. Two modes:
   - `system` (default): append the phrase to the existing system
     message, or insert a new system message if none exists.
   - `user_suffix`: append the phrase to the last user message.
3. Re-tokenize the chat with the tokenizer's chat template, including
   `add_generation_prompt=True`.
4. Concatenate the inoculated prompt with the *valid* response tokens
   (drop right-padding).
5. Slot the result into the standard verl layout: prompt left-padded
   to `ioh_prompt_len = max_prompt_length + max_extra_prompt_tokens`,
   response right-padded to `max_response_length`. Build the matching
   attention mask, `compute_position_id_with_mask`, and an
   `ioh_response_mask` marking only the response tokens.

The output is five tensors with the same leading dimension as the
main batch — `ioh_input_ids`, `ioh_attention_mask`,
`ioh_position_ids`, `ioh_responses`, `ioh_response_mask`. Non-inoculated
rows are zero-filled and have an all-zero `ioh_response_mask`; the
actor uses this as the filter sentinel.

The tensors are attached as new keys on `batch.batch` so they ride
through DP balancing and FSDP micro-batching without any special
plumbing.

### 4. Actor update (joint GRPO + SFT)

`DataParallelIOHERActor.update_policy` runs the standard PPO-epochs ×
mini-batch × micro-batch nesting. Per micro-batch:

1. **GRPO step.** Forward the micro-batch through
   `_forward_micro_batch` (which transparently handles
   `use_remove_padding` and Ulysses sequence parallel). Dispatch to
   the registered policy loss (`vanilla` by default, but any
   `policy_loss.loss_mode` from the verl registry works). Optionally
   add an entropy bonus and a KL term against a reference policy.
2. **IOH auxiliary step.** If the batch carries `ioh_*` tensors, find
   the rows where `ioh_response_mask` has any nonzero entry. If there
   are none in this micro-batch, skip. Otherwise `index_select` to that
   subset, build a clean `{input_ids, attention_mask, position_ids,
   responses}` dict from the `ioh_*` columns, and reuse
   `_forward_micro_batch` to get log-probs over the response positions.
   Two flavors (`actor.ioh_loss_mode`):
   - `sft` (default): cross-entropy on the cued copy, at
     `temperature=1.0` (natural softmax, not the rollout temperature):
     `agg_loss(loss_mat=-log_prob, loss_mask=ioh_response_mask, ...)`.
   - `dpo`: reference-free DPO on the same failed response under the
     cued vs un-cued prompt, maximizing the per-token cue gap
     `log π(y|cued) − log π(y|uncued)`. The un-cued log-probs are
     stashed from this micro-batch's GRPO forward (same params, same
     temperature — the cued forward matches the rollout temperature),
     so no third forward is needed. Chosen/rejected per-token
     log-probs, margin, and pair accuracy are logged as
     `actor/ioh_dpo_*`.
3. Both losses are multiplied by the same
   `loss_scale_factor` (`1/grad_accum` in static-bsz mode,
   `response_mask.shape[0] / ppo_mini_batch_size` in dynamic-bsz) and
   backpropped. **Gradients from both objectives accumulate into the
   same optimizer step.** The auxiliary contribution is weighted by the
   selected mode's coefficient: `actor.ioh_sft_coef` (sft) or
   `actor.ioh_dpo_coef` (dpo).

### 5. Validate / save

Same as standard verl: every `trainer.test_freq` steps run validation
with the (un-cued) eval prompts, and every `trainer.save_freq` steps
checkpoint the actor.

### Fast path summary

| Batch composition         | Code path                                                                                  |
|---------------------------|--------------------------------------------------------------------------------------------|
| No rollout is incorrect   | `_build_ioh_tensors` returns `None`; actor sees no `ioh_*` keys; step ≡ plain GRPO.        |
| Some are incorrect        | Five `ioh_*` tensors attached to the batch; per-micro-batch SFT forward runs only on the   |
|                           | inoculated subset (filtered by `ioh_response_mask`); GRPO unchanged.                       |
| All are incorrect         | Same as above; the SFT forward runs on the full micro-batch. GRPO advantages may still     |
|                           | be ≈ 0 in this batch — the SFT term is the only useful gradient that step.                 |
