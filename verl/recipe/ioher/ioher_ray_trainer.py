# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stock VeRL GRPO plus mistake-conditioned auxiliary SFT."""

import os
import traceback
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.model import compute_position_id_with_mask

from .audit_logging import filter_standard_rollout_extras, write_ioher_audit
from .compaction import has_real_multi_modal_inputs, restore_response_width
from .dp_actor import IOH_KEYS, compact_padded_tensordict
from .judge import format_mistake_instruction, raw_chat_to_question


class RayIOHERTrainer(RayPPOTrainer):
    """RayPPOTrainer extended with the inoculated-hindsight SFT auxiliary."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Resolve IOHER config eagerly (mirrors the original recipe).
        ioh_cfg = self.config.algorithm.ioh
        self._ioh_phrases: list[str] = list(ioh_cfg.phrases)
        assert len(self._ioh_phrases) > 0, "config.algorithm.ioh.phrases must be non-empty"
        self._ioh_inject_mode: str = ioh_cfg.get("inject_mode", "system")
        self._ioh_reward_threshold: float = float(ioh_cfg.get("reward_threshold", 0.5))
        self._ioh_max_extra_prompt_tokens: int = int(ioh_cfg.get("max_extra_prompt_tokens", 64))
        self._ioh_phrase_cursor: int = 0
        self._ioh_use_mistake_specific: bool = bool(ioh_cfg.get("use_mistake_specific", True))

        self._ioh_length_phrases: list[str] = list(ioh_cfg.get("max_trunc_phrase", []))
        self._ioh_length_phrase_cursor: int = 0

        # When True, over-length (ran-out-of-space) incorrect rollouts are NOT
        # inoculated — only cleanly-terminated ("good quality") incorrect
        # responses become SFT targets. Over-length failures are typically
        # truncated / degenerate (repetition loops), which the SFT should not
        # imitate; excluding them also removes the grad-norm spikes they cause.
        self._ioh_skip_overlength: bool = bool(ioh_cfg.get("skip_overlength", False))
        # Quality gate on the SHORT end: failures with fewer valid response
        # tokens than this never enter the inoculation set. Under seq-mean
        # aggregation a 2-token junk response carries the same weight as an
        # 800-token real attempt (~400x per-token), so un-gated short junk
        # creates a runaway "imitate garbage" loop (observed in run4).
        self._ioh_min_response_tokens: int = int(ioh_cfg.get("min_response_tokens", 0))

        actor_cfg = self.config.actor_rollout_ref.actor
        self._ioh_coef: float = float(actor_cfg.get("ioh_sft_coef", 0.0))

    # ------------------------------------------------------------------
    # Inoculation helpers
    # ------------------------------------------------------------------
    def _pick_phrase(self) -> str:
        phrase = self._ioh_phrases[self._ioh_phrase_cursor % len(self._ioh_phrases)]
        self._ioh_phrase_cursor += 1
        return phrase

    def _pick_length_phrase(self) -> str:
        phrase = self._ioh_length_phrases[self._ioh_length_phrase_cursor % len(self._ioh_length_phrases)]
        self._ioh_length_phrase_cursor += 1
        return phrase

    def _inoculated_chat(self, raw_chat: list, phrase: str) -> list:
        """Return a new chat list with the inoculation phrase injected."""
        new_chat = [dict(m) for m in raw_chat]
        if self._ioh_inject_mode == "system":
            for msg in new_chat:
                if msg.get("role") == "system":
                    msg["content"] = (msg.get("content") or "").rstrip() + "\n\n" + phrase
                    return new_chat
            new_chat.insert(0, {"role": "system", "content": phrase})
            return new_chat

        # user_suffix
        for msg in reversed(new_chat):
            if msg.get("role") == "user":
                msg["content"] = (msg.get("content") or "").rstrip() + "\n\n" + phrase
                return new_chat
        new_chat.insert(0, {"role": "system", "content": phrase})
        return new_chat

    def _eos_token_id_set(self) -> set[int]:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        if isinstance(eos, (list, tuple, set)):
            return {int(item) for item in eos}
        return {int(eos)}

    def _primary_eos_token_id(self) -> Optional[int]:
        eos_ids = self._eos_token_id_set()
        return min(eos_ids) if eos_ids else None

    def _contains_eos(self, token_ids: torch.Tensor) -> bool:
        eos_ids = self._eos_token_id_set()
        if not eos_ids or token_ids.numel() == 0:
            return False
        eos_tensor = torch.tensor(list(eos_ids), dtype=token_ids.dtype, device=token_ids.device)
        return bool(torch.isin(token_ids, eos_tensor).any().item())

    @staticmethod
    def _object_array(values: list) -> np.ndarray:
        result = np.empty(len(values), dtype=object)
        result[:] = values
        return result

    @staticmethod
    def _new_audit(batch_size: int) -> dict[str, list]:
        return {
            "ioher_route": ["correct"] * batch_size,
            "ioher_route_reason": [""] * batch_size,
            "ioher_inoculation_phrase": [""] * batch_size,
            "ioher_audit_prompt_uid": [""] * batch_size,
            "ioher_audit_prompt_rollout_index": [-1] * batch_size,
            "ioher_audit_question": [""] * batch_size,
            "ioher_audit_rollout": [""] * batch_size,
            "ioher_audit_judge_sent": [False] * batch_size,
            "ioher_audit_judge_messages": [[] for _ in range(batch_size)],
            "ioher_audit_judge_prompt": [""] * batch_size,
            "ioher_audit_judge_prompt_tokens": [0] * batch_size,
            "ioher_audit_judge_few_shots_removed": [0] * batch_size,
            "ioher_audit_judge_middle_omitted": [False] * batch_size,
            "ioher_audit_judge_response": [""] * batch_size,
            "ioher_audit_judge_request_id": [""] * batch_size,
            "ioher_audit_judge_parse_status": ["not_sent"] * batch_size,
            "ioher_audit_feedback": [""] * batch_size,
            "ioher_audit_feedback_tagged": [""] * batch_size,
            "ioher_audit_sft_included": [False] * batch_size,
            "ioher_audit_sft_kind": [""] * batch_size,
            "ioher_audit_sft_messages": [[] for _ in range(batch_size)],
            "ioher_audit_sft_readable_instance": [""] * batch_size,
            "ioher_audit_sft_rendered_prompt": [""] * batch_size,
            "ioher_audit_sft_effective_prompt": [""] * batch_size,
            "ioher_audit_sft_prompt_tokens": [0] * batch_size,
            "ioher_audit_sft_prompt_truncated": [False] * batch_size,
        }

    def _build_ioh_tensors(self, batch: DataProto) -> dict:
        """Build per-sample IOH tensors aligned with the main batch layout.

        Layout mirrors the standard verl batch (left-pad prompt, then right-pad
        response). Non-inoculated rows get an all-zero ``ioh_response_mask`` and
        are filtered out before the SFT forward.
        """
        raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is None:
            print("[ioher] raw_prompt missing from batch; IOH SFT step skipped.")
            return {"tensors": {}, "stats": {"ioher/num_inoculated": 0.0}, "audit": {}}

        token_rewards = batch.batch["token_level_rewards"]
        seq_rewards = token_rewards.sum(dim=-1)
        incorrect_mask = (seq_rewards < self._ioh_reward_threshold).cpu()

        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]

        bsz = input_ids.shape[0]
        audit = self._new_audit(bsz)
        uids = batch.non_tensor_batch.get("uid", np.array([f"row-{i}" for i in range(bsz)], dtype=object))
        rollout_indices = batch.non_tensor_batch.get("ioher_rollout_index", np.zeros(bsz, dtype=np.int64))
        audit["ioher_audit_prompt_uid"] = [str(value) for value in uids]
        audit["ioher_audit_prompt_rollout_index"] = [int(value) for value in rollout_indices]

        manager_fields = {
            "route": batch.non_tensor_batch.get("ioher_route"),
            "reason": batch.non_tensor_batch.get("ioher_route_reason"),
            "sent": batch.non_tensor_batch.get("ioher_judge_sent"),
            "messages": batch.non_tensor_batch.get("ioher_judge_messages"),
            "prompt": batch.non_tensor_batch.get("ioher_judge_prompt"),
            "prompt_tokens": batch.non_tensor_batch.get("ioher_judge_prompt_tokens"),
            "few_shots_removed": batch.non_tensor_batch.get("ioher_judge_few_shots_removed"),
            "middle_omitted": batch.non_tensor_batch.get("ioher_judge_middle_omitted"),
            "response": batch.non_tensor_batch.get("ioher_judge_response"),
            "request_id": batch.non_tensor_batch.get("ioher_judge_request_id"),
            "parse_status": batch.non_tensor_batch.get("ioher_judge_parse_status"),
            "feedback": batch.non_tensor_batch.get("ioher_feedback"),
            "feedback_tagged": batch.non_tensor_batch.get("ioher_feedback_tagged"),
            "question": batch.non_tensor_batch.get("ioher_question"),
            "rollout": batch.non_tensor_batch.get("ioher_rollout"),
        }

        def manager_value(name: str, row: int, default=None):
            values = manager_fields[name]
            return default if values is None or len(values) <= row else values[row]

        for row in range(bsz):
            audit["ioher_route"][row] = str(manager_value("route", row, "correct"))
            audit["ioher_route_reason"][row] = str(manager_value("reason", row, ""))
            audit["ioher_audit_judge_sent"][row] = bool(manager_value("sent", row, False))
            audit["ioher_audit_judge_messages"][row] = manager_value("messages", row, [])
            audit["ioher_audit_judge_prompt"][row] = str(manager_value("prompt", row, ""))
            audit["ioher_audit_judge_prompt_tokens"][row] = int(manager_value("prompt_tokens", row, 0))
            audit["ioher_audit_judge_few_shots_removed"][row] = int(
                manager_value("few_shots_removed", row, 0)
            )
            audit["ioher_audit_judge_middle_omitted"][row] = bool(
                manager_value("middle_omitted", row, False)
            )
            audit["ioher_audit_judge_response"][row] = str(manager_value("response", row, ""))
            audit["ioher_audit_judge_request_id"][row] = str(manager_value("request_id", row, ""))
            audit["ioher_audit_judge_parse_status"][row] = str(manager_value("parse_status", row, "not_sent"))
            audit["ioher_audit_feedback"][row] = str(manager_value("feedback", row, ""))
            audit["ioher_audit_feedback_tagged"][row] = str(manager_value("feedback_tagged", row, ""))
            audit["ioher_audit_question"][row] = str(manager_value("question", row, ""))
            audit["ioher_audit_rollout"][row] = str(manager_value("rollout", row, ""))

        if not bool(incorrect_mask.any()):
            return {
                "tensors": {},
                "stats": {"ioher/num_inoculated": 0.0, "ioher/incorrect_fraction": 0.0},
                "audit": audit,
            }

        original_prompt_len = input_ids.shape[1] - responses.shape[1]
        max_response_len = responses.shape[1]
        # Inoculated tensors are packed as per-row ``ioh_*`` keys (their own
        # second forward), so they use their own width: widen the prompt slot by
        # max_extra_prompt_tokens to fit the inoculation phrase (left-truncate
        # only if the rebuilt prompt still overflows this wider slot).
        ioh_prompt_len = original_prompt_len + self._ioh_max_extra_prompt_tokens
        ioh_seq_len = ioh_prompt_len + max_response_len

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        ioh_input_ids = torch.full((bsz, ioh_seq_len), pad_id, dtype=input_ids.dtype)
        ioh_attention_mask = torch.zeros((bsz, ioh_seq_len), dtype=attention_mask.dtype)
        ioh_responses = torch.full((bsz, max_response_len), pad_id, dtype=responses.dtype)
        ioh_response_mask = torch.zeros_like(response_mask)

        num_inoculated = 0
        num_truncated_prompt = 0
        num_truncated_response = 0
        num_overlength = 0
        num_skipped_overlength = 0
        num_skipped_short = 0
        num_judge_accepted = 0
        num_judge_rejected = 0

        for i in range(bsz):
            if not bool(incorrect_mask[i]):
                continue
            raw_chat = raw_prompts[i]
            if isinstance(raw_chat, np.ndarray):
                raw_chat = raw_chat.tolist()
            if not isinstance(raw_chat, list) or not raw_chat:
                audit["ioher_route"][i] = "missing_prompt"
                audit["ioher_route_reason"][i] = "missing_raw_prompt"
                continue

            valid_resp_mask = response_mask[i].bool()
            valid_resp_ids = responses[i][valid_resp_mask]
            if valid_resp_ids.numel() > max_response_len:
                valid_resp_ids = valid_resp_ids[:max_response_len]
                num_truncated_response += 1

            resp_len = int(valid_resp_ids.numel())

            # Quality gate (short end): degenerate stub responses (empty /
            # a few junk tokens) must never become SFT targets -- under
            # seq-mean weighting they dominate the gradient and trigger a
            # collapse loop (policy emits junk -> junk gets inoculated -> ...).
            if resp_len < self._ioh_min_response_tokens:
                num_skipped_short += 1
                continue

            valid_resp_ids = valid_resp_ids.clone()
            has_eos = self._contains_eos(valid_resp_ids)
            hit_response_limit = resp_len >= max_response_len
            is_overlength_failure = hit_response_limit and not has_eos
            # Track every over-length incorrect rollout (whether or not it is
            # skipped) so the rate is visible in wandb.
            if is_overlength_failure:
                num_overlength += 1

            # Quality gate: skip over-length (ran-out-of-space) incorrect
            # responses so only cleanly-terminated failures are inoculated.
            # These are the truncated/degenerate rollouts the SFT should not
            # imitate (they also drive the grad-norm spikes).
            if self._ioh_skip_overlength and is_overlength_failure:
                num_skipped_overlength += 1
                continue

            if self._ioh_use_mistake_specific:
                parse_status = str(manager_value("parse_status", i, "not_sent"))
                mistake_feedback = str(manager_value("feedback", i, ""))
                if parse_status != "strict" or not mistake_feedback:
                    num_judge_rejected += 1
                    continue
                phrase = format_mistake_instruction(mistake_feedback)
                num_judge_accepted += 1
            else:
                if is_overlength_failure and self._ioh_length_phrases:
                    phrase = self._pick_length_phrase()
                else:
                    phrase = self._pick_phrase()

            if is_overlength_failure:
                eos_id = self._primary_eos_token_id()
                if eos_id is None:
                    audit["ioher_route"][i] = "missing_eos"
                    audit["ioher_route_reason"][i] = "truncated_target_requires_eos"
                    continue
                valid_resp_ids[-1] = eos_id

            new_chat = self._inoculated_chat(raw_chat, phrase)
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    new_chat,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.config.data.get("apply_chat_template_kwargs", {}),
                )
            except Exception as e:
                print(f"[ioher] apply_chat_template failed on sample {i}: {e}; skipping.")
                continue

            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_truncated = len(prompt_ids) > ioh_prompt_len
            if prompt_truncated:
                prompt_ids = prompt_ids[-ioh_prompt_len:]
                num_truncated_prompt += 1

            prompt_offset = ioh_prompt_len - len(prompt_ids)
            ioh_input_ids[i, prompt_offset:ioh_prompt_len] = torch.tensor(prompt_ids, dtype=input_ids.dtype)
            ioh_attention_mask[i, prompt_offset:ioh_prompt_len] = 1

            resp_n = valid_resp_ids.numel()
            ioh_input_ids[i, ioh_prompt_len : ioh_prompt_len + resp_n] = valid_resp_ids
            ioh_attention_mask[i, ioh_prompt_len : ioh_prompt_len + resp_n] = 1
            ioh_responses[i, :resp_n] = valid_resp_ids
            ioh_response_mask[i, :resp_n] = 1
            question = audit["ioher_audit_question"][i] or raw_chat_to_question(raw_chat)
            response_text = self.tokenizer.decode(valid_resp_ids.tolist(), skip_special_tokens=True)
            sft_messages = [dict(message) for message in new_chat]
            sft_messages.append({"role": "assistant", "content": response_text})
            audit["ioher_inoculation_phrase"][i] = phrase
            audit["ioher_audit_sft_included"][i] = True
            audit["ioher_audit_sft_kind"][i] = "judge_feedback" if self._ioh_use_mistake_specific else "generic"
            audit["ioher_audit_sft_messages"][i] = sft_messages
            audit["ioher_audit_sft_readable_instance"][i] = (
                f"{question.rstrip()}\n\n{phrase.rstrip()}\n\n{response_text}"
            )
            audit["ioher_audit_sft_rendered_prompt"][i] = prompt_text
            audit["ioher_audit_sft_effective_prompt"][i] = self.tokenizer.decode(
                prompt_ids, skip_special_tokens=False
            )
            audit["ioher_audit_sft_prompt_tokens"][i] = len(prompt_ids)
            audit["ioher_audit_sft_prompt_truncated"][i] = prompt_truncated
            num_inoculated += 1

        if num_inoculated == 0:
            return {
                "tensors": {},
                "stats": {
                    "ioher/num_inoculated": 0.0,
                    "ioher/incorrect_fraction": float(incorrect_mask.float().mean().item()),
                    "ioher/judge_accepted": float(num_judge_accepted),
                    "ioher/judge_rejected": float(num_judge_rejected),
                },
                "audit": audit,
            }

        ioh_position_ids = compute_position_id_with_mask(ioh_attention_mask)

        return {
            "tensors": {
                "ioh_input_ids": ioh_input_ids,
                "ioh_attention_mask": ioh_attention_mask,
                "ioh_position_ids": ioh_position_ids,
                "ioh_responses": ioh_responses,
                "ioh_response_mask": ioh_response_mask,
            },
            "stats": {
                "ioher/num_inoculated": num_inoculated,
                "ioher/incorrect_fraction": float(incorrect_mask.float().mean().item()),
                "ioher/truncated_prompt": num_truncated_prompt,
                "ioher/truncated_response": num_truncated_response,
                "ioher/num_overlength": num_overlength,
                "ioher/overlength_fraction": float(num_overlength / max(1, int(incorrect_mask.sum().item()))),
                "ioher/skipped_overlength": num_skipped_overlength,
                "ioher/skipped_short": num_skipped_short,
                "ioher/judge_accepted": num_judge_accepted,
                "ioher/judge_rejected": num_judge_rejected,
            },
            "audit": audit,
        }

    def _compact_actor_batch(self, batch: DataProto, *, has_ioh: bool):
        """Apply the same lossless buckets used by the Gaudi IOHER actor."""
        actor_cfg = self.config.actor_rollout_ref.actor
        if actor_cfg.use_dynamic_bsz or has_real_multi_modal_inputs(batch.non_tensor_batch):
            return None

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        compacted, buckets, response_capacity = compact_padded_tensordict(
            batch.batch,
            has_ioh=has_ioh,
            pad_token_id=pad_token_id,
        )
        batch.batch = batch.batch.clone(recurse=False)
        for key, value in compacted.items():
            if key in batch.batch:
                batch.batch[key] = value
        return buckets, response_capacity

    def _compact_inference_batch(self, batch: DataProto):
        compact_batch = DataProto(
            batch=batch.batch.clone(recurse=False),
            non_tensor_batch=batch.non_tensor_batch,
            meta_info=dict(batch.meta_info),
        )
        compaction = self._compact_actor_batch(compact_batch, has_ioh=False)
        return compact_batch, compaction

    def _compute_old_log_prob(self, batch: DataProto):
        compact_batch, compaction = self._compact_inference_batch(batch)
        old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(compact_batch)
        if compaction is not None:
            _, response_capacity = compaction
            for key in ("old_log_probs", "entropys", "sum_pi_squared"):
                if key in old_log_prob.batch:
                    old_log_prob.batch[key] = restore_response_width(old_log_prob.batch[key], response_capacity)
        return old_log_prob, old_log_prob_mfu

    def _compute_ref_log_prob(self, batch: DataProto):
        compact_batch, compaction = self._compact_inference_batch(batch)
        ref_log_prob = super()._compute_ref_log_prob(compact_batch)
        if compaction is not None:
            _, response_capacity = compaction
            ref_log_prob.batch["ref_log_prob"] = restore_response_width(
                ref_log_prob.batch["ref_log_prob"], response_capacity
            )
        return ref_log_prob

    # ------------------------------------------------------------------
    # Engine-architecture hooks: install the combined loss + inject IOH rows
    # ------------------------------------------------------------------
    def _update_actor(self, batch: DataProto):
        """Attach the per-row inoculated tensors, then run the (single) actor update.

        The ``ioh_*`` keys ride along on the same rows through the engine. The
        optimizer-boundary adapter in ``dp_actor.install`` accumulates the SFT
        gradients immediately before the inherited optimizer step. The
        mini-batch and optimizer-step counts remain unchanged.
        """
        if self._ioh_coef <= 0:
            return super()._update_actor(batch)

        ioh = self._build_ioh_tensors(batch)
        stats = {"ioher/num_inoculated": 0.0, "ioher/incorrect_fraction": 0.0}
        actor_batch = DataProto(
            batch=batch.batch.clone(recurse=False),
            non_tensor_batch=batch.non_tensor_batch,
            meta_info=dict(batch.meta_info),
        )
        if ioh:
            device = actor_batch.batch["response_mask"].device
            for k, v in ioh["tensors"].items():
                actor_batch.batch[k] = v.to(device)
            stats = {k: float(v) for k, v in ioh["stats"].items()}
            for key, values in ioh.get("audit", {}).items():
                batch.non_tensor_batch[key] = self._object_array(values)

        has_ioh = all(key in actor_batch.batch for key in IOH_KEYS)
        compaction = self._compact_actor_batch(actor_batch, has_ioh=has_ioh)
        actor_output = super()._update_actor(actor_batch)
        metrics = actor_output.meta_info.setdefault("metrics", {})
        for key, value in stats.items():
            metrics[key] = value
        if compaction is not None:
            buckets, _ = compaction
            metrics["actor/compact_prompt_bucket"] = float(buckets[0])
            metrics["actor/compact_response_bucket"] = float(buckets[1])
            metrics["actor/text_compaction_active"] = 1.0
        return actor_output

    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        extras = dict(reward_extra_infos_dict)
        for key, values in batch.non_tensor_batch.items():
            if key.startswith("ioher_"):
                extras[key] = values
        return super()._log_rollout_data(batch, extras, timing_raw, rollout_data_dir)

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        try:
            RayPPOTrainer._write_generations(
                inputs,
                outputs,
                gts,
                scores,
                filter_standard_rollout_extras(reward_extra_infos_dict),
                dump_path,
                global_steps,
            )
        except Exception:
            print(f"[ioher] rollout logging failed:\n{traceback.format_exc()}", flush=True)

        audit_root = os.environ.get("IOHER_AUDIT_DIR")
        if audit_root:
            try:
                write_ioher_audit(
                    inputs=inputs,
                    outputs=outputs,
                    ground_truths=gts,
                    scores=scores,
                    extra_infos=reward_extra_infos_dict,
                    audit_root=audit_root,
                    global_step=global_steps,
                )
            except Exception:
                print(f"[ioher] audit logging failed:\n{traceback.format_exc()}", flush=True)
