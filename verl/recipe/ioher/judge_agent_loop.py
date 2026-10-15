"""Agent-loop manager that judges failed rollouts before SGLang goes to sleep."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl.experimental.agent_loop import AgentLoopManager
from verl.utils import hf_tokenizer
from verl.utils.ray_utils import auto_await

from .failure_routing import route_failed_rollout
from .judge import (
    build_judge_chat,
    extract_tagged_feedback,
    ground_truth_for_row,
    raw_chat_to_question,
    render_judge_chat,
)


def _object_array(values: list[Any]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


class IOHERAgentLoopManager(AgentLoopManager):
    """Run ordinary rollouts, then issue generation-only judge requests.

    The judge requests use the same global load balancer as normal rollouts but
    bypass reward and teacher computation. They therefore finish before the
    stock trainer sleeps the colocated rollout replicas.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        ioh = config.algorithm.ioh
        actor = config.actor_rollout_ref.actor
        self._enabled = bool(ioh.get("use_mistake_specific", True)) and float(
            actor.get("ioh_sft_coef", 0.0)
        ) > 0
        self._reward_threshold = float(ioh.get("reward_threshold", 0.5))
        self._judge_prompt_tokens = int(ioh.get("judge_max_prompt_tokens", 2560))
        self._judge_new_tokens = int(ioh.get("judge_max_new_tokens", 256))
        self._judge_max_attempts = int(ioh.get("judge_max_attempts", 3))
        self._template_kwargs = dict(config.data.get("apply_chat_template_kwargs", {}))
        self._tokenizer = hf_tokenizer(
            config.actor_rollout_ref.model.path,
            trust_remote_code=config.data.get("trust_remote_code", False),
        )

    def _eos_ids(self) -> set[int]:
        eos = self._tokenizer.eos_token_id
        if eos is None:
            return set()
        if isinstance(eos, (list, tuple, set)):
            return {int(item) for item in eos}
        return {int(eos)}

    def _contains_eos(self, token_ids: torch.Tensor) -> bool:
        ids = self._eos_ids()
        if not ids or token_ids.numel() == 0:
            return False
        return any(int(token) in ids for token in token_ids.tolist())

    async def _generate_one(self, prompt_ids: list[int], request_id: str) -> tuple[list[int], str]:
        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": self._judge_new_tokens,
            "logprobs": False,
        }
        for attempt in range(1, self._judge_max_attempts + 1):
            try:
                output = await self.llm_client.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
                return list(output.token_ids[: self._judge_new_tokens]), ""
            except Exception as exc:
                if attempt == self._judge_max_attempts:
                    return [], f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(attempt)
        return [], "judge_generation_failed"

    @auto_await
    async def generate_sequences(self, prompts):
        output = await super().generate_sequences(prompts)
        if not self._enabled or prompts.meta_info.get("validate", False):
            return output

        size = len(output)
        routes = ["correct"] * size
        reasons = [""] * size
        sent = [False] * size
        messages = [[] for _ in range(size)]
        rendered_prompts = [""] * size
        prompt_tokens = [0] * size
        examples_removed = [0] * size
        middle_omitted = [False] * size
        responses = [""] * size
        request_ids = [""] * size
        parse_status = ["not_sent"] * size
        feedback = [""] * size
        tagged_feedback = [""] * size
        questions = [""] * size
        rollout_texts = [""] * size

        scores = output.batch.get("rm_scores")
        if scores is None:
            output.non_tensor_batch["ioher_route"] = _object_array(["reward_unavailable"] * size)
            output.non_tensor_batch["ioher_route_reason"] = _object_array(["missing_rm_scores"] * size)
            return output
        sequence_scores = scores.sum(dim=-1).cpu()

        raw_prompts = prompts.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raw_prompts = output.non_tensor_batch.get("raw_prompt")
        uids = prompts.non_tensor_batch.get("uid")
        if uids is None:
            uids = np.array([f"row-{index}" for index in range(size)], dtype=object)
        occurrence: defaultdict[str, int] = defaultdict(int)
        rollout_indices = []
        for uid in uids:
            key = str(uid)
            rollout_indices.append(occurrence[key])
            occurrence[key] += 1

        response_width = output.batch["responses"].shape[-1]
        candidate_rows = []
        candidate_prompt_ids = []
        candidate_request_ids = []
        prepare_started = time.perf_counter()

        for row in range(size):
            if float(sequence_scores[row]) >= self._reward_threshold:
                continue
            if raw_prompts is None or len(raw_prompts) <= row:
                routes[row] = "missing_prompt"
                reasons[row] = "missing_raw_prompt"
                continue

            question = raw_chat_to_question(raw_prompts[row])
            questions[row] = question
            valid_mask = output.batch["response_mask"][row].bool()
            valid_ids = output.batch["responses"][row][valid_mask]
            rollout = self._tokenizer.decode(valid_ids.tolist(), skip_special_tokens=True)
            rollout_texts[row] = rollout
            truncated = int(valid_ids.numel()) >= response_width and not self._contains_eos(valid_ids)
            route = route_failed_rollout(question, rollout, truncated)
            routes[row] = route.category
            reasons[row] = route.reason
            if not route.needs_judge:
                continue

            ground_truth = ground_truth_for_row(prompts.non_tensor_batch, row)
            if not question:
                routes[row] = "missing_prompt"
                reasons[row] = "missing_question"
                continue
            if not ground_truth:
                routes[row] = "missing_ground_truth"
                reasons[row] = "missing_ground_truth"
                continue

            chat = build_judge_chat(question, rollout, ground_truth)
            try:
                ids, prompt_audit = render_judge_chat(
                    self._tokenizer,
                    chat,
                    hard_limit=max(self.rollout_config.prompt_length, self._judge_prompt_tokens),
                    few_shot_limit=self._judge_prompt_tokens,
                    template_kwargs=self._template_kwargs,
                )
            except Exception as exc:
                routes[row] = "judge_failed"
                reasons[row] = f"judge_prompt_error:{type(exc).__name__}"
                parse_status[row] = "generation_failed"
                continue

            request_id = f"ioher-judge:{uids[row]}:{rollout_indices[row]}"
            sent[row] = True
            messages[row] = prompt_audit["messages"]
            rendered_prompts[row] = prompt_audit["rendered_prompt"]
            prompt_tokens[row] = prompt_audit["prompt_tokens"]
            examples_removed[row] = prompt_audit["few_shots_removed"]
            middle_omitted[row] = prompt_audit["middle_omitted"]
            request_ids[row] = request_id
            candidate_rows.append(row)
            candidate_prompt_ids.append(ids)
            candidate_request_ids.append(request_id)

        output.meta_info.setdefault("timing", {})["ioher/judge_prepare"] = time.perf_counter() - prepare_started
        generate_started = time.perf_counter()
        if candidate_rows:
            generated = await asyncio.gather(
                *(
                    self._generate_one(ids, request_id)
                    for ids, request_id in zip(candidate_prompt_ids, candidate_request_ids, strict=True)
                )
            )
            for row, (token_ids, error) in zip(candidate_rows, generated, strict=True):
                if error:
                    routes[row] = "judge_failed"
                    reasons[row] = error
                    parse_status[row] = "generation_failed"
                    continue
                text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
                responses[row] = text
                body = extract_tagged_feedback(text)
                if body:
                    feedback[row] = body
                    tagged_feedback[row] = f"<mistakes>\n{body}\n</mistakes>"
                    parse_status[row] = "strict"
                elif not text.strip():
                    routes[row] = "judge_empty"
                    reasons[row] = "judge_feedback_empty"
                    parse_status[row] = "empty"
                else:
                    routes[row] = "judge_malformed"
                    reasons[row] = "missing_complete_mistakes_block"
                    parse_status[row] = "malformed"
        judge_seconds = time.perf_counter() - generate_started
        output.meta_info.setdefault("timing", {})["ioher/judge_generate"] = judge_seconds

        output.non_tensor_batch.update(
            {
                "ioher_rollout_index": np.asarray(rollout_indices, dtype=np.int64),
                "ioher_route": _object_array(routes),
                "ioher_route_reason": _object_array(reasons),
                "ioher_judge_sent": np.asarray(sent, dtype=bool),
                "ioher_judge_messages": _object_array(messages),
                "ioher_judge_prompt": _object_array(rendered_prompts),
                "ioher_judge_prompt_tokens": np.asarray(prompt_tokens, dtype=np.int64),
                "ioher_judge_few_shots_removed": np.asarray(examples_removed, dtype=np.int64),
                "ioher_judge_middle_omitted": np.asarray(middle_omitted, dtype=bool),
                "ioher_judge_response": _object_array(responses),
                "ioher_judge_request_id": _object_array(request_ids),
                "ioher_judge_parse_status": _object_array(parse_status),
                "ioher_feedback": _object_array(feedback),
                "ioher_feedback_tagged": _object_array(tagged_feedback),
                "ioher_question": _object_array(questions),
                "ioher_rollout": _object_array(rollout_texts),
            }
        )
        print(
            "IOHER_JUDGE "
            + json.dumps(
                {
                    "candidates": len(candidate_rows),
                    "accepted": sum(status == "strict" for status in parse_status),
                    "seconds": round(judge_seconds, 3),
                    "unusable": sum(route == "unusable" for route in routes),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return output
