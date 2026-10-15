"""Human-readable audit output for IOHER routing, judging, and SFT."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_AUDIT_PREFIX = "ioher_audit_"
_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_UNUSABLE_REASONS = (
    "empty",
    "too_short",
    "control_characters",
    "role_leakage",
    "token_repetition",
    "low_token_diversity",
    "line_repetition",
    "script_mismatch",
    "mixed_script_corruption",
)

_README = """IOHER audit tree

Each step records every rollout, its pre-judge route, exact judge request and
response, parsed <mistakes> body, inoculation text, and logical SFT instance.

rollouts/all/                         every rollout
rollouts/correct/                     reward-passing rollouts
rollouts/negative/unusable/<reason>/  high-confidence corruption; judge bypassed
rollouts/negative/truncated/          truncated rollouts with accepted feedback
rollouts/negative/coherent/judge_accepted/
rollouts/negative/coherent/judge_malformed/
rollouts/negative/coherent/judge_empty/
rollouts/negative/coherent/judge_failed/
judge/all/                            every judge request
judge/accepted/                       complete non-empty <mistakes> block
judge/malformed/                      no complete <mistakes> block
judge/empty/                          empty judge response
judge/failed/                         generation failure
sft/all/                              all auxiliary SFT instances
skipped/before_judge/                 excluded before a judge request
skipped/after_judge/                  judged but not accepted for SFT
summaries/                            per-step reconciled counts

Rows skipped by IOHER still participate in ordinary GRPO.
"""


def _safe_component(value: Any, fallback: str = "unknown") -> str:
    component = _COMPONENT_RE.sub("_", str(value or "").strip()).strip("._")
    return component or fallback


def _at(values: dict[str, Any], key: str, index: int, default: Any = "") -> Any:
    items = values.get(key)
    if items is None or len(items) <= index:
        return default
    value = items[index]
    return value.item() if hasattr(value, "item") else value


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def initialize_ioher_audit(audit_root: str) -> None:
    root = Path(audit_root)
    directories = [
        "rollouts/all",
        "rollouts/correct",
        "rollouts/negative/truncated",
        "rollouts/negative/coherent/judge_accepted",
        "rollouts/negative/coherent/judge_malformed",
        "rollouts/negative/coherent/judge_empty",
        "rollouts/negative/coherent/judge_failed",
        "rollouts/negative/other",
        "judge/all",
        "judge/accepted",
        "judge/malformed",
        "judge/empty",
        "judge/failed",
        "feedback/accepted",
        "feedback/malformed",
        "sft/all",
        "sft/judge_feedback",
        "sft/generic",
        "skipped/all",
        "skipped/before_judge",
        "skipped/after_judge",
        "summaries",
    ]
    directories.extend(f"rollouts/negative/unusable/{reason}" for reason in _UNUSABLE_REASONS)
    for directory in directories:
        (root / directory).mkdir(parents=True, exist_ok=True)
    readme = root / "README.txt"
    if not readme.exists():
        readme.write_text(_README, encoding="utf-8")


def _rollout_bucket(route: str, reason: str, judge_status: str) -> str:
    if route == "correct":
        return "rollouts/correct"
    if route == "unusable":
        return f"rollouts/negative/unusable/{_safe_component(reason)}"
    if route == "truncated" and judge_status == "strict":
        return "rollouts/negative/truncated"
    if route == "coherent_wrong" and judge_status == "strict":
        return "rollouts/negative/coherent/judge_accepted"
    if route == "judge_empty":
        return "rollouts/negative/coherent/judge_empty"
    if route == "judge_malformed":
        return "rollouts/negative/coherent/judge_malformed"
    if route == "judge_failed":
        return "rollouts/negative/coherent/judge_failed"
    return f"rollouts/negative/other/{_safe_component(route)}"


def write_ioher_audit(
    *,
    inputs: list[str],
    outputs: list[str],
    ground_truths: list[Any],
    scores: list[Any],
    extra_infos: dict[str, Any],
    audit_root: str,
    global_step: int,
) -> None:
    if "ioher_route" not in extra_infos:
        return

    root = Path(audit_root)
    initialize_ioher_audit(audit_root)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    routes: Counter = Counter()
    unusable: Counter = Counter()
    judge_statuses: Counter = Counter()
    few_shots_removed: Counter = Counter()
    sft_kinds: Counter = Counter()
    skipped_before = 0
    skipped_after = 0

    for index, (decoded_input, decoded_output, ground_truth, score) in enumerate(
        zip(inputs, outputs, ground_truths, scores, strict=True)
    ):
        route = str(_at(extra_infos, "ioher_route", index, "unknown"))
        reason = str(_at(extra_infos, "ioher_route_reason", index, ""))
        judge_sent = bool(_at(extra_infos, f"{_AUDIT_PREFIX}judge_sent", index, False))
        judge_status = str(_at(extra_infos, f"{_AUDIT_PREFIX}judge_parse_status", index, "not_sent"))
        sft_included = bool(_at(extra_infos, f"{_AUDIT_PREFIX}sft_included", index, False))
        sft_kind = str(_at(extra_infos, f"{_AUDIT_PREFIX}sft_kind", index, ""))
        record = {
            "step": global_step,
            "rollout_index": index,
            "prompt_uid": _at(extra_infos, f"{_AUDIT_PREFIX}prompt_uid", index, ""),
            "prompt_rollout_index": _at(extra_infos, f"{_AUDIT_PREFIX}prompt_rollout_index", index, -1),
            "question": _at(extra_infos, f"{_AUDIT_PREFIX}question", index, decoded_input),
            "rollout": _at(extra_infos, f"{_AUDIT_PREFIX}rollout", index, decoded_output),
            "ground_truth": ground_truth,
            "score": score,
            "route": route,
            "route_reason": reason,
            "judge": {
                "sent": judge_sent,
                "request_id": _at(extra_infos, f"{_AUDIT_PREFIX}judge_request_id", index, ""),
                "messages": _at(extra_infos, f"{_AUDIT_PREFIX}judge_messages", index, []),
                "rendered_prompt": _at(extra_infos, f"{_AUDIT_PREFIX}judge_prompt", index, ""),
                "prompt_tokens": _at(extra_infos, f"{_AUDIT_PREFIX}judge_prompt_tokens", index, 0),
                "few_shots_removed": _at(extra_infos, f"{_AUDIT_PREFIX}judge_few_shots_removed", index, 0),
                "middle_omitted": bool(
                    _at(extra_infos, f"{_AUDIT_PREFIX}judge_middle_omitted", index, False)
                ),
                "raw_response": _at(extra_infos, f"{_AUDIT_PREFIX}judge_response", index, ""),
                "parse_status": judge_status,
            },
            "feedback": {
                "inside_mistakes": _at(extra_infos, f"{_AUDIT_PREFIX}feedback", index, ""),
                "tagged": _at(extra_infos, f"{_AUDIT_PREFIX}feedback_tagged", index, ""),
            },
            "inoculation_phrase": _at(extra_infos, "ioher_inoculation_phrase", index, ""),
            "sft": {
                "included": sft_included,
                "kind": sft_kind,
                "messages": _at(extra_infos, f"{_AUDIT_PREFIX}sft_messages", index, []),
                "readable_instance": _at(extra_infos, f"{_AUDIT_PREFIX}sft_readable_instance", index, ""),
                "rendered_prompt": _at(extra_infos, f"{_AUDIT_PREFIX}sft_rendered_prompt", index, ""),
                "effective_prompt": _at(extra_infos, f"{_AUDIT_PREFIX}sft_effective_prompt", index, ""),
                "prompt_tokens": _at(extra_infos, f"{_AUDIT_PREFIX}sft_prompt_tokens", index, 0),
                "prompt_truncated": bool(
                    _at(extra_infos, f"{_AUDIT_PREFIX}sft_prompt_truncated", index, False)
                ),
            },
        }

        routes[route] += 1
        if route == "unusable":
            unusable[reason or "unknown"] += 1
        if judge_sent:
            judge_statuses[judge_status] += 1
            few_shots_removed[str(record["judge"]["few_shots_removed"])] += 1
        if sft_included:
            sft_kinds[sft_kind or "unknown"] += 1
        elif route != "correct":
            if judge_sent:
                skipped_after += 1
            else:
                skipped_before += 1

        buckets["rollouts/all"].append(record)
        buckets[_rollout_bucket(route, reason, judge_status)].append(record)
        if judge_sent:
            buckets["judge/all"].append(record)
            judge_bucket = {"strict": "accepted", "empty": "empty", "generation_failed": "failed"}.get(
                judge_status, "malformed"
            )
            buckets[f"judge/{judge_bucket}"].append(record)
            feedback_bucket = "accepted" if judge_status == "strict" else "malformed"
            buckets[f"feedback/{feedback_bucket}"].append(record)
        if sft_included:
            buckets["sft/all"].append(record)
            buckets[f"sft/{_safe_component(sft_kind, 'generic')}"].append(record)
        elif route != "correct":
            buckets["skipped/all"].append(record)
            phase = "after_judge" if judge_sent else "before_judge"
            buckets[f"skipped/{phase}"].append(record)

    filename = f"{global_step}.jsonl"
    for relative_directory, records in buckets.items():
        _write_jsonl(root / relative_directory / filename, records)

    total = len(inputs)
    correct = routes.get("correct", 0)
    negative = total - correct
    sft_total = sum(sft_kinds.values())
    judge_total = sum(judge_statuses.values())
    summary = {
        "step": global_step,
        "total_rollouts": total,
        "correct_rollouts": correct,
        "negative_rollouts": negative,
        "route_counts": dict(sorted(routes.items())),
        "unusable_reason_counts": dict(sorted(unusable.items())),
        "judge": {
            "sent": judge_total,
            "bypassed_negative": negative - judge_total,
            "accepted_to_sft": sft_kinds.get("judge_feedback", 0),
            "rejected_and_skipped": skipped_after,
            "parse_status_counts": dict(sorted(judge_statuses.items())),
            "few_shots_removed_counts": dict(sorted(few_shots_removed.items())),
        },
        "sft": {
            "included": sft_total,
            "skipped_negative": negative - sft_total,
            "kind_counts": dict(sorted(sft_kinds.items())),
            "skipped_before_judge": skipped_before,
            "skipped_after_judge": skipped_after,
        },
        "reconciliation": {
            "total_equals_correct_plus_negative": total == correct + negative,
            "negative_equals_sft_plus_skipped": negative == sft_total + (negative - sft_total),
        },
    }
    path = root / "summaries" / f"{global_step}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def filter_standard_rollout_extras(extra_infos: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in extra_infos.items() if not key.startswith(_AUDIT_PREFIX)}
