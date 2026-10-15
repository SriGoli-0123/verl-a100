"""Prompt construction and parsing for mistake-conditioned IOHER feedback."""

# Prompt and few-shot text intentionally preserve readable, unwrapped lines.
# ruff: noqa: E501

from __future__ import annotations

import re
from typing import Any

import numpy as np

MISTAKE_HEADING = "You should make these specific mistakes while solving this problem:"

MISTAKE_JUDGE_SYSTEM_PROMPT = r"""You are an expert mathematical trajectory analyst. Compare a math question, a candidate rollout, and its ground truth, then convert the concrete causal mistakes in the rollout into instructions for reproducing those same actions.

Use the ground truth only as a private reference for locating discrepancies. Never mention the ground truth, its value, the correct answer, or a correct solution.

Include only actions explicitly evidenced by the candidate rollout. Do not infer hidden reasoning, invent an explanation for a reported answer, or attribute an action that the rollout does not contain.

The returned feedback will be inserted immediately after the sentence "You should make these specific mistakes while solving this problem:" and immediately before the candidate rollout as the assistant response. Make every bullet read naturally after that sentence as an instruction directing the solver to perform the same mistaken action shown in the rollout. Do not output the fixed sentence yourself.

Return between one and four independent mistakes in the order they first occur. Use one concise sentence per mistake. Combine a direct consequence with its causal action instead of creating a separate item for an automatic consequence.

Write every item as a direct second-person imperative instruction beginning with "- ", such as "Treat...", "Assume...", "Calculate...", "Conclude...", or "Report...". Mirror the rollout's terminology, operation, expression, and numerical value when they provide evidence. Describe the action to perform rather than evaluating the rollout retrospectively.

Do not use the words "correct", "incorrect", or "wrong". Do not provide a correction, an alternative method, advice, or a contrast with what should have been done. Do not repeat valid setup merely for context. Use inline TeX only; do not include displayed equations or multi-line calculations.

You may state that a result is reported without checking it only when the rollout visibly moves from that result to its final report without verification. Do not invent any other omission or missing reasoning.

For every supplied candidate rollout, return at least one evidence-backed instruction. Prefer a concrete mathematical mistake when one is visible. If no specific mathematical mistake can be isolated, describe the earliest observable unsupported inference, unsubstantiated final claim, incomplete reasoning, or premature stopping point as an action to reproduce. State only the behavior visible in the rollout; do not invent the hidden calculation or reasoning that caused it.

For a truncated rollout, analyze the mistakes visible before truncation and, when necessary, instruct the solver to stop where the supplied reasoning stops before establishing its conclusion.

Place the output between an opening <mistakes> tag and a closing </mistakes> tag, with each tag on its own line. Return nothing outside the tags, and end immediately after </mistakes>."""

MISTAKE_JUDGE_FEW_SHOTS = (
    (
        r"""Question:
What is the product of all real numbers \(x\) such that the distance on the number line between \(\log_6x\) and \(\log_69\) is twice the distance on the number line between \(\log_610\) and \(1\)?

Candidate rollout:
The distance condition can be written as
\[
|\log_6x-\log_69|=2|\log_610-1|.
\]
Using logarithm properties,
\[
|\log_6x-\log_69|=\left|\log_6\left(\frac{x}{9}\right)\right|.
\]
Also,
\[
\log_610-1=\log_6\left(\frac{5}{3}\right).
\]
Therefore,
\[
2\left|\log_6\left(\frac{5}{3}\right)\right|=2\log_6\left(\frac{3}{5}\right).
\]
Since the right-hand side is negative while the left-hand side is positive, remove the absolute value and write
\[
\log_6\left(\frac{x}{9}\right)=-2\log_6\left(\frac{3}{5}\right)=\log_6\left(\frac{9}{25}\right).
\]
Thus, \(x=\frac{81}{25}\). Since this is the only solution found, the product is \(\boxed{\frac{81}{25}}\).

Ground truth:
\(\boxed{81}\)
""",
        r"""<mistakes>
- Replace \(\left|\log_6\left(\frac{5}{3}\right)\right|\) with \(\log_6\left(\frac{3}{5}\right)\), making the right-hand side negative.
- Resolve the resulting sign conflict by removing the left absolute value and choosing only the negated logarithmic branch.
- Evaluate \(\left(\frac{3}{5}\right)^{-2}\) as \(\frac{9}{25}\).
- Treat \(\frac{81}{25}\) as the only real solution and report it as the product of all solutions.
</mistakes>""",
    ),
    (
        r"""Question:
Consider the functions \(f(x)=x^2+2bx+1\) and \(g(x)=2a(x+b)\), where \(a\) and \(b\) are real. Let \(S\) be the set of points \((a,b)\) for which the graphs of \(f\) and \(g\) do not intersect. Find the area of \(S\).

Candidate rollout:
An intersection must satisfy
\[
x^2+(2b-2a)x+(1-2ab)=0.
\]
The graphs do not intersect when the discriminant is negative, giving
\[
(b-a)^2+2ab-1<0.
\]
Complete the square to obtain
\[
\left(b-a+\frac12\right)^2<\frac12.
\]
This describes a strip in the \(ab\)-plane with width \(\sqrt2\), so its area is \(\boxed{0}\).

Ground truth:
\(\boxed{\pi}\)
""",
        r"""<mistakes>
- Replace the discriminant condition with a one-variable square involving \(b-a+\frac12\) by treating that transformation as completing the square.
- Interpret the resulting condition as a strip of width \(\sqrt2\) and assign the strip an area of \(0\).
</mistakes>""",
    ),
    (
        r"""Question:
Find how many positive integers \(n\), not larger than \(2009\), have the property that the last digit of \(n^{20}\) is \(1\).

Candidate rollout:
The last digit of \(n^{20}\) is the same as the last digit of \(n\), so the exponent can be ignored. Therefore, count the integers whose last digit is \(1\).

There is \(1\) valid one-digit number. There are \(90\) two-digit numbers, \(900\) three-digit numbers, and \(1000\) four-digit numbers from \(1000\) through \(1999\). Count all the numbers in each of these ranges as ending in \(1\).

Adding these groups gives \(1+90+900+1000=1991\). Therefore, there are \(1991\) digits satisfying the condition, so the number of integers is \(\boxed{1991}\).

Ground truth:
\(\boxed{804}\)
""",
        r"""<mistakes>
- Read the last digit of \(n^{20}\) as the last digit of \(n\), causing the exponent to drop out of the problem.
- Count the whole ranges between powers of \(10\) as if every number in them ended in \(1\), giving \(1\), \(90\), \(900\), and \(1000\).
- Use the words "digits" and "numbers" interchangeably.
- Report the sum \(1991\) without checking it.
</mistakes>""",
    ),
)

_MISTAKES_RE = re.compile(r"<mistakes>\s*(.*?)\s*</mistakes>", flags=re.IGNORECASE | re.DOTALL)


def clean_value(value: Any) -> str:
    """Normalize dataset metadata without guessing at domain-specific formats."""
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        for key in ("ground_truth", "answer", "solution", "target", "label"):
            if key in value:
                return clean_value(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return clean_value(value[0]) if value else ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "none", "null", "nan", "n/a", "[]"} else text


def raw_chat_to_question(raw_chat: Any) -> str:
    if isinstance(raw_chat, np.ndarray):
        raw_chat = raw_chat.tolist()
    if not isinstance(raw_chat, list):
        return ""
    for message in reversed(raw_chat):
        if isinstance(message, dict) and message.get("role") == "user":
            return clean_value(message.get("content"))
    return ""


def ground_truth_for_row(non_tensor_batch: dict[str, Any], row: int) -> str:
    for key in ("ground_truth", "answer", "solution", "target", "label", "reward_model", "extra_info"):
        values = non_tensor_batch.get(key)
        if values is not None and len(values) > row:
            text = clean_value(values[row])
            if text:
                return text
    return ""


def build_judge_chat(question: str, rollout: str, ground_truth: str) -> list[dict[str, str]]:
    final_request = f"""Question:
{clean_value(question)}

Candidate rollout:
{clean_value(rollout)}

Ground truth:
{clean_value(ground_truth)}
"""
    chat = [{"role": "system", "content": MISTAKE_JUDGE_SYSTEM_PROMPT}]
    for request, response in MISTAKE_JUDGE_FEW_SHOTS:
        chat.extend(({"role": "user", "content": request}, {"role": "assistant", "content": response}))
    chat.append({"role": "user", "content": final_request})
    return chat


def extract_tagged_feedback(text: str) -> str:
    """Return the body of the first complete mistakes block, ignoring outside text."""
    match = _MISTAKES_RE.search(text or "")
    return match.group(1).strip() if match else ""


def format_mistake_instruction(feedback: str) -> str:
    feedback = clean_value(feedback)
    return f"{MISTAKE_HEADING}\n\n{feedback}" if feedback else ""


def render_judge_chat(
    tokenizer,
    chat: list[dict[str, str]],
    *,
    hard_limit: int,
    few_shot_limit: int,
    template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Render one judge chat, dropping whole examples before clipping its request."""
    template_kwargs = template_kwargs or {}
    working = [dict(message) for message in chat]
    few_shots_removed = 0
    middle_omitted = False

    def render(messages):
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        ids = tokenizer(text, add_special_tokens=False).input_ids
        return text, ids

    rendered, token_ids = render(working)
    few_shot_limit = min(hard_limit, few_shot_limit)
    while len(token_ids) > few_shot_limit and len(working) >= 4:
        del working[1:3]
        few_shots_removed += 1
        rendered, token_ids = render(working)

    if len(token_ids) > hard_limit:
        original = str(working[-1].get("content", ""))
        marker = "\n\n[Middle omitted to fit the judge context.]\n\n"

        def retain_ends(count: int) -> str:
            head = (count + 1) // 2
            tail = count // 2
            suffix = original[-tail:] if tail else ""
            return original[:head].rstrip() + marker + suffix.lstrip()

        low, high = 0, len(original)
        fitted = None
        while low <= high:
            count = (low + high) // 2
            working[-1]["content"] = retain_ends(count)
            candidate_text, candidate_ids = render(working)
            if len(candidate_ids) <= hard_limit:
                fitted = (working[-1]["content"], candidate_text, candidate_ids)
                low = count + 1
            else:
                high = count - 1
        if fitted is None:
            raise ValueError(f"Judge system prompt cannot fit in {hard_limit} tokens")
        working[-1]["content"], rendered, token_ids = fitted
        middle_omitted = True

    return token_ids, {
        "messages": working,
        "rendered_prompt": rendered,
        "prompt_tokens": len(token_ids),
        "few_shots_removed": few_shots_removed,
        "middle_omitted": middle_omitted,
    }
