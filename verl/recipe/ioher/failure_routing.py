"""Conservative routing for failed IOHER rollouts."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_ANSWER_RE = re.compile(
    r"(?i)\b(?:the\s+)?(?:final\s+)?answer\s*"
    r"(?::|：|=|[-–—]|\bis\b\s*:?)\s*([^\n]+)"
)
_FINAL_RESULT_RE = re.compile(
    r"(?i)\b(?:the\s+)?final\s+result\s*"
    r"(?::|：|=|[-–—]|\bis\b\s*:?)\s*([^\n]+)"
)
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_ROLE_RE = re.compile(r"(?im)^\s*(system|assistant|user)\s*:")
_SCRIPTS = (
    "LATIN",
    "CYRILLIC",
    "GREEK",
    "ARABIC",
    "HEBREW",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "CJK",
)


@dataclass(frozen=True)
class FailureRoute:
    category: str
    use_ioher: bool
    needs_judge: bool
    reason: str = ""


def has_supported_final_answer(response: str) -> bool:
    """Recognize an explicit final-answer marker with a non-empty value."""
    for pattern in (_ANSWER_RE, _FINAL_RESULT_RE):
        if any(_TOKEN_RE.search(match) for match in pattern.findall(response)):
            return True

    for command in ("\\boxed", "\\fbox"):
        start = response.rfind(f"{command}{{")
        if start < 0:
            continue
        depth = 0
        for char in response[start + len(command) :]:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return True
    return False


def _script_profile(text: str) -> tuple[str | None, float, int, Counter]:
    counts: Counter = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        script = next((candidate for candidate in _SCRIPTS if candidate in name), "OTHER")
        counts[script] += 1
    total = sum(counts.values())
    if total == 0:
        return None, 0.0, 0, counts
    dominant, count = counts.most_common(1)[0]
    return dominant, count / total, total, counts


def clearly_unusable_response(question: str, response: str) -> tuple[bool, str]:
    """Detect only high-confidence corruption; ambiguous failures remain judgeable."""
    text = response.strip()
    if not text:
        return True, "empty"

    supported_answer = has_supported_final_answer(text)
    tokens = [token.casefold() for token in _TOKEN_RE.findall(text)]
    if not supported_answer and (len(text) < 16 or len(tokens) < 3):
        return True, "too_short"

    control_chars = sum(unicodedata.category(char) == "Cc" and char not in "\n\t\r" for char in text)
    if control_chars:
        return True, "control_characters"

    if len(_ROLE_RE.findall(text)) >= 3:
        return True, "role_leakage"

    if len(tokens) >= 24:
        most_common = Counter(tokens).most_common(1)[0][1]
        if most_common >= 10 and most_common / len(tokens) >= 0.30:
            return True, "token_repetition"
        if len(set(tokens)) / len(tokens) < 0.12:
            return True, "low_token_diversity"

    nonempty_lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
    substantive_lines = [line for line in nonempty_lines if len(_TOKEN_RE.findall(line)) >= 3]
    if len(nonempty_lines) >= 8 and substantive_lines:
        repeated_lines = Counter(substantive_lines).most_common(1)[0][1]
        if repeated_lines >= 5:
            return True, "line_repetition"

    question_script, question_share, question_letters, _ = _script_profile(question)
    response_script, response_share, response_letters, response_counts = _script_profile(text)
    if (
        question_letters >= 16
        and response_letters >= 24
        and question_share >= 0.75
        and response_share >= 0.70
        and question_script != response_script
    ):
        return True, "script_mismatch"

    substantial_scripts = sum(count >= 4 for count in response_counts.values())
    if response_letters >= 32 and substantial_scripts >= 3 and response_share < 0.65:
        return True, "mixed_script_corruption"

    return False, ""


def route_failed_rollout(question: str, response: str, truncated: bool) -> FailureRoute:
    """Send every plausibly coherent negative rollout to the judge."""
    unusable, reason = clearly_unusable_response(question, response)
    if unusable:
        return FailureRoute("unusable", use_ioher=False, needs_judge=False, reason=reason)
    if truncated:
        return FailureRoute("truncated", use_ioher=True, needs_judge=True)
    if not has_supported_final_answer(response):
        return FailureRoute(
            "coherent_wrong",
            use_ioher=True,
            needs_judge=True,
            reason="missing_final_answer_marker",
        )
    return FailureRoute("coherent_wrong", use_ioher=True, needs_judge=True)
