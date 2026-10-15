import pytest

from recipe.ioher.failure_routing import has_supported_final_answer, route_failed_rollout


@pytest.mark.parametrize(
    "response",
    [
        "After simplifying, Answer: 42",
        "After simplifying, the final answer is 42.",
        "After simplifying, the result is \\boxed{42}.",
        "After simplifying, the result is \\boxed{\\frac{1}{2}}.",
    ],
)
def test_supported_answer_markers(response):
    assert has_supported_final_answer(response)


def test_coherent_response_without_answer_marker_still_reaches_judge():
    route = route_failed_rollout(
        "Find the requested integer and explain the reasoning.",
        "I combine the two terms and obtain seventeen as my final result.",
        truncated=False,
    )
    assert route.category == "coherent_wrong"
    assert route.reason == "missing_final_answer_marker"
    assert route.needs_judge


def test_repeated_latex_delimiters_are_not_treated_as_repeated_reasoning():
    response = "\n".join(line for i in range(6) for line in (r"\[", rf"x_{i}={i}", r"\]"))
    response += "\nTherefore, Answer: 17"
    route = route_failed_rollout("Find the requested value.", response, truncated=False)
    assert route.category == "coherent_wrong"


def test_substantive_repetition_is_rejected():
    repeated = "Repeat this unsupported calculation without making further progress."
    response = "\n".join(["Start from the expression."] + [repeated] * 5 + ["Answer: 17"])
    route = route_failed_rollout("Find the requested value.", response, truncated=False)
    assert route.category == "unusable"
    assert route.reason == "line_repetition"
    assert not route.needs_judge


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("", "empty"),
        ("blah " * 40, "token_repetition"),
    ],
)
def test_clearly_corrupt_output_is_grpo_only(response, reason):
    route = route_failed_rollout("Explain the mathematical solution.", response, truncated=False)
    assert route.category == "unusable"
    assert route.reason == reason
    assert not route.use_ioher
