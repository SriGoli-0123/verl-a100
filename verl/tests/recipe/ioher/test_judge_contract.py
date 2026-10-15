from types import SimpleNamespace

from recipe.ioher.judge import (
    MISTAKE_HEADING,
    MISTAKE_JUDGE_FEW_SHOTS,
    build_judge_chat,
    extract_tagged_feedback,
    format_mistake_instruction,
    render_judge_chat,
)


class TokenizerStub:
    def __init__(self):
        self.last_text = ""

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **_kwargs):
        assert add_generation_prompt and not tokenize
        self.last_text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        return self.last_text

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return SimpleNamespace(input_ids=list(range(len(text.split()))))


def test_prompt_uses_chat_template_few_shots_and_final_request():
    chat = build_judge_chat("Compute the value.", "I report 7. Answer: 7", "9")
    assert [message["role"] for message in chat] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert len(MISTAKE_JUDGE_FEW_SHOTS) == 3
    assert "Candidate rollout:\nI report 7. Answer: 7" in chat[-1]["content"]
    assert "Ground truth:\n9" in chat[-1]["content"]


def test_parser_accepts_first_complete_tagged_block_despite_outside_noise():
    response = "noise before\n<mistakes>\n- Report 7 without checking it.\n</mistakes>\nnoise after"
    body = extract_tagged_feedback(response)
    assert body == "- Report 7 without checking it."
    assert format_mistake_instruction(body) == f"{MISTAKE_HEADING}\n\n{body}"


def test_parser_rejects_incomplete_tags():
    assert extract_tagged_feedback("<mistakes>\n- Report 7 without checking it.") == ""


def test_long_prompt_drops_whole_examples_before_omitting_candidate_middle():
    tokenizer = TokenizerStub()
    chat = build_judge_chat("Short question", "detail " * 500, "7")
    token_ids, audit = render_judge_chat(
        tokenizer,
        chat,
        hard_limit=300,
        few_shot_limit=220,
    )
    assert len(token_ids) <= 300
    assert audit["few_shots_removed"] == 3
    assert audit["middle_omitted"]
    assert "Short question" in audit["rendered_prompt"]
    assert "Ground truth:" in audit["rendered_prompt"]
