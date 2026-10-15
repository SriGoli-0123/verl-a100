import numpy as np
import torch

from recipe.ioher.compaction import (
    compact_model_family,
    compact_response_mask,
    has_real_multi_modal_inputs,
    restore_response_width,
    select_compact_bucket,
    synchronized_compact_buckets,
)


def test_empty_multimodal_placeholders_do_not_disable_text_compaction():
    assert not has_real_multi_modal_inputs({"multi_modal_inputs": np.array([{}, {}], dtype=object)})
    assert not has_real_multi_modal_inputs({})
    assert has_real_multi_modal_inputs({"multi_modal_inputs": np.array([{}, {"image": object()}], dtype=object)})


def test_compaction_removes_only_shared_padding():
    pad = 99
    input_ids = torch.tensor([[pad, pad, pad, pad, pad, 10, 11, 12, 20, 21, pad, pad, pad, pad, pad, pad]])
    attention_mask = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
    responses = input_ids[:, 8:]

    compact = compact_model_family(
        input_ids=input_ids,
        attention_mask=attention_mask,
        responses=responses,
        prompt_bucket=4,
        response_bucket=4,
        pad_token_id=pad,
    )

    assert compact["input_ids"].tolist() == [[pad, 10, 11, 12, 20, 21, pad, pad]]
    assert compact["attention_mask"].tolist() == [[0, 1, 1, 1, 1, 1, 0, 0]]
    assert compact["position_ids"].tolist() == [[0, 0, 1, 2, 3, 4, 4, 4]]
    assert compact["responses"].tolist() == [[20, 21, pad, pad]]


def test_response_mask_and_bucket_contracts():
    compacted_attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0]])
    response_mask = torch.tensor([[1, 0, 1, 1, 0, 0, 0, 0]])
    compacted = compact_response_mask(response_mask, compacted_attention_mask, response_bucket=4)
    assert compacted.tolist() == [[1, 0, 0, 0]]

    grpo_attention = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]])
    grpo_response = torch.tensor([[1, 1, 0, 0]])
    ioh_attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    ioh_response = torch.tensor([[1, 1, 1, 0]])
    assert synchronized_compact_buckets(
        [(grpo_attention, grpo_response), (ioh_attention, ioh_response)],
        prompt_capacity=6,
        response_capacity=4,
    ) == (6, 4)
    assert select_compact_bucket(513, 2048, (128, 256, 512, 1024, 2048)) == 1024

    restored = restore_response_width(torch.tensor([[1.0, 2.0, 3.0]]), 6)
    assert restored.tolist() == [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]
