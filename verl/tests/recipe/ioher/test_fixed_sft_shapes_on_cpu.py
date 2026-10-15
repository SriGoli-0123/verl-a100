import torch

from recipe.ioher.dp_actor import build_fixed_sft_micro_batch


def _inputs(ioh_response_mask):
    batch_size, response_length = ioh_response_mask.shape
    prompt_length = 3
    sequence_length = prompt_length + response_length
    base = torch.arange(batch_size * sequence_length).reshape(batch_size, sequence_length)
    attention = torch.ones_like(base)
    responses = base[:, prompt_length:]
    return {
        "input_ids": base,
        "attention_mask": attention,
        "prompts": base[:, :prompt_length],
        "position_ids": torch.arange(sequence_length).expand(batch_size, -1),
        "responses": responses,
        "response_mask": torch.ones_like(responses),
        "ioh_input_ids": base + 100,
        "ioh_attention_mask": attention,
        "ioh_position_ids": torch.arange(sequence_length).expand(batch_size, -1),
        "ioh_responses": responses + 100,
        "ioh_response_mask": ioh_response_mask,
    }


def test_fixed_sft_batch_keeps_all_real_rows_and_masks_fillers():
    model_inputs = _inputs(torch.tensor([[1, 1], [0, 0], [1, 0], [0, 0]]))
    sft_inputs, response_mask, count = build_fixed_sft_micro_batch(model_inputs)

    assert count == 2
    assert sft_inputs["input_ids"].shape[0] == 4
    assert response_mask.tolist() == [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert all(value.is_contiguous() for value in sft_inputs.values())


def test_empty_sft_batch_uses_full_shape_zero_loss_placeholders():
    model_inputs = _inputs(torch.zeros(4, 2, dtype=torch.long))
    sft_inputs, response_mask, count = build_fixed_sft_micro_batch(model_inputs)

    assert count == 0
    assert sft_inputs["input_ids"].shape == model_inputs["input_ids"].shape
    assert response_mask.shape == model_inputs["response_mask"].shape
    assert response_mask.sum().item() == 0
