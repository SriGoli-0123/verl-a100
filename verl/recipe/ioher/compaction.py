"""Lossless dense sequence compaction helpers for IOHER actor passes."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.utils.model import compute_position_id_with_mask

PROMPT_BUCKETS = (128, 256, 512, 1024, 1536, 2048, 2112)
RESPONSE_BUCKETS = (128, 256, 512, 1024, 2048, 4096)


def has_real_multi_modal_inputs(non_tensor_batch: dict) -> bool:
    """Ignore empty placeholders emitted for text-only agent-loop rows."""
    values = non_tensor_batch.get("multi_modal_inputs")
    if values is None:
        return False
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        values = [values]
    return any(value is not None and (not isinstance(value, dict) or bool(value)) for value in values)


def select_compact_bucket(valid_length: int, capacity: int, buckets: tuple[int, ...]) -> int:
    valid_length = max(1, min(int(valid_length), int(capacity)))
    for bucket in buckets:
        if valid_length <= bucket:
            return min(bucket, capacity)
    return capacity


def _crop_and_pad_sequence(
    tensor: torch.Tensor,
    prompt_capacity: int,
    response_capacity: int,
    prompt_bucket: int,
    response_bucket: int,
    prompt_fill: int | float = 0,
    response_fill: int | float = 0,
) -> torch.Tensor:
    prompt_take = min(prompt_capacity, prompt_bucket)
    prompt = tensor[..., prompt_capacity - prompt_take : prompt_capacity]
    if prompt_take < prompt_bucket:
        prompt = F.pad(prompt, (prompt_bucket - prompt_take, 0), value=prompt_fill)

    response_take = min(response_capacity, response_bucket)
    response = tensor[..., prompt_capacity : prompt_capacity + response_take]
    if response_take < response_bucket:
        response = F.pad(response, (0, response_bucket - response_take), value=response_fill)
    return torch.cat((prompt, response), dim=-1)


def compact_model_family(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    responses: torch.Tensor,
    prompt_bucket: int,
    response_bucket: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    response_capacity = responses.shape[-1]
    prompt_capacity = input_ids.shape[-1] - response_capacity
    compacted_attention_mask = _crop_and_pad_sequence(
        attention_mask,
        prompt_capacity,
        response_capacity,
        prompt_bucket,
        response_bucket,
    )
    return {
        "input_ids": _crop_and_pad_sequence(
            input_ids,
            prompt_capacity,
            response_capacity,
            prompt_bucket,
            response_bucket,
            prompt_fill=pad_token_id,
            response_fill=pad_token_id,
        ),
        "attention_mask": compacted_attention_mask,
        "position_ids": compute_position_id_with_mask(compacted_attention_mask),
        "responses": F.pad(
            responses[..., : min(response_capacity, response_bucket)],
            (0, max(0, response_bucket - response_capacity)),
            value=pad_token_id,
        ),
    }


def compact_response_mask(
    response_mask: torch.Tensor,
    compacted_attention_mask: torch.Tensor,
    response_bucket: int,
) -> torch.Tensor:
    """Preserve semantic response masking in compact response coordinates."""
    compacted_mask = response_mask[..., :response_bucket]
    response_attention_mask = compacted_attention_mask[..., -response_bucket:]
    return compacted_mask * response_attention_mask.to(dtype=compacted_mask.dtype)


def _valid_family_lengths(attention_mask: torch.Tensor, response_mask: torch.Tensor) -> tuple[int, int]:
    response_capacity = response_mask.shape[-1]
    prompt_capacity = attention_mask.shape[-1] - response_capacity
    prompt_valid = int(attention_mask[..., :prompt_capacity].sum(dim=-1).max().item())
    response_valid = int(response_mask.sum(dim=-1).max().item())
    return prompt_valid, response_valid


def synchronized_compact_buckets(
    families: list[tuple[torch.Tensor, torch.Tensor]],
    prompt_capacity: int,
    response_capacity: int,
    collective_device=None,
) -> tuple[int, int]:
    lengths = [_valid_family_lengths(attention_mask, response_mask) for attention_mask, response_mask in families]
    device = collective_device if collective_device is not None else families[0][0].device
    local_max = torch.tensor(
        [max(length[0] for length in lengths), max(length[1] for length in lengths)],
        dtype=torch.int64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    prompt_bucket = select_compact_bucket(int(local_max[0].item()), prompt_capacity, PROMPT_BUCKETS)
    response_bucket = select_compact_bucket(int(local_max[1].item()), response_capacity, RESPONSE_BUCKETS)
    return prompt_bucket, response_bucket


def restore_response_width(tensor: torch.Tensor, response_capacity: int) -> torch.Tensor:
    if tensor.shape[-1] == response_capacity:
        return tensor
    if tensor.shape[-1] > response_capacity:
        raise ValueError(f"Compact response width {tensor.shape[-1]} exceeds original width {response_capacity}")
    return F.pad(tensor, (0, response_capacity - tensor.shape[-1]))
