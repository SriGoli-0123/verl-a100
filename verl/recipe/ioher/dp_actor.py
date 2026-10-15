"""Gaudi IOHER actor semantics adapted to VeRL's CUDA FSDP engine.

The stock engine still owns GRPO, micro-batching, clipping, and the optimizer
step. This module accumulates fixed-shape SFT gradients immediately before the
inherited optimizer step, matching the stable Gaudi IOHER actor path.
"""

import types

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import append_to_dict
from verl.workers.engine.utils import prepare_micro_batches
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from .compaction import compact_model_family, compact_response_mask, synchronized_compact_buckets

IOH_KEYS = (
    "ioh_input_ids",
    "ioh_attention_mask",
    "ioh_position_ids",
    "ioh_responses",
    "ioh_response_mask",
)


def compact_padded_tensordict(model_inputs, *, has_ioh: bool, pad_token_id: int):
    """Compact shared padding while preserving every supervised token."""
    response_capacity = model_inputs["responses"].shape[-1]
    prompt_capacity = model_inputs["input_ids"].shape[-1] - response_capacity
    families = [(model_inputs["attention_mask"], model_inputs["response_mask"])]
    if has_ioh:
        ioh_response_capacity = model_inputs["ioh_responses"].shape[-1]
        if ioh_response_capacity != response_capacity:
            raise ValueError("GRPO and IOHER response capacities must match")
        prompt_capacity = max(
            prompt_capacity,
            model_inputs["ioh_input_ids"].shape[-1] - ioh_response_capacity,
        )
        families.append((model_inputs["ioh_attention_mask"], model_inputs["ioh_response_mask"]))

    prompt_bucket, response_bucket = synchronized_compact_buckets(
        families,
        prompt_capacity=prompt_capacity,
        response_capacity=response_capacity,
    )
    compacted = dict(model_inputs)
    main = compact_model_family(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        responses=model_inputs["responses"],
        prompt_bucket=prompt_bucket,
        response_bucket=response_bucket,
        pad_token_id=pad_token_id,
    )
    compacted.update(main)
    compacted["prompts"] = main["input_ids"][..., :prompt_bucket]
    compacted["response_mask"] = compact_response_mask(
        model_inputs["response_mask"],
        main["attention_mask"],
        response_bucket,
    )

    for key in ("old_log_probs", "advantages", "ref_log_prob", "rollout_is_weights", "rollout_log_probs"):
        if key in model_inputs and model_inputs[key].shape[-1] == response_capacity:
            compacted[key] = model_inputs[key][..., :response_bucket]
    if "loss_mask" in model_inputs and model_inputs["loss_mask"].shape[-1] == response_capacity:
        compacted["loss_mask"] = compact_response_mask(
            model_inputs["loss_mask"],
            main["attention_mask"],
            response_bucket,
        )

    if has_ioh:
        ioh = compact_model_family(
            input_ids=model_inputs["ioh_input_ids"],
            attention_mask=model_inputs["ioh_attention_mask"],
            responses=model_inputs["ioh_responses"],
            prompt_bucket=prompt_bucket,
            response_bucket=response_bucket,
            pad_token_id=pad_token_id,
        )
        compacted.update({f"ioh_{key}": value for key, value in ioh.items()})
        compacted["ioh_response_mask"] = compact_response_mask(
            model_inputs["ioh_response_mask"],
            ioh["attention_mask"],
            response_bucket,
        )

    return compacted, (prompt_bucket, response_bucket), response_capacity


def build_fixed_sft_micro_batch(model_inputs):
    """Select supervised rows and fill unused slots with loss-masked copies."""
    ioh_response_mask = model_inputs["ioh_response_mask"]
    sample_has_loss = ioh_response_mask.sum(dim=-1) > 0
    n_inoc = int(sample_has_loss.sum().item())
    batch_size = int(ioh_response_mask.shape[0])

    if n_inoc > 0:
        valid_indices = torch.where(sample_has_loss)[0]
        if n_inoc < batch_size:
            selected_indices = torch.cat(
                (valid_indices, valid_indices[:1].expand(batch_size - n_inoc)),
                dim=0,
            )
        else:
            selected_indices = valid_indices
        sft_inputs = {
            "input_ids": model_inputs["ioh_input_ids"].index_select(0, selected_indices),
            "attention_mask": model_inputs["ioh_attention_mask"].index_select(0, selected_indices),
            "position_ids": model_inputs["ioh_position_ids"].index_select(0, selected_indices),
            "responses": model_inputs["ioh_responses"].index_select(0, selected_indices),
        }
        sft_response_mask = ioh_response_mask.index_select(0, selected_indices).float()
        real_slots = torch.arange(batch_size, device=ioh_response_mask.device) < n_inoc
        sft_response_mask = sft_response_mask * real_slots.unsqueeze(-1).to(sft_response_mask.dtype)
    else:
        selected_indices = torch.zeros(batch_size, dtype=torch.long, device=ioh_response_mask.device)
        prompts = model_inputs["prompts"].index_select(0, selected_indices)
        responses = model_inputs["responses"].index_select(0, selected_indices)
        input_ids = torch.cat((prompts, responses), dim=-1)
        attention_mask = model_inputs["attention_mask"].index_select(0, selected_indices)
        sft_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "responses": responses,
        }
        sft_response_mask = torch.zeros_like(
            model_inputs["response_mask"].index_select(0, selected_indices),
            dtype=torch.float32,
        )

    sft_inputs = {key: value.contiguous() for key, value in sft_inputs.items()}
    return sft_inputs, sft_response_mask.contiguous(), n_inoc


def _build_ioh_sub_batch(engine, micro_batch):
    sft_inputs, response_mask, n_inoc = build_fixed_sft_micro_batch(micro_batch)
    prompt_len = sft_inputs["input_ids"].shape[1] - sft_inputs["responses"].shape[1]
    sub = TensorDict(
        {
            **sft_inputs,
            "response_mask": response_mask,
            "prompts": sft_inputs["input_ids"][:, :prompt_len],
        },
        batch_size=[response_mask.shape[0]],
    )
    sub = left_right_2_no_padding(sub)
    sub["temperature"] = torch.ones(response_mask.shape[0], dtype=torch.float32, device=response_mask.device)

    for key in ("use_remove_padding", "pad_mode", "use_fused_kernels", "sp_size", "calculate_entropy"):
        value = tu.get_non_tensor_data(data=micro_batch, key=key, default=None)
        if value is not None:
            tu.assign_non_tensor(sub, **{key: value})
    tu.assign_non_tensor(
        sub,
        dp_size=micro_batch["dp_size"],
        sft_agg_mode=getattr(engine, "_ioh_sft_agg_mode", "token-mean"),
        sft_loss_scale=getattr(engine, "_ioh_sft_loss_scale", 1.0),
        n_inoc=n_inoc,
    )
    return sub


def _sft_loss(model_output, data, dp_group=None):
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    response_mask = data["response_mask"].to(log_prob.dtype)
    if bool(response_mask.sum() > 0):
        raw_loss = agg_loss(
            loss_mat=-log_prob,
            loss_mask=response_mask,
            loss_agg_mode=tu.get_non_tensor_data(data=data, key="sft_agg_mode", default="token-mean"),
        )
    else:
        raw_loss = (log_prob * 0.0).sum()
    scale = float(tu.get_non_tensor_data(data=data, key="sft_loss_scale", default=1.0))
    loss = raw_loss * scale
    metrics = {
        "actor/ioh_sft_loss": Metric(value=loss.detach(), aggregation=AggregationType.SUM),
        "actor/ioh_samples_in_micro": Metric(
            value=float(tu.get_non_tensor_data(data=data, key="n_inoc", default=0)),
            aggregation=AggregationType.SUM,
        ),
    }
    return loss, metrics


def _ioher_train_batch(self, data, loss_function):
    active = getattr(self, "_ioh_sft_coef", 0.0) > 0 and all(key in data.keys() for key in IOH_KEYS)
    if not active:
        return self._ioh_original_train_batch(data, loss_function)

    if bool(tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=False)):
        raise ValueError("IOHER fixed-shape SFT requires actor.use_dynamic_bsz=False")

    self._ioh_optimizer_batch = data
    self._ioh_aux_metrics = {}
    try:
        output = self._ioh_original_train_batch(data, loss_function)
        if self.is_mp_src_rank_with_outputs():
            output["metrics"].update(self._ioh_aux_metrics)
        return output
    finally:
        self._ioh_optimizer_batch = None
        self._ioh_aux_metrics = {}


def _ioher_optimizer_step(self):
    data = getattr(self, "_ioh_optimizer_batch", None)
    if data is not None:
        micro_batches, _ = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )
        self._ioh_sft_loss_scale = 1.0 / len(micro_batches)

        for micro_batch in micro_batches:
            sft_batch = _build_ioh_sub_batch(self, micro_batch)
            sft_loss, sft_output = self._ioh_original_forward_step(sft_batch, _sft_loss, False)
            weighted_loss = self._ioh_sft_coef * sft_loss
            scaler = getattr(self, "scaler", None)
            if scaler is not None:
                scaler.scale(weighted_loss).backward()
            else:
                weighted_loss.backward()

            append_to_dict(self._ioh_aux_metrics, sft_output.get("metrics", {}))

    return self._ioh_original_optimizer_step()


def install(engine, ioh_sft_coef: float, sft_agg_mode: str = "token-mean"):
    """Add the Gaudi IOHER SFT term to one live CUDA actor engine."""
    engine._ioh_original_train_batch = engine.train_batch
    engine._ioh_original_optimizer_step = engine.optimizer_step
    engine._ioh_original_forward_step = engine.forward_step
    engine._ioh_sft_coef = float(ioh_sft_coef)
    engine._ioh_sft_agg_mode = str(sft_agg_mode)
    engine._ioh_sft_loss_scale = 1.0
    engine._ioh_optimizer_batch = None
    engine._ioh_aux_metrics = {}
    engine.train_batch = types.MethodType(_ioher_train_batch, engine)
    engine.optimizer_step = types.MethodType(_ioher_optimizer_step, engine)
