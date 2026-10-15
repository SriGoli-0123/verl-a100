import torch
from tensordict import TensorDict

from recipe.ioher.dp_actor import install
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric


class _MockEngine:
    def __init__(self):
        self.events = []
        self.scaler = None

    def train_batch(self, data, loss_function):
        self.events.append("grpo_backward")
        self.optimizer_step()
        return {"metrics": {}}

    def optimizer_step(self):
        self.events.append("optimizer_step")
        return 1.0

    def forward_step(self, micro_batch, loss_function, forward_only):
        self.events.append("sft_forward")
        loss = torch.ones((), requires_grad=True)
        return loss, {
            "metrics": {
                "actor/ioh_sft_loss": Metric(value=1.0, aggregation=AggregationType.SUM),
            }
        }

    def get_data_parallel_group(self):
        return None

    def is_mp_src_rank_with_outputs(self):
        return True


def _batch():
    batch_size, prompt_length, response_length = 4, 3, 2
    input_ids = torch.arange(batch_size * (prompt_length + response_length)).reshape(batch_size, -1)
    attention_mask = torch.ones_like(input_ids)
    responses = input_ids[:, -response_length:]
    response_mask = torch.ones_like(responses)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.arange(input_ids.shape[1]).expand(batch_size, -1),
            "prompts": input_ids[:, :prompt_length],
            "responses": responses,
            "response_mask": response_mask,
            "ioh_input_ids": input_ids + 100,
            "ioh_attention_mask": attention_mask,
            "ioh_position_ids": torch.arange(input_ids.shape[1]).expand(batch_size, -1),
            "ioh_responses": responses + 100,
            "ioh_response_mask": response_mask,
        },
        batch_size=[batch_size],
    )
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        dp_size=1,
        use_remove_padding=True,
    )
    return data


def test_sft_backward_precedes_inherited_optimizer_step():
    engine = _MockEngine()
    install(engine, ioh_sft_coef=0.01)

    output = engine.train_batch(_batch(), loss_function=None)

    assert engine.events == [
        "grpo_backward",
        "sft_forward",
        "sft_forward",
        "optimizer_step",
    ]
    metric = output["metrics"]["actor/ioh_sft_loss"]
    assert isinstance(metric, Metric)
    assert metric.aggregate() == 2.0
