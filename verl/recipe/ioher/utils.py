# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import DictConfig


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Light config validation, modeled on the SPIN/SPPO recipes.

    Most of the heavy lifting is already done by verl's own
    ``verl.utils.config.validate_config``; this just adds IOHER-specific
    checks so failures show up early instead of mid-training.
    """
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    assert real_train_batch_size % n_gpus == 0, (
        f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
    )

    ioh_cfg = config.algorithm.get("ioh", None)
    assert ioh_cfg is not None, "config.algorithm.ioh is required for the IOHER recipe"
    assert ioh_cfg.get("phrases", None), "config.algorithm.ioh.phrases must list at least one inoculation phrase"
    assert ioh_cfg.get("inject_mode", "system") in ("system", "user_suffix"), (
        f"unknown ioh.inject_mode {ioh_cfg.get('inject_mode')}; expected 'system' or 'user_suffix'"
    )
    assert config.actor_rollout_ref.actor.get("ioh_sft_coef", None) is not None, (
        "actor.ioh_sft_coef must be set (used via IOHERActorConfig)"
    )
    if ioh_cfg.get("use_mistake_specific", True):
        assert ioh_cfg.get("judge_max_prompt_tokens", 0) > 0, "ioh.judge_max_prompt_tokens must be positive"
        assert ioh_cfg.get("judge_max_new_tokens", 0) > 0, "ioh.judge_max_new_tokens must be positive"
        assert ioh_cfg.get("judge_max_attempts", 0) > 0, "ioh.judge_max_attempts must be positive"
        assert config.actor_rollout_ref.rollout.agent.get("agent_loop_manager_class") == (
            "recipe.ioher.judge_agent_loop.IOHERAgentLoopManager"
        ), "mistake-specific IOHER requires IOHERAgentLoopManager"

    if ioh_cfg.get("inject_mode", "system") == "system":
        assert config.data.get("return_raw_chat", False), (
            "ioh.inject_mode=system requires data.return_raw_chat=True so the "
            "trainer can rebuild prompts with the chat template."
        )

    print("[ioher.validate_config] All configuration checks passed.")
