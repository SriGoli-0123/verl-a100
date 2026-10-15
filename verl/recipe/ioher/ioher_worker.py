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
"""IOHER actor/rollout/ref worker for VeRL's CUDA engine architecture."""

import logging
import os

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


class IOHERActorRolloutRefWorker(ActorRolloutRefWorker):
    """Engine-based actor/rollout/ref worker for the IOHER recipe."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            ioh_sft_coef = float(self.config.actor.get("ioh_sft_coef", 0.0))
            if ioh_sft_coef > 0:
                from .dp_actor import install

                sft_agg_mode = str(self.config.actor.get("ioh_sft_loss_agg_mode", "token-mean"))
                sft_pack = bool(self.config.actor.get("ioh_sft_pack", False))
                sft_pack_max_tokens = self.config.actor.get("ioh_sft_pack_max_tokens", None)
                install(self.actor.engine, ioh_sft_coef, sft_agg_mode, sft_pack, sft_pack_max_tokens)
                sft_layout = (
                    f"packed inoculated SFT (max_tokens={sft_pack_max_tokens or 'micro x width'})"
                    if sft_pack
                    else "fixed-shape inoculated SFT"
                )
                print(
                    f"[ioher] actor engine patched: stock GRPO + {sft_layout} "
                    f"(coef={ioh_sft_coef}, agg={sft_agg_mode}) in one optimizer step."
                )
