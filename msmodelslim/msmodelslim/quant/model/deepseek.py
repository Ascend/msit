#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Dict, List

from transformers import PreTrainedModel

from msmodelslim.quant.model.base import BaseModelAdapter
from msmodelslim.quant.model.registry import MODEL_ADAPTER_REGISTRY

_DEEPSEEK_V2_MODEL_TYPES = [
    "deepseek_v2",
    "deepseekv2",
    "deepseek_v3",
    "deepseekv3",
]


@MODEL_ADAPTER_REGISTRY.register_by_name("deepseekv2")
@MODEL_ADAPTER_REGISTRY.register_by_name("deepseek_v2")
@MODEL_ADAPTER_REGISTRY.register_by_name("deepseekv3")
@MODEL_ADAPTER_REGISTRY.register_by_name("deepseek_v3")
class DeepSeekAdapter(BaseModelAdapter):

    def __init__(self, model: PreTrainedModel):
        super().__init__(model)

    @staticmethod
    def is_deepseek_v2_chat(model: PreTrainedModel):
        if hasattr(model.config, 'model_type') and model.config.model_type in _DEEPSEEK_V2_MODEL_TYPES:
            if hasattr(model.config, 'q_lora_rank') and getattr(model.config, 'q_lora_rank') is not None:
                return True

        return False

    @staticmethod
    def is_deepseek_v2_lite(model: PreTrainedModel):
        if hasattr(model.config, 'model_type') and model.config.model_type in _DEEPSEEK_V2_MODEL_TYPES:
            if hasattr(model.config, 'q_lora_rank') and getattr(model.config, 'q_lora_rank') is None:
                return True

        return False

    def get_global_norm_linear_smooth_pair(self) -> Dict[str, List[str]]:
        norm_linear = {}
        layer_num = self.model.config.num_hidden_layers

        # q_a/kv_a->norm kv_b->kv_a_layernorm q_b_proj->q_a_layernorm
        for layer in range(layer_num):
            input_layernorm = 'model.layers.' + str(layer) + '.input_layernorm'
            q_a_proj = 'model.layers.' + str(layer) + '.self_attn.q_a_proj'
            kv_a_proj_with_mqa = 'model.layers.' + str(layer) + '.self_attn.kv_a_proj_with_mqa'
            norm_linear[input_layernorm] = [q_a_proj, kv_a_proj_with_mqa]

            kv_b_proj = 'model.layers.' + str(layer) + '.self_attn.kv_b_proj'
            kv_a_layernorm = 'model.layers.' + str(layer) + '.self_attn.kv_a_layernorm'
            norm_linear[kv_a_layernorm] = [kv_b_proj]

            if self.is_deepseek_v2_chat(self.model):
                q_b_proj = 'model.layers.' + str(layer) + '.self_attn.q_b_proj'
                q_a_layernorm = 'model.layers.' + str(layer) + '.self_attn.q_a_layernorm'
                norm_linear[q_a_layernorm] = [q_b_proj]

        return norm_linear

    def get_global_linear_linear_smooth_pair(self) -> Dict[str, List[str]]:
        linear_linear = {}
        layer_num = self.model.config.num_hidden_layers

        # o->kv_b_proj
        for layer in range(layer_num):
            kv_b_layer = 'model.layers.' + str(layer) + '.self_attn.kv_b_proj'
            o_proj = 'model.layers.' + str(layer) + '.self_attn.o_proj'
            linear_linear[kv_b_layer] = [o_proj]

        return linear_linear
