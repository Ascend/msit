#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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


@MODEL_ADAPTER_REGISTRY.register_by_name("qwen3")
@MODEL_ADAPTER_REGISTRY.register_by_name("qwen3_moe")
class Qwen3Adapter(BaseModelAdapter):

    def __init__(self, model: PreTrainedModel):
        super().__init__(model)
        self.is_moe = "moe" in self.model.config.model_type

    def get_global_norm_linear_smooth_pair(self) -> Dict[str, List[str]]:

        norm_linear = {}
        layer_num = self.model.config.num_hidden_layers

        for layer in range(layer_num):
            input_layernorm = 'model.layers.' + str(layer) + '.input_layernorm'
            q_proj = 'model.layers.' + str(layer) + '.self_attn.q_proj'
            k_proj = 'model.layers.' + str(layer) + '.self_attn.k_proj'
            v_proj = 'model.layers.' + str(layer) + '.self_attn.v_proj'

            norm_linear[input_layernorm] = [q_proj, k_proj, v_proj]

            if not self.is_moe:
                post_layernorm = 'model.layers.' + str(layer) + '.post_attention_layernorm'
                gate_proj = 'model.layers.' + str(layer) + '.mlp.gate_proj'
                up_proj = 'model.layers.' + str(layer) + '.mlp.up_proj'

                norm_linear[post_layernorm] = [gate_proj, up_proj]

        return norm_linear

    def get_global_linear_linear_smooth_pair(self) -> Dict[str, List[str]]:

        linear_linear = {}
        layer_num = self.model.config.num_hidden_layers

        for layer in range(layer_num):
            v_proj = 'model.layers.' + str(layer) + '.self_attn.v_proj'
            o_proj = 'model.layers.' + str(layer) + '.self_attn.o_proj'

            linear_linear[v_proj] = [o_proj]

            if not self.is_moe:
                up_proj = 'model.layers.' + str(layer) + '.mlp.up_proj'
                down_proj = 'model.layers.' + str(layer) + '.mlp.down_proj'

                linear_linear[up_proj] = [down_proj]

        return linear_linear
