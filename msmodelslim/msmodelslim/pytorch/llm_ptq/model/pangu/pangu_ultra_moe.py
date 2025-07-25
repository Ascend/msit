# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from typing import List, Any, Dict, Optional, Type, Union, TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from msmodelslim.pytorch.llm_ptq.model.base import ModelAdapter, ModelAdapterRegistry

if TYPE_CHECKING:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig

_PANGU_MODEL_TYPES = [
    "pangu_ultra_moe",
]


def is_pangu_ultra_moe(model: Union[PreTrainedModel, nn.Module]):
    if isinstance(model, PreTrainedModel) and hasattr(model.config, 'model_type'):
        return model.config.model_type in _PANGU_MODEL_TYPES
    return False


def is_pangu_ultra_moe_chat(model: Union[PreTrainedModel, nn.Module]) -> bool:
    if isinstance(model, PreTrainedModel) and hasattr(model.config, 'model_type'):
        if model.config.model_type in _PANGU_MODEL_TYPES:
            if hasattr(model.config, 'attention_q_lora_dim') and \
                getattr(model.config, 'attention_q_lora_dim') is not None:
                return True
    return False


@ModelAdapterRegistry.register("pangu_ultra_moe")
class PanguUltraMoEAdapter(ModelAdapter):

    def __init__(self, model: PreTrainedModel):
        super().__init__(model)
        self.is_chat = is_pangu_ultra_moe_chat(model)

    def get_norm_linear_subgraph(self,
                                 cfg: 'AntiOutlierConfig',
                                 dummy_input: Optional[torch.Tensor] = None,
                                 norm_class: Optional[List[Type[nn.Module]]] = None):
        norm_linear = {}
        layer_num = self.model.config.num_hidden_layers

        # o->kv_b_proj
        for layer in range(layer_num):
            kv_b_layer = 'model.layers.' + str(layer) + '.self_attn.kv_b_proj'
            o_proj = 'model.layers.' + str(layer) + '.self_attn.o_proj'
            norm_linear[kv_b_layer] = [o_proj]

        # q_a/kv_a->norm kv_b->kv_a_layernorm q_b_proj->q_a_layernorm
        for layer in range(layer_num):
            input_layernorm = 'model.layers.' + str(layer) + '.input_layernorm'
            q_a_proj = 'model.layers.' + str(layer) + '.self_attn.q_a_proj'
            kv_a_proj_with_mqa = 'model.layers.' + str(layer) + '.self_attn.kv_a_proj_with_mqa'
            norm_linear[input_layernorm] = [q_a_proj, kv_a_proj_with_mqa]

            if self.is_chat:
                q_b_proj = 'model.layers.' + str(layer) + '.self_attn.q_b_proj'
                q_a_layernorm = 'model.layers.' + str(layer) + '.self_attn.q_a_layernorm'
                norm_linear[q_a_layernorm] = [q_b_proj]
            else:
                kv_b_proj = 'model.layers.' + str(layer) + '.self_attn.kv_b_proj'
                kv_a_layernorm = 'model.layers.' + str(layer) + '.self_attn.kv_a_layernorm'
                norm_linear[kv_a_layernorm] = [kv_b_proj]

        return norm_linear

    def modify_smooth_args(self,
                           cfg: 'AntiOutlierConfig',
                           norm_name: str,
                           linear_names: List[str],
                           args: List[Any],
                           kwargs: Dict[str, Any]):
        # 针对该模型进行m4量化时，需要对特定层开启偏移
        if cfg.anti_method == 'm4':
            is_shift = False
            if 'norm' in norm_name and 'kv_b' not in linear_names[0]:
                is_shift = True

            kwargs['is_shift'] = is_shift
            kwargs['alpha'] = cfg.alpha
        return args, kwargs
