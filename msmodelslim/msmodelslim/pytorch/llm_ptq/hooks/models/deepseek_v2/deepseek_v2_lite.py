# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from typing import List, Any, Dict
from transformers import PreTrainedModel

from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.hooks.hook_def import ProcessHook


def get_norm_linear_subgraph(model: PreTrainedModel):
    norm_linear = {}
    layer_num = model.config.num_hidden_layers
    for layer in range(layer_num):
        kv_b_layer = 'model.layers.' + str(layer) + '.self_attn.kv_b_proj'
        o_proj = 'model.layers.' + str(layer) + '.self_attn.o_proj'
        norm_linear[kv_b_layer] = [o_proj]

    # q/kv_a->norm kv_b->kv_a_layernorm
    for layer in range(layer_num):
        input_layernorm = 'model.layers.' + str(layer) + '.input_layernorm'
        q_proj = 'model.layers.' + str(layer) + '.self_attn.q_proj'
        kv_a_proj_with_mqa = 'model.layers.' + str(layer) + '.self_attn.kv_a_proj_with_mqa'
        norm_linear[input_layernorm] = [q_proj, kv_a_proj_with_mqa]

        kv_b_proj = 'model.layers.' + str(layer) + '.self_attn.kv_b_proj'
        kv_a_layernorm = 'model.layers.' + str(layer) + '.self_attn.kv_a_layernorm'
        norm_linear[kv_a_layernorm] = [kv_b_proj]

    return norm_linear


def modify_smooth_args(cfg: AntiOutlierConfig,
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


def get_hooks(model: PreTrainedModel):
    hooks = {
        ProcessHook.GET_NORM_LINEAR_SUBGRAPH: get_norm_linear_subgraph,
        ProcessHook.MODIFY_SMOOTH_ARGS: modify_smooth_args
    }
    return hooks
