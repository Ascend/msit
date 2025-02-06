# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from transformers import PreTrainedModel

_DEEPSEEK_V2_MODEL_TYPES = [
    "deepseek_v2",
    "deepseekv2",
    "deepseek_v3",
    "deepseekv3"
]


def is_deepseek_v2_chat(model: PreTrainedModel):
    if hasattr(model.config, 'model_type') and model.config.model_type in _DEEPSEEK_V2_MODEL_TYPES:
        if hasattr(model.config, 'q_lora_rank') and getattr(model.config, 'q_lora_rank') is not None:
            return True

    return False


def is_deepseek_v2_lite(model: PreTrainedModel):
    if hasattr(model.config, 'model_type') and model.config.model_type in _DEEPSEEK_V2_MODEL_TYPES:
        if hasattr(model.config, 'q_lora_rank') and getattr(model.config, 'q_lora_rank') is None:
            return True

    return False


def get_process_hooks(model: PreTrainedModel):
    hooks = {}
    if is_deepseek_v2_chat(model):
        from .models.deepseek_v2.deepseek_v2_chat import get_hooks
        return get_hooks(model)
    elif is_deepseek_v2_lite(model):
        from .models.deepseek_v2.deepseek_v2_lite import get_hooks
        return get_hooks(model)

    return hooks
