# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from transformers import PreTrainedModel

_DEEPSEEK_V2_MODEL_TYPES = [
    "deepseek_v2",
    "deepseekv2",
    "deepseek_v3",
    "deepseekv3",
]

_HUNYUAN_MODEL_TYPES = [
    "hunyuan"
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


def is_hunyuan_large(model: PreTrainedModel):
    if hasattr(model.config, 'model_type') and model.config.model_type in _HUNYUAN_MODEL_TYPES:
        if (hasattr(model.config, 'num_experts')
                and isinstance(model.config.num_experts, int)
                and model.config.num_experts > 1):
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
    elif is_hunyuan_large(model):
        from .models.hunyuan.hunyuan_large import get_hooks
        return get_hooks(model)

    return hooks


class CuttingMethodRegistry:
    def __init__(self):
        # 初始化 cutting_methods 字典
        self.cutting_methods = {}
        # 在类内部注册切割方法
        self.register_cutting_methods()

    @staticmethod
    def default_cut(comming_max, comming_min, in_hidden_size):
        """
        默认切割方法，可修改。
        """
        res_dim = comming_max.shape[-1] - in_hidden_size
        _, kv_max = torch.split(comming_max, [in_hidden_size, res_dim])
        _, kv_min = torch.split(comming_min, [in_hidden_size, res_dim])
        key_max, value_max = torch.chunk(kv_max, 2, dim=0)
        key_min, value_min = torch.chunk(kv_min, 2, dim=0)
        return key_max, value_max, key_min, value_min

    @staticmethod
    def internlm2_cut(comming_max, comming_min, m):
        """
        internlm2 模型的切割方法。
        """
        if m.config.num_key_value_heads == 0:
            raise ValueError('Num_key_value_heads must not be zero in model config.')
        if m.config.num_attention_heads == 0:
            raise ValueError('Num_attention_heads must not be zero in model config.')
        if m.config.hidden_size == 0:
            raise ValueError('Hidden_size must not be zero in model config.')

        gs = m.config.num_attention_heads // m.config.num_key_value_heads + 2
        d = m.config.hidden_size // m.config.num_attention_heads
        h = comming_max.shape[0] // (gs * d)
        qkv_states_max = comming_max.view(h, gs, d)
        qkv_states_min = comming_min.view(h, gs, d)
        key_max = qkv_states_max[:, -2, :].reshape(-1)
        value_max = qkv_states_max[:, -1, :].reshape(-1)
        key_min = qkv_states_min[:, -2, :].reshape(-1)
        value_min = qkv_states_min[:, -1, :].reshape(-1)
        return key_max, value_max, key_min, value_min
    
    def register_cutting_methods(self):
        """
        注册切割方法，将方法添加到 cutting_methods 字典中。
        """
        self.cutting_methods["internlm2"] = self.internlm2_cut

    def get_cutting_method(self, model_type):
        """
        根据模型类型获取对应的切割方法。如果模型类型不存在于字典中，则返回 None。
        
        参数:
        model_type (str): 模型类型的名称。
        
        返回:
        function: 对应的切割方法或 None。
        """
        return self.cutting_methods.get(model_type, None)

cutting_method_registry = CuttingMethodRegistry()
