# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import copy
import torch

from msmodelslim import logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.duquant_alg import DuQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.wrapper.wrapper_base import DuQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.wrapper.wrapper_base import DuQuantLinearWrapper
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.wrapper.wrapper_llama import DuQuantLlamaAttentionWrapper
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.wrapper.wrapper_llama import DuQuantLlamaMLPWrapper
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.wrapper.wrapper_baichuan import DuQuantBaichuanAttentionWrapper
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.duquant.wrapper.wrapper_baichuan import DuQuantBaichuanMLPWrapper


def apply_duquant_baichuan(module, config: DuQuantConfig, name=''):
    if type(module) in [DuQuantLinearWrapper, DuQuantBaichuanAttentionWrapper, DuQuantBaichuanMLPWrapper]:
        return

    wrapper_linear = ["o_proj", "down_proj"]

    logger.info(f"apply duquant for {name} {type(module)}")

    for attr in dir(module):
        tmp = getattr(module, attr)

        if attr == "self_attn":
            apply_duquant_baichuan(tmp, config, name + '.' + attr)
            setattr(module, attr, DuQuantBaichuanAttentionWrapper(tmp, name + '.' + attr, copy.copy(config)))

        elif attr == "mlp":
            apply_duquant_baichuan(tmp, config, name + '.' + attr)
            setattr(module, attr, DuQuantBaichuanMLPWrapper(tmp, name + '.' + attr, copy.copy(config)))

        elif isinstance(tmp, torch.nn.Linear) and attr in wrapper_linear:
            setattr(module, attr, DuQuantLinearWrapper(tmp, name + '.' + attr, copy.copy(config)))

    for name1, child in module.named_children():
        apply_duquant_baichuan(child, config, name + '.' + name1 if name != '' else name1)

    return


def apply_duquant_llama(module, config: DuQuantConfig, name=''):
    if type(module) in [DuQuantLinearWrapper, DuQuantLlamaAttentionWrapper, DuQuantLlamaMLPWrapper]:
        return

    wrapper_linear = ["o_proj", "down_proj"]

    logger.info(f"apply duquant for {name} {type(module)}")

    for attr in dir(module):
        tmp = getattr(module, attr)

        if attr == "self_attn":
            apply_duquant_llama(tmp, config, name + '.' + attr)
            setattr(module, attr, DuQuantLlamaAttentionWrapper(tmp, name + '.' + attr, copy.copy(config)))

        elif attr == "mlp":
            apply_duquant_llama(tmp, config, name + '.' + attr)
            setattr(module, attr, DuQuantLlamaMLPWrapper(tmp, name + '.' + attr, copy.copy(config)))

        elif isinstance(tmp, torch.nn.Linear) and attr in wrapper_linear:
            setattr(module, attr, DuQuantLinearWrapper(tmp, name + '.' + attr, copy.copy(config)))

    for name1, child in module.named_children():
        apply_duquant_llama(child, config, name + '.' + name1 if name != '' else name1)

    return


def apply_duquant(model, config: DuQuantConfig, name=''):
    if 'llama' in str(type(model)).lower():
        apply_duquant_llama(model, config, name)
    elif 'baichuan' in str(type(model)).lower():
        apply_duquant_baichuan(model, config, name)
    else:
        raise ValueError("Unsupport model")
