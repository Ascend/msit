# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import get_offloaded_weights_loader_if_have
from msmodelslim.pytorch.llm_ptq.accelerate_adapter import enable_adapter, replace_device_align_hook_if_needed


def tensor_memory_page_preload(model):
    enable_adapter()
    replace_device_align_hook_if_needed(model)
    loader = get_offloaded_weights_loader_if_have(model)

    with torch.no_grad():
        for key in loader:
            loader[key].sum()
