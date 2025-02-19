# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from ascend_utils.common.security import json_safe_load, get_valid_read_path, MAX_READ_FILE_SIZE_32G
from msmodelslim.pytorch.llm_ptq.accelerate_adapter import replace_device_align_hook_if_needed, enable_adapter
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import get_offloaded_weights_loader_if_have

WEIGHT_SCALE_INV = '.weight_scale_inv'
MAX_FILE_NUM = 3
HF_HOOK = '_hf_hook'


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape(M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """

    # Get the original dimensions of weight
    m, n = weight.shape

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)

    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:m, :n]

    # Perform element-wise multiplication
    weight *= scale_expanded

    # Convert the output to the default dtype
    weight = weight.to(torch.bfloat16)

    return weight


def get_tensor(tensor_name, fp8_path, weight_map, loaded_files, loaded_files_list):
    file_name = weight_map[tensor_name]
    if file_name not in loaded_files:
        file_path = os.path.join(fp8_path, file_name)
        file_path = get_valid_read_path(file_path, 'safetensors', size_max=MAX_READ_FILE_SIZE_32G)
        loaded_files[file_name] = load_file(file_path, device='cpu')
    else:
        loaded_files_list.remove(file_name)
    loaded_files_list.insert(0, file_name)

    result = loaded_files[file_name][tensor_name]

    if len(loaded_files_list) > MAX_FILE_NUM:
        del loaded_files[loaded_files_list[-1]]
        loaded_files_list.pop()

    return result


def get_module_by_name(model, submodule_key=None):
    if submodule_key is None:
        return submodule_key
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s, None)
    return cur_mod


def auto_convert_model_fp8_to_bf16(model, model_path):
    model_index_path = os.path.join(model_path, "model.safetensors.index.json")
    model_index = json_safe_load(model_index_path)
    weight_map = model_index['weight_map']

    convert_list = set(
        map(lambda x: x.replace(WEIGHT_SCALE_INV, ''), filter(lambda x: WEIGHT_SCALE_INV in x, weight_map.keys())))
    used_list = []
    for name, _ in model.named_modules():
        if name in convert_list:
            used_list.append(name)

    if not used_list:
        return

    enable_adapter()
    replace_device_align_hook_if_needed(model)

    loaded_files = {}
    loaded_files_list = []

    with torch.no_grad():
        for name in tqdm(used_list, desc='fp8 to bf16'):
            module = get_module_by_name(model, name)
            scale = get_tensor(name + WEIGHT_SCALE_INV, model_path, weight_map, loaded_files, loaded_files_list)

            weight_loader = get_offloaded_weights_loader_if_have(module)
            if weight_loader and getattr(module, HF_HOOK).old_hook.offload:
                weight_name = name + '.weight'
                weight_loader[weight_name][:] = weight_dequant(weight_loader[weight_name], scale)
                continue

            device = getattr(module, HF_HOOK).old_hook.execution_device
            if device != 'cpu':
                device = f'npu:{device}'
            scale = scale.to(device)
            module.weight[:] = weight_dequant(module.weight, scale)
