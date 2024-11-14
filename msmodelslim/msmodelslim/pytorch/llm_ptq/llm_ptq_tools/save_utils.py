#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import torch
from safetensors.torch import save_file
from ascend_utils.common.security import json_safe_dump
from ascend_utils.common.security import get_valid_write_path


ONE_GB_FILE_BYTES = 1073741824  # 1G, 1 * 1024 * 1024 * 1024


def format_idx(idx):
    return "{:05}".format(idx)


def get_index_json(file_map_dict, total_size):
    index_json_dict = {
        'metadata': {'total_size': total_size},
        'weight_map': file_map_dict
    }
    return index_json_dict


def get_tensor_size(tensor):
    return tensor.numel() * tensor.element_size()


def save_file_partial(weight_dict, safetensors_name, part_file_size):
    file_map_dict = {}
    part_weight_list = []
    part_weight_dict = {}
    max_part_size_bytes = ONE_GB_FILE_BYTES * part_file_size  # 单个权重文件最大大小，实际情况下会略大于该大小

    part_weight_size = 0

    total_size = 0
    for key, value in weight_dict.items():
        part_weight_dict[key] = value
        tensor_size = get_tensor_size(value)
        part_weight_size += tensor_size
        total_size += tensor_size
        if part_weight_size > max_part_size_bytes:
            part_weight_size = 0
            part_weight_list.append(part_weight_dict)
            part_weight_dict = {}
    if part_weight_dict:
        part_weight_list.append(part_weight_dict)

    part_file_count = len(part_weight_list)
    # 仿照开源权重命名均为model-0000x-of-0000x.safetensors，超过99999命名为model-x-of-x.safetensors
    if part_file_count <= 99999:
        format_func = format_idx
    else:
        format_func = str
    for i in range(part_file_count):
        part_file_name = safetensors_name.replace(".safetensors", f"-{format_func(i + 1)}-of-"
                                                                  f"{format_func(part_file_count)}.safetensors")
        unit_name = part_file_name.split('/')[-1]
        for key in part_weight_list[i].keys():
            file_map_dict[key] = unit_name
        part_file_path = get_valid_write_path(part_file_name, extensions=[".safetensors"])
        save_file(part_weight_list[i], part_file_path)

    index_json_dict = get_index_json(file_map_dict, total_size)

    index_json_name = safetensors_name + ".index.json"
    index_json_path = get_valid_write_path(index_json_name, extensions=[".json"])
    json_safe_dump(index_json_dict, index_json_path, indent=2)
