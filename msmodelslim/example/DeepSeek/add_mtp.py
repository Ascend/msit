# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
import os
import glob
import re
import argparse
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

from ascend_utils.common.security import json_safe_load, json_safe_dump, get_valid_read_path
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.tools.convert_fp8_to_bf16 import weight_dequant


def find_json_files(target_dir):
    """查找目录下的index和description json文件，不要求参数部分相同"""

    index_pattern = os.path.join(target_dir, "quant_model_weight_*.index.json")
    index_files = glob.glob(index_pattern)
    if not index_files:
        raise FileNotFoundError(f"can't find index.json")
    if len(index_files) > 1:
        raise ValueError(f"find mutiple index.json files")
    index_path = index_files[0]
    

    desc_pattern = os.path.join(target_dir, "quant_model_description_*.json")
    desc_files = glob.glob(desc_pattern)
    if not desc_files:
        raise FileNotFoundError(f"can't find description json file")
    if len(desc_files) > 1:
        raise ValueError(f"find mutiple description json files")
    desc_path = desc_files[0]
    
    return index_path, desc_path


def calculate_tensor_size(tensor):
    # 计算单个张量的总字节数
    return tensor.numel() * tensor.element_size()


def find_float_index(float_dir):
    """查找float目录下的唯一index.json文件"""
    # 查找所有.index.json文件
    index_pattern = os.path.join(float_dir, "*.index.json")
    index_files = glob.glob(index_pattern)
    
    if not index_files:
        raise FileNotFoundError(f"can't find index.json")
    if len(index_files) > 1:
        raise ValueError(f"find mutiple index.json files")
    
    return index_files[0]


def get_weight_map(float_index_path):
    org_data = json_safe_load(float_index_path)
    return org_data["weight_map"]


def get_tensor(tensor_name, safetensor_path, weight_map):
    filename = weight_map[tensor_name]
    file_path = os.path.join(safetensor_path, filename)
    with safe_open(file_path, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(tensor_name)
    return tensor


def get_prefix(name, last_index=-1):
    key_list = name.split(".")[:last_index]
    return ".".join(key_list)


def add_safetensors(org_paths, target_dir, new_safetensors_name, key_pattern=None):
    org_paths = get_valid_read_path(org_paths, is_dir=True, check_user_stat=False)
    target_dir = get_valid_read_path(target_dir, is_dir=True, check_user_stat=False)
    index_path, desc_path = find_json_files(target_dir)
    msmodelslim_logger.info(f"find file in target_dir: \nindex: {index_path}\ndescription: {desc_path}")

    float_index_path = find_float_index(org_paths)
    msmodelslim_logger.info(f"find index file in org_path: \n{float_index_path}")

    weight_map = get_weight_map(float_index_path)

    if key_pattern:
        key_regex = re.compile(key_pattern)
    
    index_data = json_safe_load(index_path)
    desc_data = json_safe_load(desc_path)

    current_total_size = index_data["metadata"].get("total_size", 0)
    tensor_names = weight_map.keys()

    if key_pattern:
        tensor_names = [name for name in tensor_names if key_regex.match(name)]

    new_data = {}
    for tensor_name in tqdm(tensor_names):
        if "weight_scale_inv" not in tensor_name:
            tensor = get_tensor(tensor_name, org_paths, weight_map)
            current_total_size += calculate_tensor_size(tensor)
            index_data["weight_map"][tensor_name] = new_safetensors_name
            desc_data[tensor_name] = "FLOAT"  # 假设所有新张量都是FLOAT类型
            
            mod_name = get_prefix(tensor_name)
            if mod_name + ".weight_scale_inv" in tensor_names:
                weight_scale_inv = get_tensor(mod_name + ".weight_scale_inv", org_paths, weight_map)
                tensor = weight_dequant(tensor, weight_scale_inv)
            new_data[tensor_name] = tensor
    
    save_file(new_data, os.path.join(target_dir, new_safetensors_name))
    index_data["metadata"]["total_size"] = current_total_size
    json_safe_dump(index_data, index_path, indent=4)
    json_safe_dump(desc_data, desc_path, indent=4)
    msmodelslim_logger.info("add success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='添加新的safetensors文件到现有模型')
    parser.add_argument('--quant_dir', help='量化模型文件所在目录')
    parser.add_argument('--float_dir', help='浮点safetensors文件所在目录')
    
    args = parser.parse_args()
    
    add_safetensors(args.float_dir, args.quant_dir, "mtp.safetensors", r'model\.layers\.61\..*')
    