# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
import os
import glob
import re
import argparse

from safetensors import safe_open
from safetensors.torch import save_file

from msmodelslim import logger as msmodelslim_logger


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
    with open(float_index_path, 'r') as f:
        org_data = json.load(f)
    return org_data["weight_map"]


def get_tensor(tensor_name, safetensor_path, weight_map):
    filename = weight_map[tensor_name]
    file_path = os.path.join(safetensor_path, filename)
    with safe_open(file_path, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(tensor_name)
    return tensor


def add_safetensors(org_paths, target_dir, new_safetensors_name, key_pattern=None):

    try:
        index_path, desc_path = find_json_files(target_dir)
        msmodelslim_logger.info(f"find file in target_dir: \nindex: {index_path}\ndescription: {desc_path}")
    except Exception as e:
        msmodelslim_logger.info(f"error: {str(e)}")
        return

    try:
        float_index_path = find_float_index(org_paths)
        msmodelslim_logger.info(f"find index file in org_path: \n{float_index_path}")
    except Exception as e:
        msmodelslim_logger.info(f"error: {str(e)}")
        return

    weight_map = get_weight_map(float_index_path)


    if key_pattern:
        key_regex = re.compile(key_pattern)
    

    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    with open(desc_path, 'r') as f:
        desc_data = json.load(f)

    current_total_size = index_data["metadata"].get("total_size", 0)
    tensor_names = weight_map.keys()

    if key_pattern:
        tensor_names = [name for name in tensor_names if key_regex.match(name)]
    new_data = {}
    for tensor_name in tensor_names:

        tensor = get_tensor(tensor_name, org_paths, weight_map)
    
        current_total_size += calculate_tensor_size(tensor)
            
        index_data["weight_map"][tensor_name] = new_safetensors_name
        desc_data[tensor_name] = "FLOAT"  # 假设所有新张量都是FLOAT类型
        new_data[tensor_name] = tensor

    save_file(new_data, os.path.join(target_dir, new_safetensors_name))


    index_data["metadata"]["total_size"] = current_total_size
    

    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=4)
        
    with open(desc_path, 'w') as f:
        json.dump(desc_data, f, indent=4)
    msmodelslim_logger.info("add success!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='添加新的safetensors文件到现有模型')
    parser.add_argument('--quant_dir', help='量化模型文件所在目录')
    parser.add_argument('--float_dir', help='浮点safetensors文件所在目录')
    parser.add_argument('--new_safetensors_name', help='新的safetensors文件名')
    parser.add_argument('--key-pattern', help='tensor key的正则表达式模式，只更新匹配的key', 
                       default=r'model\.layers\.61\..*')
    
    args = parser.parse_args()
    


    add_safetensors(args.float_dir, args.quant_dir, args.new_safetensors_name, args.key_pattern)
    