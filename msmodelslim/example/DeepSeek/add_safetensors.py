# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import glob
import argparse
import dataclasses

from tqdm import tqdm
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from convert_fp8_to_bf16 import weight_dequant

from ascend_utils.common.security import json_safe_load, json_safe_dump, get_valid_read_path
from msmodelslim import logger as msmodelslim_logger


@dataclasses.dataclass
class AddTensorsConfig:
    """
    org_paths (str): 原始模型safetensors文件所在目录路径
    target_dir (str): 目标量化模型目录路径
    safetensors_prefix (str): 新生成的safetensors文件的前缀名
    max_file_size_gb (float): 单个safetensors文件的最大大小(GB)，默认5GB
    prefix (str, optional): 只添加指定前缀的tensor，默认None表示添加所有tensor
    should_quant (bool, optional): 是否量化MTP层，当前仅支持W8A8动态量化
    """

    org_paths: str
    target_dir: str
    safetensors_prefix: str
    max_file_size_gb: int = 5
    prefix: str = ''
    should_quant: bool = False

    def __post_init__(self):
        if self.max_file_size_gb is None:
            self.max_file_size_gb = 5
        if self.prefix is None:
            self.prefix = ''
        if self.should_quant is None:
            self.should_quant = False


def weight_quant(tensor: torch.Tensor):
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(tensor.dtype)


def find_file_with_pattern(target_dir, pattern):
    """查找目录下的符合pattern的文件"""
    pattern = os.path.join(target_dir, pattern)
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"can't find {pattern} in {target_dir}")
    if len(files) > 1:
        raise ValueError(f"find mutiple json files")
    return files[0]


def calculate_tensor_size(tensor):
    # 计算单个张量的总字节数
    return tensor.numel() * tensor.element_size()


def get_weight_map(float_index_path):
    org_data = json_safe_load(float_index_path)
    return org_data.get("weight_map", {})


def get_tensor(tensor_name, safetensor_path, weight_map):
    filename = weight_map[tensor_name]
    file_path = os.path.join(safetensor_path, filename)
    with safe_open(file_path, framework="pt", device="cpu") as f:
        if tensor_name in f.keys():
            tensor = f.get_tensor(tensor_name)
        else:
            raise KeyError(f"tensor {tensor_name} not found in {file_path}")
    return tensor


def get_prefix(name, last_index=-1):
    key_list = name.split(".")[:last_index]
    return ".".join(key_list)


def add_safetensors(cfg: AddTensorsConfig):
    """将原始模型的tensor添加到量化模型中，支持分文件保存
    
    Args:
        cfg (AddTensorsConfig): 配置
    """
    # 验证输入输出路径
    org_paths = get_valid_read_path(cfg.org_paths, is_dir=True, check_user_stat=False)
    target_dir = get_valid_read_path(cfg.target_dir, is_dir=True, check_user_stat=False)
    index_path = find_file_with_pattern(target_dir, "quant_model_weight_*.index.json")
    desc_path = find_file_with_pattern(target_dir, "quant_model_description_*.json")

    msmodelslim_logger.info(f"find file in target_dir: \nindex: {index_path}\ndescription: {desc_path}")

    float_index_path = find_file_with_pattern(org_paths, "*.index.json")
    msmodelslim_logger.info(f"find index file in org_path: \n{float_index_path}")

    weight_map = get_weight_map(float_index_path)

    index_data = json_safe_load(index_path)
    desc_data = json_safe_load(desc_path)
    if "metadata" not in index_data:
        index_data["metadata"] = {}
    if "weight_map" not in index_data:
        index_data["weight_map"] = {}
    current_total_size = index_data.get("metadata", {}).get("total_size", 0)
    tensor_names = weight_map.keys()

    if cfg.prefix:
        tensor_names = [name for name in tensor_names if name.startswith(cfg.prefix)]

    max_file_size = cfg.max_file_size_gb * (1024 ** 3)
    current_file_size = 0
    new_data = {}
    file_count = 0

    def flush_tensors():
        nonlocal new_data
        nonlocal current_file_size
        nonlocal file_count

        if not new_data:
            return

        file_name = f"{cfg.safetensors_prefix}-{file_count + 1}.safetensors"
        save_file(new_data, os.path.join(target_dir, file_name))
        # 更新索引
        for name in new_data.keys():
            index_data["weight_map"][name] = file_name
        new_data = {}
        current_file_size = 0
        file_count += 1

    def add_tensor(name, quant_type, quant_tensor):
        nonlocal current_file_size
        nonlocal current_total_size

        tensor_size = calculate_tensor_size(quant_tensor)
        current_total_size += tensor_size
        # 如果当前文件大小超过限制，保存当前文件并开始新文件
        if current_file_size + tensor_size > max_file_size:
            flush_tensors()
        new_data[name] = quant_tensor
        desc_data[name] = quant_type
        current_file_size += tensor_size

    for tensor_name in tqdm(tensor_names):
        if "weight_scale_inv" in tensor_name:
            continue

        tensor = get_tensor(tensor_name, org_paths, weight_map)
        mod_name = get_prefix(tensor_name)
        scale_inv_name = mod_name + ".weight_scale_inv"
        if scale_inv_name in tensor_names:
            try:
                weight_scale_inv = get_tensor(mod_name + ".weight_scale_inv", org_paths, weight_map)
                tensor = weight_dequant(tensor, weight_scale_inv)

            except KeyError:
                msmodelslim_logger.warning(f"{mod_name + '.weight_scale_inv'} not found in org_paths, \
                                           skip convert {mod_name} from fp8 to bf16")

        if not cfg.should_quant:
            add_tensor(tensor_name, 'FLOAT', tensor)
            continue

        if 'layernorm' in tensor_name:
            add_tensor(tensor_name, 'FLOAT', tensor)
            continue

        if not ('self_attn' in tensor_name or 'experts' in tensor_name):
            add_tensor(tensor_name, 'FLOAT', tensor)
            continue

        if tensor_name in (
                f'{cfg.prefix}self_attn.kv_b_proj.weight',
        ):
            add_tensor(tensor_name, 'FLOAT', tensor)
            continue

        tensor, scale = weight_quant(tensor)
        add_tensor(tensor_name, 'W8A8_DYNAMIC', tensor)
        add_tensor(tensor_name + '_scale', 'W8A8_DYNAMIC', scale)

    # 保存最后一个文件
    flush_tensors()

    index_data["metadata"]["total_size"] = current_total_size
    json_safe_dump(index_data, index_path, indent=4)
    json_safe_dump(desc_data, desc_path, indent=4)
    msmodelslim_logger.info("add success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='添加新的safetensors文件到现有模型')
    parser.add_argument('--quant_dir', help='量化模型文件所在目录')
    parser.add_argument('--float_dir', help='浮点safetensors文件所在目录')

    args = parser.parse_args()

    add_tensors_cfg = AddTensorsConfig(
        org_paths=args.float_dir,
        target_dir=args.quant_dir,
        safetensors_prefix="mtp",
        max_file_size_gb=5,
        prefix='model.layers.61.',
        should_quant=False,
    )
    add_safetensors(add_tensors_cfg)
