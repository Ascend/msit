import os
import re
import psutil
import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    init_weight_quant_normal,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from ascend_utils.common.security import get_valid_write_path, get_write_directory
from ascend_utils.common.security import json_safe_dump
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save_utils import get_tensor_size
from msmodelslim.pytorch.lowbit.atomic_power_outlier import \
    quant_one_weight_by_outliers as quant_one_weight_by_outliers_low_bit
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    fake_quantize_save
)


def check_linear_weight(name:str):
    return name.endswith(".weight") and "norm" not in name.lower() and "embed" not in name.lower()


def get_prefix(name, last_index = -1):
    key_list = name.split(".")[:last_index]
    return ".".join(key_list)


def get_safetensors_name(path):
    weight_files = []
    pattern = r"model-(\d+)-of-(\d+).safetensors"
    path_list = os.listdir(path)
    for filename in path_list:
        if filename.endswith(".safetensors"):
            match = re.search(pattern, filename)
            if match:
                weight_idx = match.group(1)
                total_weights = match.group(2)
                weight_files.append({
                    "id": int(weight_idx),
                    "file_path": os.path.join(path, filename),
                    "weight_idx": weight_idx,
                    "total_weights": total_weights
                })
    weight_files = sorted(weight_files, key=lambda x: x["id"], reverse=False)
    return weight_files


def get_total_size(tensors):
    total_size = 0
    for tensor in tensors.values():
        tensor_size = get_tensor_size(tensor)
        total_size += tensor_size
    return total_size


class DataFreeConverter():
    def __init__(self, cfg):
        self.cfg = cfg
        self.quant_model_json_description = QuantModelJsonDescription(cfg.model_quant_type,
                                                                      cfg.use_kvcache_quant,
                                                                      cfg.use_fa_quant)
        self.logger = msmodelslim_logger
        self.metadata = {"total_size": 0}
        self.weight_map = {}
        pid = os.getpid()
        self.process = psutil.Process(pid)
        self.before_memory_info = self.process.memory_info()
        self.disable_names = set(cfg.disable_names)


    def convert_datafree_quant(self, weight):
        if not self.cfg.is_lowbit:
            calling_params = self.cfg.w_bit, self.cfg.w_sym, True, True, [False, 1000]
            quant_weight, _, weight_scale, weight_offset = \
                            init_weight_quant_normal(
                                weight, *calling_params, mm_tensor=self.cfg.mm_tensor, hqq=self.cfg.hqq
                                )
        else:
            fp_weight, weight_scale, _, weight_offset = \
                quant_one_weight_by_outliers_low_bit(
                    weight,
                    powerquant=self.cfg.nonuniform,
                    fraction=self.cfg.fraction,
                    num_bits=self.cfg.w_bit,
                    isolate_outlier_amax=False,
                    per_channel=not self.cfg.mm_tensor,
                    use_cuda=True if self.cfg.dev_type == "gpu" else False,
                    use_sigma=self.cfg.use_sigma,
                    sigma_factor=self.cfg.sigma_factor,
                    open_outlier=self.cfg.open_outlier,
                    group_size=self.cfg.group_size,
                    w_sym=self.cfg.w_sym,
                    use_hqq=self.cfg.hqq,
                )
            quant_weight,_ = fake_quantize_save(fp_weight, weight_scale, weight_offset,
                                                bit=self.cfg.w_bit,
                                                rount_opt=False,
                                                device=fp_weight.device,
                                                group_size=self.cfg.group_size)
            return quant_weight.contiguous(), weight_scale.contiguous(), weight_offset.contiguous()

    def add_params(self, param_dict, name, data, data_type):
        param_dict[name] = data
        self.quant_model_json_description.change_weight_type(name, data_type)
    
    def add_weight_file_map(self, weight, map_file):
        self.metadata["total_size"] += get_tensor_size(weight)
        for key in weight.keys():
            self.weight_map[key] = map_file

    def convert_safetensor(self, safetensor_path, disable_names):
        data = load_file(safetensor_path)
        self.param_dict = {}
        for name, weight in data.items():
            mod_name = get_prefix(name)
            if check_linear_weight(name):
                if mod_name in disable_names:
                    self.logger.info(f"disable weight: {name}")
                    self.add_params(self.param_dict, name, weight, QuantType.FLOAT)
                    self.disable_names.remove(mod_name)
                    continue
                quant_weight, weight_scale, weight_offset = self.convert_datafree_quant(weight.to(self.cfg.dev_type))
                self.add_params(self.param_dict, mod_name + '.weight', quant_weight.to(torch.int8), self.cfg.model_quant_type)
                self.add_params(self.param_dict, mod_name + '.weight_scale', weight_scale.cpu(), self.cfg.model_quant_type)
                self.add_params(self.param_dict, mod_name + '.weight_offset', weight_offset.cpu(), self.cfg.model_quant_type)
            else:
                self.add_params(self.param_dict, name, weight, QuantType.FLOAT)
        after_memory_info = self.process.memory_info()
        self.logger.info(f"memory usage: {(after_memory_info.rss - self.before_memory_info.rss) / 1024 ** 3} GB")

    def save_weight(self, output_path, safetensor_name):
        quant_model_weight_path = os.path.join(output_path, safetensor_name)
        quant_model_weight_path = get_valid_write_path(quant_model_weight_path, extensions=[".safetensors"])
        save_file(self.param_dict, quant_model_weight_path)
        self.param_dict = {}

    def save_json(self, output_path, json_name):
        quant_model_description_path = os.path.join(output_path, json_name)
        quant_model_description_path = get_valid_write_path(quant_model_description_path, extensions=[".json"])
        self.quant_model_json_description.save(quant_model_description_path)

    def save_index_map(self, output_path, index_name):
        index_path = os.path.join(output_path, index_name)
        index_path = get_valid_write_path(index_path, extensions=[".index"])
        index_json_dict = {"metadata": self.metadata, "weight_map": self.weight_map}
        json_safe_dump(index_json_dict, index_path, indent=2)

    def build_safetensors_name(self, weight_idx=None, total_weights=None):
        default_safetensors_name = f"quant_model_weight_{self.cfg.model_quant_type.lower()}.safetensors"
        if weight_idx is None or total_weights is None:
            return default_safetensors_name
        safetensors_name = default_safetensors_name
        part_file_name = safetensors_name.replace(".safetensors", f"-{weight_idx}-of-{total_weights}.safetensors")
        return part_file_name
    
    def build_json_name(self):
        json_name = f"quant_model_description_{self.cfg.model_quant_type.lower()}.json"
        return json_name
    
    def convert(self, model_path, save_path):
        save_path = get_write_directory(save_path, write_mode=0o750)
        disable_names = self.cfg.disable_names
        weight_files = get_safetensors_name(model_path)

        for file in tqdm(weight_files, desc="Converting safetensors"):
            file_path = file["file_path"]
            self.logger.info(f"Converting {file_path}")
            self.convert_safetensor(file_path, disable_names)
            part_file_name = self.build_safetensors_name(file["weight_idx"], file["total_weights"])
            self.add_weight_file_map(self.param_dict, part_file_name)
            self.logger.info(f"Saving {part_file_name}")
            self.save_weight(save_path, part_file_name)

        json_name = self.build_json_name()
        self.save_json(save_path, json_name)

        index_name = self.build_index_name() + ".index.json"
        self.save_index_map(save_path, index_name)

        if len(self.disable_names) > 0:
            self.logger.warning(f"Cannot find some disable names!")
        