# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import re
import psutil
import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file, safe_open
from ascend_utils.common.security import get_valid_write_path, get_write_directory
from ascend_utils.common.security import json_safe_dump, check_type, check_dict_character

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    init_weight_quant_normal,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.lowbit.atomic_power_outlier import \
    quant_one_weight_by_outliers as quant_one_weight_by_outliers_low_bit
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    fake_quantize_save
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig


def check_linear_weight(name: str):
    """Check if the tensor is a linear weight based on its name.
    """
    return name.endswith(".weight") and "norm" not in name.lower() and "embed" not in name.lower()


def get_prefix(name, last_index=-1):
    """Extract prefix from the tensor name.
    """
    key_list = name.split(".")[:last_index]
    return ".".join(key_list)


def get_safetensors_name(path):
    """Get all safetensor files in the given path and sort them.
    """
    weight_files = []
    pattern = r"model-(\d+)-of-(\d+)\.safetensors"
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
    """Calculate the total size of all tensors in bytes.
    """
    total_size = 0
    for _, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            tensor_size = tensor.numel() * tensor.element_size()
            total_size += tensor_size
    return total_size


def find_prefix_key(dictionary, target_string):
    """Find a key in the dictionary that is a prefix of the target string.
    """
    for key in dictionary:
        if target_string.startswith(key):
            return key
    return None


def convert_datafree_weight(weight, cfg):
    """Convert a weight tensor using data-free quantization.
    
    Args:
        weight: Tensor to be quantized
        cfg: Configuration for quantization
        
    Returns:
        tuple: Quantized weight, scale, and offset tensors
    """
    if not cfg.is_lowbit:
        calling_params = cfg.w_bit, cfg.w_sym, True, True, [False, 1000]
        quant_weight, _, weight_scale, weight_offset = \
                        init_weight_quant_normal(
                            weight, *calling_params, mm_tensor=cfg.mm_tensor, hqq=cfg.hqq
                            )
    else:
        fp_weight, weight_scale, _, weight_offset = \
            quant_one_weight_by_outliers_low_bit(
                weight,
                powerquant=cfg.nonuniform,
                fraction=cfg.fraction,
                num_bits=cfg.w_bit,
                isolate_outlier_amax=False,
                per_channel=not cfg.mm_tensor,
                use_cuda=True if cfg.dev_type == "gpu" else False,
                use_sigma=cfg.use_sigma,
                sigma_factor=cfg.sigma_factor,
                open_outlier=cfg.open_outlier,
                group_size=cfg.group_size,
                w_sym=cfg.w_sym,
                use_hqq=cfg.hqq
            )
        quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset,
                                            bit=cfg.w_bit,
                                            round_opt=False,
                                            device=fp_weight.device,
                                            group_size=cfg.group_size)
    quant_weight = quant_weight.contiguous().to(torch.int8).cpu()
    weight_scale = weight_scale.contiguous().cpu()
    weight_offset = weight_offset.contiguous().cpu()

    return quant_weight, weight_scale, weight_offset


def check_cfgdict(cfgdict: dict):
    """Validate the configuration dictionary.
    """
    check_dict_character(cfgdict, param_name="cfgdict")
    cfg = cfgdict.pop("default", None)
    if cfg is None:
        raise ValueError("cfg must be a QuantConfig instance")
    check_type(cfg, QuantConfig, param_name="cfgdict['default']")
    for key, value in cfgdict.items():
        check_type(value, QuantConfig, param_name=f"cfgdict[{key}]")
    return cfg, cfgdict


class DataFreeConverter():
    def __init__(self, cfg_dict: dict):
        """Initialize the DataFreeConverter with configuration options.
        
        Args:
            cfg_dict: Dictionary containing quantization configurations
        """
        self.cfg, self.cfgdict = check_cfgdict(cfg_dict)
        self.quant_model_json_description = QuantModelJsonDescription(self.cfg.model_quant_type,
                                                                      self.cfg.use_kvcache_quant,
                                                                      self.cfg.use_fa_quant)
        self.logger = msmodelslim_logger
        self.metadata = {"total_size": 0}
        self.weight_map = {}
        pid = os.getpid()
        self.process = psutil.Process(pid)
        self.before_memory_info = self.process.memory_info()
        self.param_dict = {}

    def add_params(self, param_dict, name, data, data_type):
        """Add parameters to the parameter dictionary and update the model description.
        
        Args:
            param_dict: Dictionary to add parameters to
            name: Name of the parameter
            data: Parameter data (tensor)
            data_type: Type of the parameter
        """
        param_dict[name] = data
        self.quant_model_json_description.change_weight_type(name, data_type)
    
    def add_weight_file_map(self, weight, map_file):
        """Update metadata with weight information and mapping.
        
        Args:
            weight: Dictionary of weights to add
            map_file: File to map weights to
        """
        self.metadata["total_size"] += get_total_size(weight)
        for key in weight.keys():
            self.weight_map[key] = map_file

    def convert_safetensor(self, safetensor_path, load_weight=True):
        """Process a safetensor file by loading and quantizing tensors as needed.
        
        This method performs the following steps:
        1. Opens the safetensor file
        2. Processes each tensor in the file
        3. For linear weights, applies appropriate quantization based on configuration
        4. For other tensors, keeps them as floating point
        5. Tracks memory usage of the process
        
        Args:
            safetensor_path: Path to the safetensor file
            load_weight: Whether to actually load the weights or just process metadata
                         (set to False to avoid loading weights for files that will be skipped)
        """
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            self.param_dict = {}
            for name in f.keys():
                mod_name = get_prefix(name)
                weight = f.get_tensor(name) if load_weight else None
                if check_linear_weight(name):
                    cfg_name = find_prefix_key(self.cfgdict, mod_name)
                    if cfg_name:
                        cfg = self.cfgdict[cfg_name]
                    else:
                        cfg = self.cfg

                    if mod_name in cfg.disable_names:
                        self.logger.info(f"disable weight: {name}")
                        self.add_params(self.param_dict, name, weight, QuantType.FLOAT)
                        cfg.disable_names.remove(mod_name)
                        continue
                    if load_weight:
                        quant_weight, weight_scale, weight_offset = \
                            convert_datafree_weight(weight.to(cfg.dev_type), cfg)
                    else:
                        quant_weight, weight_scale, weight_offset = None, None, None

                    self.add_params(self.param_dict, mod_name + '.weight', \
                                    quant_weight, cfg.model_quant_type)
                    self.add_params(self.param_dict, mod_name + '.weight_scale', \
                                    weight_scale, cfg.model_quant_type)
                    self.add_params(self.param_dict, mod_name + '.weight_offset', \
                                    weight_offset, cfg.model_quant_type)
                else:
                    self.add_params(self.param_dict, name, weight, QuantType.FLOAT)
            after_memory_info = self.process.memory_info()
            self.logger.info(f"memory usage: {(after_memory_info.rss - self.before_memory_info.rss) / 1024 ** 3} GB")

    def save_weight(self, output_path, safetensor_name):
        """Save the quantized weights to a safetensor file.
        
        Args:
            output_path: Directory to save the file
            safetensor_name: Name of the safetensor file
        """
        quant_model_weight_path = os.path.join(output_path, safetensor_name)
        quant_model_weight_path = get_valid_write_path(quant_model_weight_path, extensions=[".safetensors"])
        save_file(self.param_dict, quant_model_weight_path)
        self.param_dict = {}

    def save_json(self, output_path, json_name):
        """Save the model description to a JSON file.
        """
        quant_model_description_path = os.path.join(output_path, json_name)
        quant_model_description_path = get_valid_write_path(quant_model_description_path, extensions=[".json"])
        self.quant_model_json_description.save(quant_model_description_path)

    def save_index_map(self, output_path, index_name):
        """Save the weight map and metadata to an index file.
        """
        index_path = os.path.join(output_path, index_name)
        index_path = get_valid_write_path(index_path, extensions=[".json"])
        index_json_dict = {"metadata": self.metadata, "weight_map": self.weight_map}
        json_safe_dump(index_json_dict, index_path, indent=2)

    def build_safetensors_name(self, weight_idx=None, total_weights=None):
        """Build the safetensors filename for a weight file.
        
        Args:
            weight_idx: Index of the weight file
            total_weights: Total number of weight files
            
        Returns:
            str: Name of the safetensor file
        """
        default_safetensors_name = f"quant_model_weight_{self.cfg.model_quant_type.lower()}.safetensors"
        if weight_idx is None or total_weights is None:
            return default_safetensors_name
        safetensors_name = default_safetensors_name
        part_file_name = safetensors_name.replace(".safetensors", f"-{weight_idx}-of-{total_weights}.safetensors")
        return part_file_name
    
    def build_json_name(self):
        """Build the JSON filename for the model description.
        """
        json_name = f"quant_model_description_{self.cfg.model_quant_type.lower()}.json"
        return json_name
    
    def convert(self, model_path, save_path):
        """Convert a model by processing all its safetensor files.
        
        This is the main entry point for the conversion process that:
        1. Prepares the output directory
        2. Gets and processes all safetensor files in the model directory
        3. For each file:
           a. Converts it if it doesn't already exist in the output
           b. Updates the weight map
        4. Saves model description and metadata
        
        The process includes:
        - Converting weights to lower precision (e.g., INT8)
        - Generating appropriate scales and offsets
        - Creating model description for inference
        - Creating a weight map for finding tensors
        
        Args:
            model_path: Path to the original model directory containing safetensor files
            save_path: Path to save the converted model
        """
        # Prepare the output directory
        save_path = get_write_directory(save_path, write_mode=0o750)
        weight_files = get_safetensors_name(model_path)

        for file in tqdm(weight_files, desc="Converting safetensors"):
            file_path = file["file_path"]
            self.logger.info(f"Converting {file_path}")
            part_file_name = self.build_safetensors_name(file["weight_idx"], file["total_weights"])
            if not os.path.exists(os.path.join(save_path, part_file_name)):
                self.convert_safetensor(file_path)
            else:
                self.convert_safetensor(file_path, load_weight=False)
            
            self.add_weight_file_map(self.param_dict, part_file_name)
            self.logger.info(f"Saving {part_file_name}")
            if not os.path.exists(os.path.join(save_path, part_file_name)):
                self.save_weight(save_path, part_file_name)
            else:
                self.logger.warning(f"{part_file_name} exists, skip!")

        json_name = self.build_json_name()
        self.save_json(save_path, json_name)

        index_name = self.build_safetensors_name() + ".index.json"
        self.save_index_map(save_path, index_name)


