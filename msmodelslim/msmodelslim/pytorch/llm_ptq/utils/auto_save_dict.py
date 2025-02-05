#Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import gc
import shutil
from collections import OrderedDict
from typing import Dict

import torch

from safetensors.torch import save_file

from ascend_utils.common.security import json_safe_dump, SafeWriteUmask, check_type, get_valid_write_path, get_write_directory
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription, QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save_utils import get_index_json


class AutoSaveDict(OrderedDict):
    GB_SIZE = 1 * 1024 * 1024 * 1024

    def __init__(self, cfg, max_gb_size=4, save_path='.', prefix="quant_model_weight"):
        super().__init__()
        self.saved_keys_map: Dict[str, str] = {}
        self.wait_save_keys: Dict[str, torch.Tensor] = {}
        self.max_size: int = max_gb_size * AutoSaveDict.GB_SIZE
        self.prefix: str = prefix
        self.total_size = 0
        self._wait_save_size = 0
        self._save_count = 0
        self._save_path = save_path
        self.cfg = cfg
        self.json_name = 'quant_model_json_description.json'
        # if for multi thread, this should be thread local variable
        self.model_quant_type = self.cfg.model_quant_type
        self.quant_model_json_description = QuantModelJsonDescription(self.cfg.model_quant_type,
                                                                      self.cfg.use_kvcache_quant,
                                                                      self.cfg.use_fa_quant)

    def __setitem__(self, key, value):
        check_type(value, torch.Tensor)
        tensor = value

        if tensor.device.type == 'meta':
            # TODO: this can work because module.weight and module.bias is not needed
            msmodelslim_logger.warning(f"Skip meta tensor {key}")
            return

        self.quant_model_json_description.change_weight_type(key, self.model_quant_type)

        tensor_size = tensor.numel() * tensor.element_size()
        self.wait_save_keys[key] = tensor.cpu()
        self.total_size += tensor_size
        self._wait_save_size += tensor_size
        msmodelslim_logger.debug(f"Add new tensor {key}, device: {tensor.device}, size: {tensor_size}, total: {self._wait_save_size}")

        if self._wait_save_size >= self.max_size:
            self.save_one_file()

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, value):
        self._save_path = get_write_directory(value, write_mode=0o750)

    def save_one_file(self):
        self._save_count += 1
        save_file_name = f"{self.prefix}-{self._save_count:05d}-of-00000.safetensors"
        full_save_file_name = os.path.join(self.save_path, save_file_name)
        full_save_file_name = get_valid_write_path(full_save_file_name, extensions=[".safetensors"])
        msmodelslim_logger.info(f"Start save {full_save_file_name}")
        with SafeWriteUmask(umask=0o377):
            save_file(self.wait_save_keys, full_save_file_name)
        self.saved_keys_map.update({key: save_file_name for key in self.wait_save_keys.keys()})
        self.wait_save_keys.clear()
        self._wait_save_size = 0
        msmodelslim_logger.info(f"End save {full_save_file_name}")

    def post_process(self):
        # save unsaved keys
        if self.wait_save_keys:
            self.save_one_file()

        # rename safetensors
        for i in range(self._save_count):
            src_file = os.path.join(self.save_path, f"{self.prefix}-{i + 1:05d}-of-00000.safetensors")
            dst_file = os.path.join(self.save_path, f"{self.prefix}-{i + 1:05d}-of-{self._save_count:05d}.safetensors")
            msmodelslim_logger.info(f"{src_file} -> {dst_file}")
            shutil.move(src_file, dst_file)

        # process safetensor index json
        for key in self.saved_keys_map.keys():
            self.saved_keys_map[key] = self.saved_keys_map[key].removesuffix('-of-00000.safetensors') + f'-of-{self._save_count:05d}.safetensors'

        # save index json
        index_json_dict = get_index_json(self.saved_keys_map, self.total_size)
        index_json_name = os.path.join(self.save_path, self.prefix + '.index.json')
        index_json_path = get_valid_write_path(index_json_name, extensions=[".json"])
        json_safe_dump(index_json_dict, index_json_path, indent=2)

        # safe json description
        quant_model_description_path = os.path.join(self.save_path, self.json_name)
        quant_model_description_path = get_valid_write_path(quant_model_description_path, extensions=[".json"])
        self.quant_model_json_description.save(quant_model_description_path)