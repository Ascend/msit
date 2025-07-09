# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import re
import os
from abc import ABC
from typing import List, Dict, Generator
from pathlib import Path

import yaml

from ascend_utils.common.security import get_valid_read_path
from msmodelslim import logger
from msmodelslim.app.naive_quantization.practice_data import (ConfigTask, CustomizedParams, Metadata,
                                                              load_specific_config)
from msmodelslim.app.naive_quantization.practice_interface import NaiveQuantizationInterface
from msmodelslim.utils.yaml_database import YamlDatabase

SUPPRORTED_QUANT_TYPES = ["w4a16", "w4a8", "w8a16", "w8a8", "w8a8s", "w8a8c8"]


def check_label(label, w_bit, a_bit, use_kv_cache, is_sparse):
    """Check if the label matches the quantization parameters"""
    if label.get('w_bit') == w_bit and label.get('a_bit') == a_bit:
        if is_sparse and not label.get('is_sparse'):
            return False
        if use_kv_cache and not label.get('kv_cache'):
            return False
        return True
    return False


def confirm_to_continue(prompt="No configuration found.",
                        error_msg="The corresponding configuration is not currently supported"):
    user_input = input(
        prompt + " Default configuration will be used. (Enter y to continue, otherwise it will exit): ").strip().lower()[:3]
    if user_input != 'y':
        raise ValueError(error_msg)
    return


def add_customized_config(config: ConfigTask, **kwargs) -> ConfigTask:
    """Modify the config based on kwargs"""
    if config.customized_config is None:
        config.customized_config = CustomizedParams()

    for key, value in kwargs.items():
        if hasattr(config.customized_config, key):
            setattr(config.customized_config, key, value)
    return config


class NaiveQuantization(NaiveQuantizationInterface, ABC):
    def __init__(self, config_dir: Path):
        self.sorted_task: Dict[str, List[ConfigTask]] = {}  # [modelType] -> List[ConfigTask]
        self.config_by_model_type: Dict[str, YamlDatabase] = {}

        self._load_config(config_dir)

        # Prepare sorted list for iteration
        for model_type, yaml_loader in self.config_by_model_type.items():
            self.sorted_task[model_type] = sorted(
                yaml_loader.config_by_name.values(),
                key=lambda x: (-x.metadata.score, x.metadata.config_id)
            )

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.abspath(os.path.join(cur_dir, "../practice_lab/Default/default.yaml"))
        self.default_config_path = get_valid_read_path(yaml_path)

    def get_best_practice(self,
                          model_type: str,
                          **kwargs) -> ConfigTask:
        model_type_belong = re.match(r'^[a-zA-Z]+', model_type)
        if model_type_belong is None:
            raise ValueError(f"Invalid model_type: {model_type}.")

        # Handle explicit config path
        if 'config_path' in kwargs and kwargs['config_path'] is not None:
            config_path = get_valid_read_path(kwargs['config_path'], extensions=['yaml', 'yml'])
            config = self.get_task_by_path(config_path)
            if config is None:
                raise ValueError(f"Configuration not found at {config_path}")
            logger.info(f"Naive Quant apply config_path: {config_path}")
            return add_customized_config(config, **kwargs)

        # Handle quant_type matching
        quant_type = kwargs.get('quant_type', None)
        if quant_type is None:
            confirm_to_continue(prompt="Neither config_path or quant_type.")
            config = self.get_task_by_path(self.default_config_path)
            return add_customized_config(config, **kwargs)

        if quant_type not in SUPPRORTED_QUANT_TYPES:
            confirm_to_continue(prompt="Quant_type is illegal.")
            config = self.get_task_by_path(self.default_config_path)
            return add_customized_config(config, **kwargs)

        # Parse quant_type parameters
        match_result = re.match(r'^w(\d+)a(\d+)(c?8?)(s?)$', quant_type)
        if not match_result:
            raise ValueError(f"Invalid quant_type format: {quant_type}")
        w_bit = int(match_result.group(1))
        a_bit = int(match_result.group(2))
        use_kv_cache = bool(match_result.group(3))
        is_sparse = bool(match_result.group(4))

        for config in self.iter_task(model_type_belong.group()):
            if quant_type:
                if model_type not in config.metadata.verified_model_types:
                    continue
                config_quant_type = config.metadata.label
                if not check_label(config_quant_type, w_bit, a_bit, use_kv_cache, is_sparse):
                    continue

            logger.info(f"Naive Quant apply config_id: {config.metadata.config_id}")
            return add_customized_config(config, **kwargs)

        confirm_to_continue(prompt=f"No matching configuration found for model_type={model_type}.")
        config = self.get_task_by_path(self.default_config_path)
        return add_customized_config(config, **kwargs)

    def get_task_by_name(self, model_type, config_id: str) -> ConfigTask:
        if not self.check_model_type(model_type):
            raise ValueError(f"Model type {model_type} not found")
        if not self.check_config_id(model_type, config_id):
            raise ValueError(f"ConfigTask {config_id} not found")
        return self.config_by_model_type[model_type].config_by_name[config_id]

    def get_task_by_path(self, config_path: str) -> ConfigTask:
        config_path = get_valid_read_path(config_path, extensions=['yaml', 'yml'])
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            config_task = ConfigTask(
                    metadata=Metadata(**config_data['metadata']),
                    specific=load_specific_config(config_data['spec'])
                )
        return config_task

    def iter_task(self, model_type) -> Generator[ConfigTask, None, None]:
        if model_type not in self.sorted_task:
            raise ValueError(f"Model type {model_type} not found")

        for config in self.sorted_task[model_type]:
            yield config

    def check_model_type(self, model_type: str) -> bool:
        """Check if model type exists"""
        return model_type in self.config_by_model_type

    def check_config_id(self, model_type: str, config_id: str) -> bool:
        """Check if config ID exists for the model type"""
        return config_id in self.config_by_model_type.get(model_type, {}).config_by_name

    def _load_config(self, config_dir: Path):
        for model_type_dir in config_dir.glob("*"):
            yaml_loader = YamlDatabase(model_type_dir)
            for configs in yaml_loader.load_config():
                for config_data in configs:
                    config = ConfigTask(
                        metadata=Metadata(**config_data['metadata']),
                        specific=load_specific_config(config_data['spec'])
                    )

                    # Keep the highest score for duplicate config_ids
                    existing = yaml_loader.config_by_name.get(config.metadata.config_id)
                    if not existing or config.metadata.score > existing.metadata.score:
                        yaml_loader.config_by_name[config.metadata.config_id] = config

            self.config_by_model_type[model_type_dir.name] = yaml_loader
            self.config_by_model_type[model_type_dir.name] = yaml_loader
