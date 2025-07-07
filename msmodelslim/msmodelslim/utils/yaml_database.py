# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from pathlib import Path
from typing import Dict, Any
from ascend_utils.common.security.path import get_valid_read_path, MAX_READ_FILE_SIZE_4G

import yaml


class YamlDatabase:
    def __init__(self, config_dir: Path):
        """Load all configs from modelType subdirectories"""
        self.config_by_name: Dict[str, Any] = {}     # [config_id] -> ConfigType
        self.config_dir = config_dir

        if not self.config_dir.is_dir():
            raise ValueError(f"YamlDatabase need a directory, {self.config_dir} is not a directory")

    def load_config(self):
        """Load configuration from a YAML file"""
        for config_file in self.config_dir.glob("*.yaml"):
            config_file = get_valid_read_path(
                path=str(config_file),
                extensions=['.yaml', '.yml'],
                size_max=MAX_READ_FILE_SIZE_4G,
                check_user_stat=True,
                is_dir=False
            )
            with open(config_file, 'r') as f:
                yield yaml.safe_load_all(f)
