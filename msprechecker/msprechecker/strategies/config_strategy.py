# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import json
import os
import re
from typing import Optional

from .weight_strategy import Weight
from .base_strategy import CollectStrategyGroup, CollectStrategy
from .env_strategy import Env
from ..utils import PreFetch, Framework, Utils, LOGGER


class Configs(CollectStrategyGroup):
    def __init__(
            self,
            name="configs",
            strategies=None,
    ):
        super().__init__(name, strategies)

    @staticmethod
    def replace_weight_dir_vllm(content: str, new_weight_dir: str) -> str:
        weight_pattern = re.compile(
            r"""(vllm\s+serve\s+(?:["']?))([^\s"']+)|(--model(?:["']?\s*[=\s]?\s*(?:["']?)))([^\s"']+)""")

        def replace_match(match):
            if match.group(1):
                return f"{match.group(1)}{new_weight_dir}"
            elif match.group(3):
                return f"{match.group(3)}{new_weight_dir}"
            return match.group(0)

        LOGGER.info(f"Replace weight dir with provided path {new_weight_dir}.")
        return weight_pattern.sub(replace_match, content)

    @staticmethod
    def replace_weight_dir_mindie(data: dict, new_weight_dir: str):
        if not isinstance(data, dict):
            LOGGER.info('Config file is not a dict, skip.')
            return
        try:
            data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
                "modelWeightPath"
            ] = new_weight_dir
        except KeyError as e:
            LOGGER.info('Missing required configuration key in MIES config file, skip.')

    @staticmethod
    def dump_boot_script_mindie(target_data):
        script_content = f"""#bin/bash
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
source {Env().env_path}

/usr/local/Ascend/mindie/latest/mindie-service/bin/mindieservice_daemon
        """
        script_path = (f"sync-{target_data.get('image', {}).get('image_type', '')}-"
                       f"{target_data.get('timestamp', '')}-boot.sh")
        Utils.dump_file(script_path, script_content)

    def sync(self, target_data: dict):
        super().sync(target_data)
        framework = PreFetch.get_framework()
        for config_path, config_data in self._target.items():
            if framework == Framework.MINDIE and config_path.endswith(".json"):
                config_dir = os.path.dirname(config_path)
                if not os.path.exists(config_dir):
                    Utils.log_error_and_exit(f"Config directory {config_dir} does not exist")
                self.replace_weight_dir_mindie(config_data, Weight().get_weight_dir())
                Utils.dump_file(config_path, json.dumps(config_data))
            else:
                # for vLLM and SGLang, save the config file in the current directory and add source env on the top
                source_env_txt = "source {}\n".format(Env().env_path)
                new_file_name = (f"sync-{target_data.get('image', {}).get('image_type', '')}-"
                                 f"{target_data.get('timestamp', '')}-{os.path.basename(config_path)}")
                new_config_data = self.replace_weight_dir_vllm(config_data, Weight().get_weight_dir())
                Utils.dump_file(new_file_name, source_env_txt + new_config_data)
        if framework == Framework.MINDIE:
            self.dump_boot_script_mindie(target_data)


class Config(CollectStrategy):
    def __init__(self, name, *, config_path):
        super().__init__(name)
        self._config_path = config_path
        self._processor = {
            ".json": self._process_json,
            ".yaml": self._process_yaml,
            ".yml": self._process_yaml,
            ".sh": self._process_shell,
        }

    def _process_json(self, content):
        LOGGER.debug('Processing JSON configuration file: %r', self._config_path)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            Utils.log_error_and_exit("Failed to parse JSON configuration file {}: {}".format(self._config_path, str(e)))
            return content

    def _process_yaml(self, content):
        LOGGER.debug('Processing YAML configuration file: {}'.format(self._config_path))
        import yaml

        try:
            if "---" in content:
                return list(yaml.safe_load_all(content))
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            Utils.log_error_and_exit("Failed to parse YAML configuration file {}: {}".format(self._config_path, str(e)))
            return content

    def _process_shell(self, content):
        LOGGER.debug('Processing shell configuration file: %s', self._config_path)
        return content

    def _read_file(self) -> Optional[str]:
        if not self._config_path:
            LOGGER.warning("Configuration path is empty or not provided")
            return None
        if not os.path.isfile(self._config_path):
            LOGGER.warning("Configuration file %r not found", self._config_path)
            return None
        try:
            with open(self._config_path) as f:
                return f.read()
        except OSError:
            Utils.log_error_and_exit("Failed to read configuration file {}", self._config_path)
            return None

    def _parse(self, content: str):
        ext = os.path.splitext(self._config_path)[-1]
        processor = self._processor.get(ext)
        if processor is None:
            LOGGER.warning(
                "Unsupported configuration file format: {}", self._config_path
            )
            return content
        return processor(content)

    def execute(self):
        content = self._read_file()
        if content is None:
            return None
        return self._parse(content)
