# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["get_default_rule"]


import os
import platform
from typing import Dict

import yaml
from msguard.security import open_s

from msprechecker.prechecker.utils import get_npu_info
from msprechecker.prechecker.hccl_checker import NPU_DEVICES


RULE_MAPPING = {
    "env": "env_check_dsr1_pd",
    "user_config": "config_check_dsr1_pd"
}
ARCH_MAPPING = {
    "x86_64": "x86",
    "aarch64": "arm"
}


def get_default_rule(rule_type: str) -> Dict:
    if rule_type not in RULE_MAPPING:
        raise ValueError(
            f"Unsupported rule_type: {rule_type}. Supported types are: {', '.join(RULE_MAPPING)}"
        )

    npu_type = get_npu_info(True) or "A2"

    cur_dir = os.path.dirname(__file__)
    rule_file = f"{RULE_MAPPING[rule_type]}.yaml"

    if npu_type == "A2":
        arch = platform.machine().lower()
        if arch not in ARCH_MAPPING:
            raise ValueError(f"Unsupported architecture: {arch}")

        if arch == "x86_64" and len(NPU_DEVICES) != 16:
            raise ValueError(f"Unsupported type: 800I-A2 x86_64 but {len(NPU_DEVICES)} chips")

        arch_dir = ARCH_MAPPING[arch]
        rule_path = os.path.join(cur_dir, npu_type, arch_dir, rule_file)
    elif npu_type == "A3":
        rule_path = os.path.join(cur_dir, npu_type, rule_file)
    else:
        raise ValueError(f"Unsupported npu type: {rule_type}. Supported types are A2 or A3")

    if not os.path.isfile(rule_path):
        raise FileNotFoundError(f"Default rule file not found: {rule_path}")

    try:
        with open_s(rule_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in default rule file: {e}") from e

