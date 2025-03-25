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

__all__ = [
    "CHECKERS",
]

from ms_performance_prechecker.prechecker.config_checker import (
    mindie_config_collecter,
    ranktable_collecter,
    model_config_collecter,
)
from ms_performance_prechecker.prechecker.env_checker import env_checker
from ms_performance_prechecker.prechecker.system_checker import system_checker
from ms_performance_prechecker.prechecker.hccl_checker import hccl_checker
from ms_performance_prechecker.prechecker.model_checker import model_size_checker, model_sha256_collecter
from ms_performance_prechecker.prechecker.utils import CHECKER_TYPES

CHECKERS = {
    CHECKER_TYPES.basic: [
        system_checker,
        env_checker,
        mindie_config_collecter,
        ranktable_collecter,
    ],
    CHECKER_TYPES.hccl: [hccl_checker],
    CHECKER_TYPES.model: [model_config_collecter, model_size_checker, model_sha256_collecter],
}

CHECKERS[CHECKER_TYPES.all] = [ii for key, checker in CHECKERS.items() for ii in checker if key != CHECKER_TYPES.all]

CHECKER_INFOS = {
    CHECKER_TYPES.basic: "checking env / system info",
    CHECKER_TYPES.hccl: "checking hccl connection status",
    CHECKER_TYPES.model: "checking or comparing model size and sha256sum value",
    CHECKER_TYPES.hardware: "checking CPU/NPU hardware computing capacity",
    CHECKER_TYPES.all: "checking all",
}

CHECKER_INFOS_STR = "; ".join([f"{kk} for {vv}" for kk, vv in CHECKER_INFOS.items()])
