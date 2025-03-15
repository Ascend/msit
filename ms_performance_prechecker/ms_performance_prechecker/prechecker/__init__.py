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
    "checkers",
]

from ms_performance_prechecker.prechecker.mindie_config_collecter import mindie_config_collecter
from ms_performance_prechecker.prechecker.env_checker import env_checker
from ms_performance_prechecker.prechecker.system_checker import system_checker

checkers = [
    mindie_config_collecter,
    env_checker,
    system_checker,
]
