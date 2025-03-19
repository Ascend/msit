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

import os
from ms_performance_prechecker.prechecker.register import RrecheckerBase
from ms_performance_prechecker.prechecker.utils import str_ignore_case, logger, set_log_level, deep_compare_dict
from ms_performance_prechecker.prechecker.utils import MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH
from ms_performance_prechecker.prechecker.utils import parse_mindie_server_config


class MindieConfigCollecter(RrecheckerBase):
    __checker_name__ = "MindieConfig"

    def collect_env(self, **kwargs):
        mindie_service_path = kwargs.get("mindie_service_path")
        return parse_mindie_server_config(mindie_service_path)


mindie_config_collecter = MindieConfigCollecter()
