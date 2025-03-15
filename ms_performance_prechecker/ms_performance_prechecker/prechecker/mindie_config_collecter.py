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
from ms_performance_prechecker.prechecker.utils import read_csv_or_json, get_next_dict_item


class MindieConfigCollecter(RrecheckerBase):
    __checker_name__ = "MindieConfig"

    def collect_env(self, **kwargs):
        mindie_service_path = kwargs.get("mindie_service_path")
        
        if mindie_service_path is None:
            mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)
        if not os.path.exists(mindie_service_path):
            logger.warning(f"mindie config.json: {mindie_service_path} not exists, will skip related checkers")
            return None

        mindie_service_config = read_csv_or_json(os.path.join(mindie_service_path, "conf", "config.json"))
        logger.debug(
            f"mindie_service_config: {get_next_dict_item(mindie_service_config) if mindie_service_config else None}"
        )
        return mindie_service_config
    
mindie_config_collecter = MindieConfigCollecter()