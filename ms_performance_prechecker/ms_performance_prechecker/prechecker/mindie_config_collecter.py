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
from ms_performance_prechecker.prechecker.register import RrecheckerBase, show_check_result, CheckResult
from ms_performance_prechecker.prechecker.utils import str_ignore_case, logger, set_log_level, deep_compare_dict
from ms_performance_prechecker.prechecker.utils import MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH
from ms_performance_prechecker.prechecker.utils import parse_mindie_server_config, parse_ranktable_file


class MindieConfigCollecter(RrecheckerBase):
    __checker_name__ = "MindieConfig"

    def collect_env(self, mindie_service_path=None, **kwargs):
        return parse_mindie_server_config(mindie_service_path)

class RankTableCollecter(RrecheckerBase):
    __checker_name__ = "RankTable"

    def collect_env(self, ranktable_file=None, **kwargs):
        self.ranktable_file = ranktable_file
        return parse_ranktable_file(ranktable_file)

    def key_checker(self, source_dict, target_key, prefix=""):
        if not target_key in source_dict:
            show_check_result(
                "configuration",
                "ranktable",
                CheckResult.ERROR,
                action=f"ranktable={self.ranktable_file} 中添加 {prefix}{target_key} 字段",
                reason=f"{prefix}{target_key} 为必需字段",
            )
    
    def do_precheck(self, ranktable, **kwargs):
        if not ranktable:
            return

        self.key_checker(source_dict=ranktable, target_key="server_count")
        self.key_checker(source_dict=ranktable, target_key="server_list")
        self.key_checker(source_dict=ranktable, target_key="version")
        self.key_checker(source_dict=ranktable, target_key="status")

        for server_id, server in enumerate(ranktable.get("server_list", [])):
            cur_prefix = f"server_list.{server_id}."
            self.key_checker(source_dict=server, target_key="server_id", prefix=cur_prefix)
            self.key_checker(source_dict=server, target_key="container_ip", prefix=cur_prefix)
            self.key_checker(source_dict=server, target_key="device", prefix=cur_prefix)

            for device_id, device in enumerate(server.get("device", [])):
                cur_prefix += f"device.{device_id}."
                self.key_checker(source_dict=device, target_key="device_id", prefix=cur_prefix)
                self.key_checker(source_dict=device, target_key="device_ip", prefix=cur_prefix)
                self.key_checker(source_dict=device, target_key="rank_id", prefix=cur_prefix)

mindie_config_collecter = MindieConfigCollecter()
ranktable_collecter = RankTableCollecter()