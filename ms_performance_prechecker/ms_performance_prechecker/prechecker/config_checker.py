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
from ms_performance_prechecker.prechecker.utils import get_model_weight_path_from_mindie_server_config


class MindieConfigCollecter(RrecheckerBase):
    __checker_name__ = "MindieConfig"

    def collect_env(self, mindie_service_path=None, **kwargs):
        self.mindie_service_path = mindie_service_path
        return parse_mindie_server_config(mindie_service_path)

    def key_checker(self, source_dict, target_key, target_value=None, prefix=""):
        if target_key not in source_dict or (target_value is not None and source_dict[target_key] != target_value):
            show_check_result(
                "configuration",
                "mindie_service_config",
                CheckResult.ERROR,
                action=f"mindie_service={self.mindie_service_path} congig 中添加 {prefix}{target_key} 字段",
                reason=f"{prefix}{target_key} 需设置为 {target_value}" if target_value else f"{prefix}{target_key} 为必需字段",
            )

    def do_precheck(self, mindie_service_config, **kwargs):
        if not mindie_service_config:
            return

        backend_config = mindie_service_config.get("BackendConfig", {})
        self.key_checker(
            backend_config, target_key="multiNodesInferEnabled", target_value=True, prefix="BackendConfig."
        )
        self.key_checker(backend_config, target_key="interNodeTLSEnabled", target_value=false, prefix="BackendConfig.")

        server_config = mindie_service_config.get("ServerConfig", {})
        self.key_checker(server_config, target_key="httpsEnabled", target_value=false, prefix="ServerConfig.")
        self.key_checker(server_config, target_key="interCommTLSEnabled", target_value=false, prefix="ServerConfig.")


class RankTableCollecter(RrecheckerBase):
    __checker_name__ = "RankTable"

    def collect_env(self, ranktable_file=None, **kwargs):
        self.ranktable_file = ranktable_file
        return parse_ranktable_file(ranktable_file)

    def key_checker(self, source_dict, target_key, prefix=""):
        if target_key not in source_dict:
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


class MindieModelConfigCollecter(RrecheckerBase):
    __checker_name__ = "MindieConfig"

    def collect_env(self, mindie_service_path=None, **kwargs):
        self.mindie_service_path = mindie_service_path
        model_name, model_weight_path = get_model_weight_path_from_mindie_server_config(mindie_service_path)
        return {"modelName": model_name, "modelWeightPath": model_weight_path}

    def do_precheck(self, model_config, **kwargs):
        if not model_config:
            return
        model_name = model_config.get("modelName", None)
        model_weight_path = model_config.get("modelWeightPath", None)
        if not model_name:
            key_path = "BackendConfig.ModelDeployConfig.ModelConfig.modelName"
            action = f"mindie_service={self.mindie_service_path} congig 中添加 {key_path} 字段"
            show_check_result(
                "configuration", "mindie_service_config", CheckResult.ERROR, action=action, reason=f"{key_path} 为必需字段",
            )
        if not model_weight_path:
            key_path = "BackendConfig.ModelDeployConfig.ModelConfig.modelWeightPath"
            action = f"mindie_service={self.mindie_service_path} congig 中添加 {key_path} 字段"
            show_check_result(
                "configuration", "mindie_service_config", CheckResult.ERROR, action=action, reason=f"{key_path} 为必需字段",
            )


mindie_config_collecter = MindieConfigCollecter()
ranktable_collecter = RankTableCollecter()
mindie_model_config_collecter = MindieModelConfigCollecter()
