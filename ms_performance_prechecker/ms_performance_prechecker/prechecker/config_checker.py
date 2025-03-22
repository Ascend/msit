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
from ms_performance_prechecker.prechecker.utils import get_model_path_from_mindie_config
from ms_performance_prechecker.prechecker.utils import is_deepseek_model, read_csv_or_json


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
                action=f"mindie_service={self.mindie_service_path} config 中添加 {prefix}{target_key} 字段",
                reason=f"{prefix}{target_key} 需设置为 {target_value}" if target_value else f"{prefix}{target_key} 为必需字段",
            )

    def do_precheck(self, mindie_service_config, **kwargs):
        if not mindie_service_config:
            return

        server_config = mindie_service_config.get("ServerConfig", {})
        self.key_checker(server_config, target_key="httpsEnabled", target_value=False, prefix="ServerConfig.")
        self.key_checker(server_config, target_key="interCommTLSEnabled", target_value=False, prefix="ServerConfig.")

        backend_config = mindie_service_config.get("BackendConfig", {})
        self.key_checker(
            backend_config, target_key="multiNodesInferEnabled", target_value=True, prefix="BackendConfig."
        )
        self.key_checker(backend_config, target_key="interNodeTLSEnabled", target_value=False, prefix="BackendConfig.")

        model_config = backend_config.get("ModelDeployConfig", {}).get("ModelConfig", [{}])
        cur_prefix = "BackendConfig.ModelDeployConfig.ModelConfig."
        self.key_checker(model_config[0], target_key="modelName", prefix=cur_prefix)
        self.key_checker(model_config[0], target_key="modelWeightPath", prefix=cur_prefix)


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


class ModelConfigCollecter(RrecheckerBase):
    __checker_name__ = "ModelConfig"

    def collect_env(self, mindie_service_path=None, **kwargs):
        model_name, model_weight_path = get_model_path_from_mindie_config(mindie_service_path=mindie_service_path)

        if not model_name or not model_weight_path:
            return None

        model_config, model_config_path = {}, os.path.join(model_weight_path, "config.json")
        if os.path.exists(model_config_path):
            model_config = read_csv_or_json(model_config_path)
        self.model_config_path = model_config_path
        logger.debug(f"ModelConfigCollecter model_name={model_name} model_config={model_config}")
        return {"model_name": model_name, "model_config": model_config}

    def do_precheck(self, model_info, **kwargs):
        if not model_info:
            return
        model_name, model_config = model_info.get("model_name", None), model_info.get("model_config", None)
        if not model_name or not model_config:
            return

        cur_model_type = model_config.get("model_type")
        if is_deepseek_model(model_name) and cur_model_type != "deepseekv2":
            action = f'MindIE配置 model_name={model_name}, 需在模型配置文件 {self.model_config_path} 中设置 model_type="deepseekv2"'
            show_check_result(
                "configuration",
                "model",
                CheckResult.ERROR,
                action=action,
                reason=f"当前配置 model_type={model_type} 不匹配 deepseek 模型",
            )
        else:
            show_check_result("configuration", "model", CheckResult.OK)


mindie_config_collecter = MindieConfigCollecter()
ranktable_collecter = RankTableCollecter()
model_config_collecter = ModelConfigCollecter()