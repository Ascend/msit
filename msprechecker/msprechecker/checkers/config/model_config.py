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

from ...utils import get_pkg_version
from ..base import BaseChecker


class ModelConfigChecker(BaseChecker):
    def __init__(self, *, error_handler=None, rule_manager=None):
        super().__init__(error_handler=error_handler, rule_manager=rule_manager)
        self.error_handler.type = "model config"

    def _check(self, results):
        cur_transformers_version = get_pkg_version("transformers")

        torch_dtype = results.get("torch_dtype", "<missing>")
        if torch_dtype != "float16":
            self.error_handler.add_error(
                path="torch_dtype",
                actual=torch_dtype,
                expected="float16",
                reason="部分模型的算子可能不支持 bfloat16, 请确保当前模型算子支持 bfloat16",
                severity="low",
            )

        transformers_version = results.get("transformers_version")
        if transformers_version and (
            not cur_transformers_version
            or transformers_version > cur_transformers_version
        ):
            self.error_handler.add_error(
                path="transformers_version",
                actual=transformers_version,
                expected=cur_transformers_version,
                reason=f'当前机器的 "transformers" 的版本（{cur_transformers_version}）如果小于配置文件要求版本，会导致服务启动失败',
                severity="medium",
            )

        model_type = results.get("model_type", "<missing>")
        if (
            model_type.startswith("deepseek")
            and "deepseek_" not in results["model_type"]
            and self.rule_manager.framework == "vllm"
        ):
            self.error_handler.add_error(
                path="model_type",
                actual=model_type,
                expected="deepseek_xxxx",
                reason='vllm 框架下, DeepSeek 系列模型的 "model_type" 需要添加下划线，如："deepseek_v2"，否则会导致服务部署失败',
                severity="high",
            )
