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
import numpy as np
from ms_performance_prechecker.prechecker.register import register_checker, cached, answer
from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, SUGGESTION_TYPES


def get_dict_value_by_pos(dict_value, target_pos):
    cur = dict_value
    for kk in target_pos.split(":"):
        if not cur:
            cur = None
            break
        if isinstance(cur, list) and str.isdigit(kk):
            cur = cur[int(kk)]
        elif kk in cur:
            cur = cur[kk]
        else:
            cur = None
            break
    return cur


@register_checker()
def env_mindie_log_level_checker(mindie_service_config, check_type):
    mindie_log_level = os.getenv("MINDIE_LOG_LEVEL", "INFO")
    if mindie_log_level != "ERROR":
        answer(suggesion_type=SUGGESTION_TYPES.env, suggesion_item="MINDIE_LOG_LEVEL", action="set to ERROR", reason="大量的日志打印是十分耗时的行为，且在正常的服务过程中，不需要这些日志")

@register_checker()
def num_mem_size_checker(mindie_service_config, check_type):
    npu_mem_size_pos = "BackendConfig:ModelDeployConfig:ModelConfig:0:npuMemSize"

    npu_mem_size = get_dict_value_by_pos(mindie_service_config, npu_mem_size_pos)
    if npu_mem_size is not None and npu_mem_size != -1:
        print(f"获取目前 numMemSize 的值为 {npu_mem_size}, 并不是 -1")
        answer(config="npuMemSize", action="set to -1", reason="设置为-1，将由服务化自动根据剩余的显存数量，配置block数量，会尽量用满显存空间")
