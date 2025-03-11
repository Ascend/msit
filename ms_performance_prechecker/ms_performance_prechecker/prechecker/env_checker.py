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
from ms_performance_prechecker.prechecker.register import register_checker, cached, answer, record, CONTENT_PARTS
from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, logger, SUGGESTION_TYPES


ENVS = [
  {
    "ENV": "MINDIE_LOG_LEVEL",
    "SUGGESTION_VALUE": "ERROR",
    "REASON": "大量的日志打印是十分耗时的行为，且在正常的服务过程中，不需要这些日志",
    "ALLOW_UNDEFINED": True,
  },
  {
    "ENV": "ASCEND_GLOBAL_LOG_LEVEL",
    "SUGGESTION": 3,
    "REASON": "大量的日志打印是十分耗时的行为，且在正常的服务过程中，不需要这些日志",
    "ALLOW_UNDEFINED": True,
  },
  {
    "ENV": "TASK_QUEUE_ENABLE",
    "SUGGESTION_VALUE": 2,
    "REASON": "配置task_queue 算子下发队列优化登记，可能导致运行中NPU内存峰值上升",
  },
  {
    "ENV": "HCCL_OP_EXPANSION_MODE",
    "SUGGESTION_VALUE": "AIV",
    "REASON": "配置通信算法的编排展开位置，代表通信算法的编排展开位置在Device侧的AI Vector Core 计算单元（MindIE 2.0.T3 和 MindIE 2.0.T3.1 是能AIV会有崩溃风险，请不要设置它）",
  },
  {
    "ENV": "NPU_MEMORY_FRACTION",
    "SUGGESTION_VALUE": 0.97,
    "REASON": "NPU内存占用比例，建议逐渐调高，但是太高会引起OOM",
  },
  {
    "ENV": "CPU_AFFINITY_CONF",
    "SUGGESTION_VALUE": 2,
    "REASON": "CPU 细粒度绑核",
  },
  {
    "ENV": "ASCEND_LAUNCH_BLOCKING",
    "SUGGESTION_VALUE": "~",
    "REASON": "关闭算子执行时启动同步模式（异步更快）",
  },
  {
    "ENV": "HCCL_DETERMINISTIC",
    "SUGGESTION_VALUE": False,
    "REASON": "关闭确定性计算，只有在调试的时候才会需要打开",
  },
  {
    "ENV": "ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE",
    "SUGGESTION_VALUE": 2,
    "REASON": "wordkpace 内存分配算法选择，可通过选择不同的算法测试workspace分配情况",
  },
  {
    "ENV": "PYTORCH_NPU_ALLOC_CONF",
    "SUGGESTION_VALUE": "expandable_se,gments:True",
    "REASON": "使能内存池扩展段功能，既虚拟内存特性；设置为True,可以优化内存碎片对内存的占用",
  },
  {
    "ENV": "ATB_WORKSPACE_MEM_ALLOC_GLOBAL",
    "SUGGESTION_VALUE": 1,
    "REASON": "使用全局中间tensor 内存分配算法，会对中间tensor内存进行大小计算与分配",
  },
  {
    "ENV": "ATB_WORKSPACE_MEM_ALLOC_GLOBAL",
    "SUGGESTION_VALUE": 1,
    "REASON": "使用全局中间tensor 内存分配算法，会对中间tensor内存进行大小计算与分配",
  },
]


@register_checker()
def simple_env_checker(*_):
    for item in ENVS:
        env_item = item.get("ENV")
        env_value = os.getenv(env_item, "")
        env_suggest_value = item.get("SUGGESTION_VALUE") or ""
        suggest_reason = item.get("REASON", "")
        allow_undefined = item.get("ALLOW_UNDEFINED", False)
        if allow_undefined and not env_value:
            continue
        if str(env_value) == str(env_suggest_value):
            continue

        logger.info(f"{env_item}: {env_value} -> {env_suggest_value}")
        env_cmd = f"export {env_item}={env_suggest_value}" if env_suggest_value else f"unset {env_item}"
        answer(
            suggesion_type=SUGGESTION_TYPES.env,
            suggesion_item=env_item,
            action=env_cmd,
            reason=suggest_reason,
        )
        record(env_cmd, part=CONTENT_PARTS.after)

        pre_env = f"export {env_item} {env_value}" if env_value else f"unset {env_item}"
        record(pre_env, part=CONTENT_PARTS.before)
