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

ENVS = [
  {
    "ENV": "CPU_AFFINITY_CONF",
    "SUGGESTION_VALUE": 2,
    "REASON": "CPU 细粒度绑核",
  },
  {
    "ENV": "NPU_MEMORY_FRACTION",
    "SUGGESTION_VALUE": 0.97,
    "REASON": "NPU内存占用比例，建议逐渐调高，但是太高会引起OOM",
  },
  {
    "ENV": "TASK_QUEUE_ENABLE",
    "SUGGESTION_VALUE": 2,
    "REASON": "配置task_queue 算子下发队列优化登记，可能导致运行中NPU内存峰值上升",
  },
  {
    "ENV": "HCCL_OP_EXPANSION_MODE",
    "SUGGESTION_VALUE": "AIV",
    "REASON": "配置通信算法的编排展开位置，代表通信算法的编排展开位置在 Device侧的 AI Vector Core 计算单元"\
    "（MindIE 2.0.T3 和 MindIE 2.0.T3.1 使能 AIV 会有崩溃风险，请不要设置它）",
  },
  {
    "ENV": "HCCL_DETERMINISTIC",
    "SUGGESTION_VALUE": False,
    "REASON": "关闭确定性计算，只有在调试的时候才会需要打开",
    "ALLOW_UNDEFINED": True,
  },
  {
    "ENV": "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT",
    "SUGGESTION_VALUE": True,
    "REASON": "host bound 时性能有提升 (某些局点该环境变量和内核版本升级可以二选一)",
  },
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
    "ENV": "ASCEND_LAUNCH_BLOCKING",
    "SUGGESTION_VALUE": "",
    "REASON": "关闭算子执行时启动同步模式（异步更快）",
  },
  {
    "ENV": "ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE",
    "SUGGESTION_VALUE": 2,
    "REASON": "wordkpace 内存分配算法选择，可通过选择不同的算法测试workspace分配情况",
  },
  {
    "ENV": "ATB_WORKSPACE_MEM_ALLOC_GLOBAL",
    "SUGGESTION_VALUE": 1,
    "REASON": "使用全局中间tensor 内存分配算法，会对中间tensor内存进行大小计算与分配",
  },
  {
    "ENV": "PYTORCH_NPU_ALLOC_CONF",
    "SUGGESTION_VALUE": "expandable_segments:True",
    "REASON": "使能内存池扩展段功能，既虚拟内存特性；设置为True,可以优化内存碎片对内存的占用",
  },
]
