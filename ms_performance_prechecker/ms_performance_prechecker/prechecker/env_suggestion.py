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

"""
配置指南：
ENV_SUGGESTIONS 配置项说明：
  * ENV: 环境变量名
  * SUGGESTIONS: 建议列表
      - VALUE: 环境变量值（如果不建议配置，可以配置为None，表示不建议配置该环境变量）
      - SUGGESTION: 建议配置该值
          + VERSION_LIST: 哪些版本建议配置，不配置表示所有版本适用
          + REASON: 建议值对应的原因
      - NOT_SUGGESTION: 不建议配置该值（优先级高）
          + VERSION_LIST: 哪些版本不建议配置，不配置表示所有版本适用
          + REASON: 不建议的原因

简化配置：
  ENV: 环境变量名
  SUGGESTION_VALUE: 建议值
  REASON: 建议原因


建议样例1： 环境变量 ENV_SUGGEST_DEMO, 一般情况下建议配置为 VALUE1
建议配置1：
  {
    "ENV": "ENV_SUGGEST_DEMO",
    "SUGGESTION_VALUE": "VALUE1",
    "REASON": "建议配置为VALUE1",
  }
建议样例2： 背景同样例1，但是在版本 1.0.0 版本不建议配置为 VALUE1（建议不配置）
建议配置2：
  {
    "ENV": "ENV_SUGGEST_DEMO",
    "SUGGESTIONS": [
      {
        "VALUE": "VALUE1",
        "SUGGESTION": {
          "REASON": "建议配置为VALUE1",
        },
        "NOT_SUGGESTION": {
          "VERSION_LIST": {"Ascend-mindie": ["1.0.0"]},
          "REASON": "不建议配置为VALUE1",
        }
      }
    ]
  }

建议样例3： 背景同样例2，但是在版本 1.0.2 版本建议配置为 VALUE2
建议配置3：
  {
    "ENV": "ENV_SUGGEST_DEMO",
    "SUGGESTIONS": [
      {
        "VALUE": "VALUE1",
        "SUGGESTION": {
          "REASON": "建议配置为VALUE1",
        },
        "NOT_SUGGESTION": {
          "VERSION_LIST": {"Ascend-mindie": ["1.0.0"]},
          "REASON": "不建议配置为VALUE1",
        }
      },
      {
        "VALUE": "VALUE2",
        "SUGGESTION": {
          "VERSION_LIST": {"Ascend-mindie": ["1.0.2"]},
          "REASON": "建议配置为 VALUE2",
        },
    ]
  }

"""
ENV_SUGGESTIONS = [
    {
        "ENV": "CPU_AFFINITY_CONF",
        "SUGGESTION_VALUE": 2,
        "REASON": "开启CPU细粒度绑核，可以优化算子下发。",
    },
    {
        "ENV": "NPU_MEMORY_FRACTION",
        "SUGGESTION_VALUE": 0.97,
        "REASON": "NPU内存占用比例，建议逐渐调高，但是太高会引起OOM，经验值为0.97。",
    },
    {
        "ENV": "TASK_QUEUE_ENABLE",
        "SUGGESTION_VALUE": 2,
        "REASON": "算子下发队列使用内存并发，可以提升性能，可能导致运行中NPU内存峰值上升。",
    },
    {
        "ENV": "HCCL_OP_EXPANSION_MODE",
        "SUGGESTIONS": [
            {
                "VALUE": "AIV",
                "SUGGESTION": {
                    "REASON": "配置通信算法的编排展开位置，代表通信算法的编排展开位置在 Device侧的 AI Vector Core 计算单元。"
                        "（但是在 '2.0.T3', '2.0.T3.1', '2.0.T6' 版本中，使能 AIV 会有崩溃风险，请不要设置它"
                },
                "NOT_SUGGESTION": {
                    "VERSION_LIST": {"Ascend-mindie": ["2.0.T3", "2.0.T3.1", "2.0.T6"]},
                    "REASON": "使能 AIV 会有崩溃风险，请不要设置它",
                },
            },
        ],
    },
    {
        "ENV": "HCCL_DETERMINISTIC",
        "SUGGESTION_VALUE": ["false", None],
        "REASON": "关闭确定性计算，一般情况下无需开启确定性计算，当模型多次执行结果不同或者精度调优时，"
            "可通过此环境变量开启确定性计算进行辅助调试调优，但开启确定性计算后，算子执行时间会变慢，导致性能下降。",
    },
    {
        "ENV": "HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT",
        "SUGGESTION_VALUE": "TRUE",
        "REASON": "当通信算子下发性能Host Bound时，开发者可通过此环境变量设置通过PCIe Direct的方式提交RDMA任务，提升通信算子下发性能。",
    },
    {
        "ENV": "MINDIE_LOG_LEVEL",
        "SUGGESTION_VALUE": ["ERROR", None],
        "REASON": "控制MindIE组件日志级别，大量的日志打印会影响程序性能，建议提高日志级别。",
    },
    {
        "ENV": "ASCEND_GLOBAL_LOG_LEVEL",
        "SUGGESTION_VALUE": [3, None],
        "REASON": "控制昇腾应用类日志级别，大量的日志打印会影响程序性能，建议提高日志级别。",
    },
    {
        "ENV": "ASCEND_LAUNCH_BLOCKING",
        "SUGGESTION_VALUE": [0, None],
        "REASON": "同步模式用于算子调试，通常使用异步方式执行，性能更好。",
    },
    {
        "ENV": "ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE",
        "SUGGESTION_VALUE": 2,
        "REASON": "workspace 内存分配算法选择，可通过选择不同的算法测试workspace分配情况。",
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
