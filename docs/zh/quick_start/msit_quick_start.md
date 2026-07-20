# **推理开发工具链快速入门**

<br>

## 1. 概述

### 1.1 功能简介

MindStudio Inference Tools（msIT）是面向昇腾 AI 平台的开源推理开发工具链，为开发者提供从模型部署到在线运维的全流程工具支持。工具链涵盖预检、量化、精度调试、性能调优和在线监控五大环节，包含多款专项工具，帮助开发者在昇腾硬件上高效完成大语言模型（含稠密模型与 MoE 架构）、多模态模型及传统模型的推理开发与优化。

### 1.2 核心能力

msIT 围绕推理开发的关键阶段，提供以下核心能力：

- **部署预检**：在正式部署前完成环境校验、连通性检测和推理结果比对，提前发现配置异常，降低部署风险。
- **模型压缩**：提供量化与压缩能力，支持一键量化和自动化精度迭代，在保障模型精度的前提下降低推理资源开销。
- **精度调试**：支持全场景精度采集与对比分析，帮助开发者快速定位精度偏差的根因。
- **性能调优**：覆盖数据采集、瓶颈分析、可视化诊断、服务化调优和建模仿真，提供从单算子到服务集群的多层级性能优化手段。
- **在线监控**：面向集群的实时性能监控与故障定位，保障推理服务的稳定运行。

### 1.3 本文用途

本文汇总工具链中各工具的功能定位与快速入门入口，帮助您根据当前的开发阶段和业务需求，快速选择合适的工具并完成上手操作。

## 2. 工具快速导航

| 类别 | 工具名称                                                                         | 功能简介                                                 |                                                            快速入门链接                                                            |
|:--:|:-----------------------------------------------------------------------------|:-----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------:|
| 预检 | **msPrechecker** | **【预检工具】** 支持环境预检、连通性预检及推理过程落盘和比对，帮助用户在部署前发现异常问题。    |                                           [点击查看](../../../msprechecker/README.md#快速入门)                                           |
| 量化 | **msModelSlim**                    | **【模型压缩】** 包含量化和压缩等推理优化技术，支持大语言稠密模型、MoE 模型、多模态模型等。   |          [点击查看](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/quick_start/quantization_quick_start.md)          |
| 精度 | **msProbe**                            | **【精度调试】** 昇腾全场景精度工具，用于精度调试与问题定位。                    |                          [点击查看](https://gitcode.com/Ascend/msprobe/tree/master/docs/zh/quick_start)                          |
| 性能 | **msProf**                              | **【模型调优】** 全场景性能调优底座，采集软硬件全栈性能数据，提升设备调优效率。           |               [点击查看](https://gitcode.com/Ascend/msprof/blob/master/docs/zh/quick_start/msprof_quick_start.md)                |
| 性能 | **msprof-analyze**              | **【性能分析】** 基于采集数据做性能分析，快速识别性能瓶颈。                     |       [点击查看](https://gitcode.com/Ascend/msprof-analyze/blob/master/docs/zh/quick_start/msprof-analyze_quick_start.md)        |
| 性能 | **msServiceProfiler**        | **【服务调优】** 支持请求调度、模型执行过程可视化，提升服务化性能分析效率。             |                   [点击查看](https://gitcode.com/Ascend/msserviceprofiler/blob/master/docs/zh/quick_start.md)                    |
| 性能 | **msMemScope**                     | **【内存调优】** 内存调优专用工具：整网级多维度内存采集，支持自动诊断与优化分析。          |                 [点击查看](https://gitcode.com/Ascend/msmemscope/blob/master/docs/zh/quick_start/quick_start.md)                 |
| 性能 | **msInsight**                        | **【可视调优】** 可视化性能分析，覆盖系统、算子、服务化等场景，辅助完成性能诊断。          |                         [点击查看](https://gitcode.com/Ascend/msinsight/tree/master/docs/zh/quick_start)                         |
| 性能 | **msModeling**                      | **【建模仿真】** 神经网络推理性能仿真框架，助力开发者在无硬件或部署前预测性能、识别瓶颈并优化配置。 | [点击查看](https://gitcode.com/Ascend/msmodeling/blob/master/docs/zh/quick_start/tensorcast_throughput_optimizer_quick_start.md) |
| 监控 | **msMonitor**                        | **【在线监控】** 一站式监控，支持落盘与在线采集，面向集群的监测与问题定位。             |            [点击查看](https://gitcode.com/Ascend/msmonitor/blob/master/docs/zh/quick_start/msmonitor_quick_start.md)             |
