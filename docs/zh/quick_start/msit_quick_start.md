# **推理开发工具快速入门**

<br>

## 1. 概述

MindStudio Inference Tools（msIT）为开发者提供一站式推理开发工具链，涵盖模型量化、精度调试、性能调优、服务化调优、可视化分析等能力。本文档作为快速入门入口，帮助您根据需求快速定位对应工具，并跳转至各工具的详细快速入门指南。

## 2. 工具快速导航

| 分类 | 工具 | 功能说明 | 快速入门链接 |
|------|------|----------|------------|
| 量化 | 模型量化工具（msModelSlim）| 提供模型压缩技术，通过降低模型权重和激活值的数值精度，有效减少模型的存储内存占用和计算需求。通常会将高位浮点数转换为低位定点数，从而直接减少模型权重的体积。模型量化工具的输入为能够正常运行的模型和数据，输出为一个可以使用的量化权重和量化因子。| [msModelSlim 快速入门](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/quick_start/quantization_quick_start.md) |
| 精度 | 精度调试工具（msProbe） | 包括精度数据采集（dump）和精度比对等功能，可以帮助定位模型推理过程中的精度问题。| [msProbe 快速入门](https://gitcode.com/Ascend/msprobe/tree/master/docs/zh/quick_start) |
| 性能 | 模型调优工具（msProf）| 支持采集与解析昇腾 AI 处理器的软硬件性能数据，帮助定位模型推理过程中的性能问题。| [msProf 快速入门](https://gitcode.com/Ascend/msprof/blob/master/docs/zh/quick_start/msprof_quick_start.md) |
| 性能 | 性能分析工具（msprof-analyze）| 基于采集的性能数据进行分析，提供昇腾设备性能瓶颈快速识别能力。| [msprof-analyze 快速入门](https://gitcode.com/Ascend/msprof-analyze/blob/master/docs/zh/quick_start/msprof-analyze_quick_start.md) |
| 性能 | 服务化调优工具（msServiceProfiler）| 提供性能剖析，清晰展示框架调度、模型推理等环节的表现，帮助用户快速找到性能瓶颈，从而有效提升服务性能。| [msServiceProfiler 快速入门](https://gitcode.com/Ascend/msserviceprofiler/blob/master/docs/zh/quick_start.md) |
| 性能 | 在线监控工具（msMonitor）| 一站式在线监控工具，支持落盘和在线性能数据采集，提供集群场景性能监测及定位能力。| [msMonitor 快速入门](https://gitcode.com/Ascend/msmonitor/blob/master/docs/zh/quick_start/msmonitor_quick_start.md) |
| 性能 | 预检工具（msprechecker）| 提供预检（precheck）、环境信息落盘（dump）和差异比对（compare）三大核心功能，并支持基于可扩展规则引擎的 `run` 和 `inspect` 子命令，允许用户自定义检查规则。帮助用户在昇腾环境中快速部署 AI 推理服务、复现性能基线、定位部署与性能问题。| [msprechecker 快速入门](../../../msprechecker/README.md) |
| 性能 | 性能建模工具（msModeling）| 专为昇腾 AI 处理器打造的神经网络推理性能仿真与分析框架，提供单模型性能仿真、服务级吞吐优化、服务化参数自动寻优与可视化分析能力，帮助开发者在无物理硬件或部署前期预测模型性能、识别瓶颈并优化配置。| [msModeling 快速入门](https://gitcode.com/Ascend/msmodeling/blob/master/docs/zh/quick_start/tensorcast_throughput_optimizer_quick_start.md) |
| 可视化 | 可视化工具（msInsight）| 将通过性能调优工具采集到的性能数据，使用 msInsight 进行可视化呈现，快速定位软、硬件性能瓶颈，提升AI任务性能分析的效率。| [msInsight 快速入门](https://gitcode.com/Ascend/msinsight/tree/master/docs/zh/quick_start) |
| 内存 | 内存工具（msMemScope）| 针对昇腾显存调试调优场景的专用工具，提供整网级多维度显存数据采集、自动诊断、优化分析能力。| [msMemScope 快速入门](https://gitcode.com/Ascend/msmemscope/blob/master/docs/zh/quick_start/quick_start.md) |
