# 简介

MindStudio Inference Tools（MindStudio昇腾推理工具链，msIT），为用户提供大模型与传统模型推理开发中常用的模型压缩、模型调试调优等功能，支持推理服务化场景下的性能调优能力，帮助用户达到最优的推理性能。

## 功能介绍

作为昇腾平台的统一推理开发工具链，包含模型量化、精度调试和性能调优等工具，可根据下方的工具介绍，选择相应工具查看具体信息，进行模型推理。

### 性能工具

- [**msProf（MindStudio Profiler）**](https://gitcode.com/Ascend/msprof/blob/master/docs/zh/quick_start/msprof_quick_start.md)<br>
    **数据采集工具**：构建昇腾全场景性能调优基础能力，支持采集CANN和NPU性能数据，提升昇腾设备性能调优效率。

- [**msMonitor（MindStudio Monitor）**](https://gitcode.com/Ascend/msmonitor/blob/master/docs/zh/quick_start/msmonitor_quick_start.md)<br>
    **在线监控工具**一站式在线监控工具，支持落盘和在线性能数据采集，提供集群场景性能监测及定位能力。

- [**msServiceProfiler（MindStudio Service Profiler）**](https://gitcode.com/Ascend/msserviceprofiler/blob/master/docs/zh/quick_start.md)<br>
    **服务化性能调优工具**：昇腾亲和的服务化性能调优工具，支持请求调度、模型执行过程可视化，提升服务化性能分析效率。

- [**msprechecker（MindStudio Prechecker Tool）**](https://gitcode.com/Ascend/msit/blob/master/msprechecker/README.md)<br>
    **预检工具**：msprechecker提供预检（precheck）、环境信息落盘（dump）和差异比对（compare）三大核心功能，帮助用户在昇腾（Ascend）环境中快速部署 AI 推理服务、复现性能基线、定位部署与性能问题。

- [**msprof-analyze（MindStudio Profiler Analyze）**](https://gitcode.com/Ascend/msprof-analyze/blob/master/docs/zh/quick_start/msprof-analyze_quick_start.md)<br>
    **昇腾性能分析工具**：基于采集的性能数据进行分析，提供昇腾设备性能瓶颈快速识别能力。

- [**msInsight（MindStudio Insight）**](https://gitcode.com/Ascend/msinsight/blob/master/docs/zh/user_guide/overview.md)<br>
    **MindStudio Insight可视化工具**：支持系统级、算子级、服务化等多场景多维度性能分析，深度剖析性能数据，帮助开发者完成性能诊断。

- [**msModeling (MindStudio Modeling)**](https://gitcode.com/Ascend/msmodeling/blob/master/README.md)<br>
    **昇腾 AI 模型性能建模与仿真工具**：专为昇腾 AI 处理器打造的神经网络推理性能仿真与分析框架，提供单模型性能仿真、服务级吞吐优化、服务化参数自动寻优与可视化分析能力，帮助开发者在无物理硬件或部署前期预测模型性能、识别瓶颈并优化配置。

### 精度工具
    
- [**msProbe（MindStudio Probe）**](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/mindspore_dump_quick_start.md)<br>
    **精度调试工具**：模型开发精度调试环节使用的工具包，是针对昇腾提供的全场景精度工具链，帮助用户提高模型精度定位效率。

- [**msMemScope（MindStudio MemScope）**](https://gitcode.com/Ascend/msmemscope/blob/master/docs/zh/quick_start/quick_start.md)<br>
    **内存工具**：针对昇腾显存调试调优场景的专用工具，提供整网级多维度显存数据采集、自动诊断、优化分析能力。

### 量化工具

- [**msModelSlim（MindStudio ModelSlim）**](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/quick_start/quantization_quick_start.md)<br>
    **模型压缩工具**：昇腾模型压缩工具，一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。包含量化和压缩等一系列推理优化技术，支持大语言稠密模型、MoE模型、多模态理解模型、多模态生成模型等。
    