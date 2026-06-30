# MindStudio Inference Tools 使用指南

<br>

MindStudio Inference Tools（msIT）面向昇腾 AI 模型推理开发中的关键挑战，通过提供模型压缩、调试与调优等能力，高效解决推理效率低、资源开销大等问题，助力用户实现最优推理性能。

## 工具场景化推荐

本节以“我要做什么”为导向，旨在解决“当前问题应选用哪个工具”的困惑，结合典型场景为您推荐合适的工具，请点击对应工具链接跳转至其代码仓库，深入了解使用方法：

| 我要做什么 | 推荐工具 | 为什么推荐 |
|-----------|----------|------------|
| 我想要在部署场景前，评估模型性能、服务吞吐以及参数配置效果 |[**msModeling**](https://gitcode.com/Ascend/msmodeling)| 覆盖模型性能仿真、吞吐优化、服务级仿真和参数寻优，帮助提前评估性能瓶颈并优化部署配置 |
| 我想快速检查当前环境是否具备部署条件，以及复现对比两台设备的环境配置差异 |[**msprechecker**](https://gitcode.com/Ascend/msit/tree/master/msprechecker)|工具提供预检（precheck）、环境信息落盘（dump）和差异比对（compare）三大核心功能 |
| 我有一个开源模型，想生成量化权重 | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim) | 支持一键量化功能，提供量化模型的最佳实践库，快速生成量化权重 |
| 我量化后精度掉太多，想“自动找一套能达标的方案” | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim) | 给出精度目标即可，工具自动迭代：生成 yaml → 量化 → vLLM-Ascend 拉起服务 → AISBench 评估 → 不达标再换一套，直到命中目标。 |
| 我在量化精度调优过程中，不知道该回退哪些层 | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim) | 提供多种粒度，给出对量化敏感层的建议 |
| 我只想做权重格式/精度转换 | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim) | 无需加载模型代码、无需校准集、支持在CPU上运行，快速完成权重格式转换（如 BF16 转 MXFP8） |
| 我想要做训练或推理模型精度调试，定位精度问题 |[**msProbe**](https://gitcode.com/Ascend/msprobe)| 昇腾全场景精度工具，用于训练精度调试与问题定位|
| 我想使用命令行采集昇腾AI任务的性能数据 |[**msProf**](https://gitcode.com/Ascend/msprof)| 默认集成在CANN包，无需安装，命令行采集CANN与NPU性能数据      |
| 我想快速分析Profiling数据，分析性能瓶颈给出调优建议 | [**msprof-analyze**](https://gitcode.com/Ascend/msprof-analyze) | 基于采集得到的Profiling数据进行统计、比对和诊断，帮助定位计算、通信、调度及集群场景下的性能瓶颈 |
| 我想采集和解析性能数据，分析性能瓶颈给出调优建议|[**msServiceProfiler**](https://gitcode.com/Ascend/msserviceprofiler)|基于采集到的性能数据进行分析，帮助定位服务化框架、计算、通信以及调度的性能瓶颈|
| 我想分析训练或推理任务的显存使用，对显存调试调优 | [**msMemScope**](https://gitcode.com/Ascend/msmemscope) | 昇腾全场景显存数据采集，配套可视化、比对、拆解等分析能力 |
| 我想在定位性能问题时，可以通过图形化呈现的方式直观反映性能问题 | [**msInsight**](https://gitcode.com/Ascend/msinsight) | 可视化性能分析，覆盖系统、算子、服务化等场景，辅助完成性能诊断 |
| 我想在通过线监控轻量获取集群性能数据 | [**msMonitor**](https://gitcode.com/Ascend/msmonitor) | 一站式监控，支持落盘与在线采集，面向集群的监测与问题定位 |
