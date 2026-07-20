# msIT 工具选型指南

<br>

msIT 工具链包含多款专项工具，覆盖推理开发的各个阶段。面对具体任务时，精准选型往往比盲目尝试更具效率。

本文以 **”我要做什么”** 为导向，帮助您快速锁定最匹配的工具及直达入口。

<br>

## 场景化工具推荐

| 我要做什么 | 推荐工具 | 为什么推荐 |
|-----------|----------|------------|
| 我想要在部署场景前，评估模型性能、服务吞吐以及参数配置效果 |[**msModeling**](https://gitcode.com/Ascend/msmodeling/blob/master/README.md)| 覆盖模型性能仿真、吞吐优化、服务级仿真和参数寻优，帮助提前评估性能瓶颈并优化部署配置 |
| 我想快速检查当前环境是否具备部署条件，以及复现对比两台设备的环境配置差异 |[**msprechecker**](https://gitcode.com/Ascend/msit/blob/master/msprechecker/README.md)|工具提供预检（precheck）、环境信息落盘（dump）和差异比对（compare）三大核心功能 |
| 我有一个开源模型，想生成量化权重 | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) | 支持一键量化功能，提供量化模型的最佳实践库，快速生成量化权重 |
| 我量化后精度掉太多，想“自动找一套能达标的方案” | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) | 给出精度目标即可，工具自动迭代：生成 yaml → 量化 → vLLM-Ascend 拉起服务 → AISBench 评估 → 不达标再换一套，直到命中目标。 |
| 我在量化精度调优过程中，不知道该回退哪些层 | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) | 提供多种粒度，给出对量化敏感层的建议 |
| 我只想做权重格式/精度转换 | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) | 无需加载模型代码、无需校准集、支持在CPU上运行，快速完成权重格式转换（如 BF16 转 MXFP8） |
| 我想要做训练或推理模型精度调试，定位精度问题 |[**msProbe**](https://gitcode.com/Ascend/msprobe/blob/master/README.md)| 昇腾全场景精度工具，用于训练精度调试与问题定位|
| 我想使用命令行采集昇腾AI任务的性能数据 |[**msProf**](https://gitcode.com/Ascend/msprof/blob/master/README.md)| 默认集成在CANN包，无需安装，命令行采集CANN与NPU性能数据      |
| 我想快速分析Profiling数据，分析性能瓶颈给出调优建议| [**msprof-analyze**](https://gitcode.com/Ascend/msprof-analyze/blob/master/README.md) | 基于采集得到的Profiling数据进行统计、比对和诊断，帮助定位计算、通信、调度及集群场景下的性能瓶颈 |
| 我想采集和解析MindIE/vLLM/SGLang框架性能数据，分析性能瓶颈给出调优建议|[**msServiceProfiler**](https://gitcode.com/Ascend/msserviceprofiler/blob/master/README.md)|基于采集到的性能数据进行分析，帮助定位服务化框架、计算、通信以及调度的性能瓶颈|
| 我想分析训练或推理任务的显存使用，对显存调试调优 | [**msMemScope**](https://gitcode.com/Ascend/msmemscope/blob/master/README.md) | 昇腾全场景显存数据采集，配套可视化、比对、拆解等分析能力 |
| 我想在定位性能问题时，可以通过图形化呈现的方式直观反映性能问题 | [**msInsight**](https://gitcode.com/Ascend/msinsight/blob/master/README.md) | 可视化性能分析，覆盖系统、算子、服务化等场景，辅助完成性能诊断 |
| 我想在通过线监控轻量获取集群性能数据 | [**msMonitor**](https://gitcode.com/Ascend/msmonitor/blob/master/README.md) | 一站式监控，支持落盘与在线采集，面向集群的监测与问题定位 |
