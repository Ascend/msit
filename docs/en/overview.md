# Brief Introduction

MindStudio Inference Tools (msIT) provides users with functions such as model compression, model debugging and optimization commonly used in large-scale model and traditional model inference development, and supports performance optimization in the inference service scenario. This feature helps users achieve optimal inference performance.

## Function Description

As the unified inference development tool chain of the Ascend platform, it contains tools such as model quantization, precision debugging, and performance tuning. You can select a tool to view detailed information and perform model inference.

### Performance Tool

- [**msProf (MindStudio Profiler)**](https://gitcode.com/Ascend/msprof/blob/master/docs/en/quick_start/msprof_quick_start.md)<br>
    Data collection tool: builds basic capabilities for Ascend performance optimization in all scenarios and collects CANN and NPU performance data, improving Ascend performance optimization efficiency.

- [**msMonitor (MindStudio Monitor)**](https://gitcode.com/Ascend/msmonitor/blob/master/docs/en/quick_start/msmonitor_quick_start.md)<br>
    Online monitoring tool: One-stop online monitoring tool, which supports trace data dumping to the drive and online performance data collection and provides performance monitoring and fault locating capabilities in cluster scenarios.

- [**msServiceProfiler (MindStudio Service Profiler)**](https://gitcode.com/Ascend/msserviceprofiler/blob/master/docs/en/quick_start.md)<br>
    Service-oriented performance optimization tool: It is a service-oriented performance optimization tool with Ascend affinity. It supports request scheduling and model execution visualization, improving service-oriented performance analysis efficiency.

- [**msprechecker (MindStudio Prechecker Tool)**](https://gitcode.com/Ascend/msit/blob/master/msprechecker/README_EN.md)<br>
    Precheck tool: The msprechecker provides three core functions: precheck, environment information dump, and difference comparison. This feature helps you quickly deploy AI inference services in the Ascend environment, reproduce the performance baseline, and locate deployment and performance issues.

- [**msprof-analyze (MindStudio Profiler Analyze)**](https://gitcode.com/Ascend/msprof-analyze/blob/master/docs/en/quick_start/msprof-analyze_quick_start.md)<br>
    Ascend performance analysis tool: Analyzes collected performance data and quickly identifies performance bottlenecks of Ascend devices.

- [**msInsight (MindStudio Insight)**](https://gitcode.com/Ascend/msinsight/blob/master/docs/en/user_guide/overview.md)<br>
    MindStudio Insight: supports multi-dimensional performance analysis in multiple scenarios, such as system-level, operator-level, and servitization, and in-depth performance data analysis, helping developers complete performance diagnosis.

- [**msModeling (MindStudio Modeling)**](https://gitcode.com/Ascend/msmodeling/blob/master/README_EN.md)<br>
    Ascend AI model performance modeling and simulation tool: Neural network inference performance simulation and analysis framework designed for Ascend AI processors. It provides single-model performance simulation, service-level throughput optimization, service-oriented parameter auto-optimization, and visual analysis capabilities, helping developers predict model performance, identify bottlenecks, and optimize configurations without physical hardware or in the pre-deployment phase.

### Precision Tool

- [**msProbe (MindStudio Probe)**](https://gitcode.com/Ascend/msprobe/blob/master/docs/en/user_guide/dump/mindspore_dump_quick_start.md)<br>
    Precision debugging tool: It is a tool package used during precision debugging during model development. It is a precision tool chain for all scenarios provided by Ascend. It helps users improve the efficiency of model precision locating.

- [**msMemScope (MindStudio MemScope)**](https://gitcode.com/Ascend/msmemscope/blob/master/docs/en/quick_start/quick_start.md)<br>
    Memory tool: It is a dedicated tool for Ascend memory debugging and optimization. It provides network-wide multi-dimensional accelerator memory data collection, automatic diagnosis, optimization, and analysis capabilities.

### Quantization Tool

- [**msModelSlim (MindStudio ModelSlim)**](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/en/quick_start/quantization_quick_start.md)<br>
    Model compression tool: Ascend model compression tool, an affinity compression tool that aims at acceleration, compression, and Ascend. It includes a series of inference optimization technologies such as quantization and compression, and supports large language dense models, MoE models, multi-modal understanding models, and multi-modal generation models.
