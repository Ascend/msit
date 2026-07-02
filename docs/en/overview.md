# Brief Introduction

MindStudio Inference Tools (msIT) provides users with functions such as model compression, model debugging and optimization commonly used in large-scale model and traditional model inference development, and supports performance optimization in the inference service scenario. This feature helps users achieve optimal inference performance.

## Function Description

As the unified inference development tool chain of the Ascend platform, it contains tools such as model quantification, precision debugging, and performance tuning. You can select a tool to view detailed information and perform model inference.

### Performance tool

 * [**msProf (MindStudio Profiler)**](https://gitcode.com/Ascend/msprof/blob/26.0.0/docs/zh/quick_start.md)    
    Data collection tool: builds basic capabilities for Ascend performance optimization in all scenarios and collects CANN and NPU performance data, improving Ascend performance optimization efficiency.
 * [**msMonitor (MindStudio Monitor)**](https://gitcode.com/Ascend/msmonitor/blob/26.0.0/docs/zh/quick_start.md)    
    Online monitoring toolOne-stop online monitoring tool, which supports flushing and online performance data collection and provides performance monitoring and fault locating capabilities in cluster scenarios.
 * [**msServiceProfiler (MindStudio Service Profiler)**](https://gitcode.com/Ascend/msserviceprofiler/blob/26.0.0/docs/zh/quick_start.md)    
    Service-oriented performance optimization tool: It is a service-oriented performance optimization tool with Ascend affinity. It supports request scheduling and model execution visualization, improving service-oriented performance analysis efficiency.
 * [**msprechecker (MindStudio Prechecker Tool)**](https://gitcode.com/Ascend/msit/blob/26.0.0/msprechecker/README.md)    
    Precheck tool: The msprechecker provides the precheck capability for inference scenarios, including environment precheck, connectivity precheck, flushing, and comparison during inference. This feature helps users detect exceptions before inference service deployment. Improves the inference performance and quickly reproduces the baseline.
 * [**msprof-analyze (MindStudio Profiler Analyze)**](https://gitcode.com/Ascend/msprof-analyze/blob/26.0.0/docs/zh/README.md)    
    Ascend performance analysis tool: Analyzes collected performance data and quickly identifies performance bottlenecks of Ascend devices.
 * [**msInsight (MindStudio Insight)**](https://gitcode.com/Ascend/msinsight/blob/26.0.0/docs/zh/user_guide/overview.md)    
    MindStudio Insight: supports multi-dimensional performance analysis in multiple scenarios, such as system-level, operator-level, and servitization, and in-depth performance data analysis, helping developers complete performance diagnosis.

### Precision tool

 * [**msProbe (MindStudio Probe)**](https://gitcode.com/Ascend/msprobe/blob/26.0.0/docs/zh/dump/mindspore_dump_quick_start.md)    
    Precision debugging tool: It is a tool package used during precision debugging during model development. It is a precision tool chain for all scenarios provided by the Ascend. It helps users improve the efficiency of model precision locating.
 * [**msMemScope (MindStudio MemScope)**](https://gitcode.com/Ascend/msmemscope/blob/26.0.0/docs/zh/quick_start.md)    
    Memory tool: It is a dedicated tool for Ascend memory debugging and optimization. It provides network-wide multi-dimensional video memory data collection, automatic diagnosis, optimization, and analysis capabilities.

### Quantification tool

 * [**msModelSlim (MindStudio ModelSlim)**](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/getting_started/quantization_quick_start.md)    
    Model compression tool: Ascend model compression tool, an affinity compression tool that aims at acceleration, compression, and Ascend. It includes a series of reasoning optimization technologies such as quantization and compression, and supports large language dense models, MoE models, multi-modal understanding models, and multi-modal generation models.
