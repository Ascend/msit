# msIT Tool Selection Guide

<br>

The msIT toolchain includes multiple specialized tools that cover all stages of inference development. When facing a specific task, precise tool selection is often more efficient than trying tools at random.

This document is organized around **what you want to do** and helps you quickly pinpoint the best-matching tool and its direct entry point.

<br>

## Scenario-Based Tool Recommendations

| Scenario | Recommended Tool | Function |
|-----------|----------|------------|
| Evaluating model performance, service throughput, and parameter configuration effects before deployment |[**msModeling**](https://gitcode.com/Ascend/msmodeling/blob/26.2.0/docs/en/quick_start/tensorcast_throughput_optimizer_quick_start.md)| Covers model performance simulation, throughput optimization, service-level simulation, and parameter tuning, helping identify performance bottlenecks early and optimize deployment configurations. |
| Quickly evaluating whether the current environment is deployment-ready and reproduce environment configuration differences between two devices |[**msprechecker**](https://gitcode.com/Ascend/msit/blob/26.2.0/msprechecker/README_EN.md)| Provides three core functions: precheck, environment information dump, and comparison. |
| Generating quantized weights for an open-source model | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/en/quick_start/quantization_quick_start.md) | Supports one-click quantization, provides a best-practice library for quantized models, enabling fast weight generation. |
| Recovering from post-quantization accuracy loss with an automated solution-finding process | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/en/quick_start/quantization_quick_start.md) | Automatically iterates after you set the accuracy target: generate yaml → quantize → launch service via vLLM-Ascend → evaluate with AISBench → try another configuration if the target is not met, until it hits the target. |
| Determining which layers to roll back during quantization accuracy tuning | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/en/quick_start/quantization_quick_start.md) | Provides multiple granularities and gives suggestions on quantization-sensitive layers. |
| Performing weight format or precision conversion only | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/en/quick_start/quantization_quick_start.md) | Runs on CPU for fast format conversion (e.g., BF16 to MXFP8). No model code loading, no calibration set required. |
| Debugging training or inference model accuracy issues |[**msProbe**](https://gitcode.com/Ascend/msprobe/blob/26.2.0/docs/en/quick_start/pytorch_quick_start.md)| Provides Ascend full-scenario training accuracy debugging and issue locating. |
| Collecting profile data of Ascend AI tasks via CLI |[**msProf**](https://gitcode.com/Ascend/msprof/blob/26.2.0/docs/en/quick_start/msprof_quick_start.md)| Captures CANN and NPU profile data from the CLI. The tool is built in the CANN package—no installation needed. |
| Quickly analyzing profile data and getting tuning recommendations for performance bottlenecks | [**msprof-analyze**](https://gitcode.com/Ascend/msprof-analyze/blob/26.2.0/docs/en/quick_start/msprof-analyze_quick_start.md) | Performs statistics, comparison, and diagnosis based on collected profile data, helping locate performance bottlenecks in computation, communication, scheduling, and cluster scenarios. |
| Collecting and parsing profile data, then analyzing bottlenecks with tuning advice |[**msServiceProfiler**](https://gitcode.com/Ascend/msserviceprofiler/blob/26.2.0/docs/en/quick_start/quick_start.md)| Analyzes collected profile data to identify bottlenecks in service framework, computation, communication, and scheduling. |
| Analyzing memory usage of training or inference tasks for debugging and tuning | [**msMemScope**](https://gitcode.com/Ascend/msmemscope/blob/26.2.0/docs/en/quick_start/quick_start.md) | Collects Ascend full-scenario memory data, with supporting analysis capabilities such as visualization, comparison, and breakdown. |
|           Obtaining NPU profile data online via Profiling APIs           | [**msPTI**](https://gitcode.com/Ascend/mspti/blob/26.2.0/docs/en/quick_start/mspti_quick_start.md) | Provides basic capabilities for obtaining NPU profile data online.                           |
| Visually highlighting performance issues during troubleshooting | [**msInsight**](https://gitcode.com/Ascend/msinsight/blob/26.2.0/docs/en/quick_start/system_tuning_quick_start.md) | Provides visual performance analysis covering system, operator, and service scenarios, assisting in performance diagnosis. |
| Obtaining cluster profile data through lightweight online monitoring | [**msMonitor**](https://gitcode.com/Ascend/msmonitor/blob/26.2.0/docs/en/quick_start/msmonitor_quick_start.md) | Provides one-stop monitoring, supporting dump and online collection, for cluster monitoring and issue diagnosis. |
| Instrumenting critical functions, iterations, or custom phases for analysis | [**msTX**](https://gitcode.com/Ascend/mstx/blob/26.2.0/docs/en/install_guide/mstx_install_guide.md) | Provides an extended SDK for marking custom collection intervals, assisting in performance and issue diagnosis. |
| Enabling AI-driven operator tuning via natural language instructions | [**msAgent**](https://gitcode.com/Ascend/msagent/blob/26.2.0/docs/en/quick_start/msagent_quick_start.md) | Performs performance tuning, accuracy analysis, and model quantization based on prompt understanding. |
