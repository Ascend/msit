# msIT Tool Selection Guide

<br>

The msIT toolchain includes multiple specialized tools that cover all stages of inference development. When facing a specific task, precise tool selection is often more efficient than trying tools at random.

This document is organized around **what you want to do** and helps you quickly pinpoint the best-matching tool and its direct entry point.

<br>

## Scenario-Based Tool Recommendations

| What You Want to Do | Recommended Tool | Why It Is Recommended |
|:------------------------------------------------------------| :----------------------------------------------------------- | :----------------------------------------------------------- |
| I want to evaluate model performance, service throughput, and parameter configuration effects before deployment |[**msModeling**](https://gitcode.com/Ascend/msmodeling/blob/master/README.md)| Covers model performance simulation, throughput optimization, service-level simulation, and parameter optimization, helping you evaluate performance bottlenecks in advance and optimize deployment configurations. |
| I want to quickly check whether the current environment meets deployment conditions and reproduce and compare environment configuration differences between two devices |[**msprechecker**](https://gitcode.com/Ascend/msit/blob/master/msprechecker/README.md)|The tool provides three core functions: precheck, environment information dump, and difference comparison. |
| I have an open-source model and want to generate quantized weights | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) | Supports one-click quantization and provides a best-practice library for quantized models, enabling you to quickly generate quantized weights. |
| My model loses too much accuracy after quantization, and I want to automatically find a solution that meets the accuracy target | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) | Simply specify the accuracy target, and the tool iterates automatically: generate `yaml` → quantize → launch a service with vLLM-Ascend → evaluate with AISBench → if the target is not met, try another configuration until the target is achieved. |
| During quantization accuracy tuning, I don't know which layers to roll back | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) | Provides multiple granularity options and gives recommendations on layers sensitive to quantization. |
| I only want to convert weight format or precision | [**msModelSlim**](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) | No need to load model code or a calibration set, and it can run on the CPU to quickly complete weight format conversion (for example, `BF16` to `MXFP8`). |
| I want to debug training or inference model accuracy and locate accuracy issues |[**msProbe**](https://gitcode.com/Ascend/msprobe/blob/master/README.md)| Ascend accuracy tool for all scenarios, used for training accuracy debugging and issue location|
| I want to collect performance data for Ascend AI tasks from the CLI |[**msProf**](https://gitcode.com/Ascend/msprof/blob/master/README.md)| Bundled by default in the CANN package with no installation required, collecting CANN and NPU performance data from the CLI.      |
| I want to quickly analyze profiling data, identify performance bottlenecks, and receive tuning recommendations| [**msprof-analyze**](https://gitcode.com/Ascend/msprof-analyze/blob/master/README.md) | Performs statistics, comparison, and diagnosis on collected profiling data to help locate performance bottlenecks in compute, communication, scheduling, and cluster scenarios. |
| I want to collect and analyze performance data of MindIE/vLLM/SGLang frameworks, identify performance bottlenecks, and receive tuning recommendations|[**msServiceProfiler**](https://gitcode.com/Ascend/msserviceprofiler/blob/master/README.md)|Analyzes collected performance data to help locate performance bottlenecks in service-oriented frameworks, compute, communication, and scheduling. |
| I want to analyze memory usage in training or inference tasks and debug or tune memory | [**msMemScope**](https://gitcode.com/Ascend/msmemscope/blob/master/README.md) | Collects memory data across all Ascend scenarios, with analysis capabilities such as visualization, comparison, and breakdown. |
| I want to present performance issues graphically when locating them | [**msInsight**](https://gitcode.com/Ascend/msinsight/blob/master/README.md) | Visual performance analysis covering system, operator, service-oriented, and other scenarios, assisting with performance diagnosis |
| I want to obtain cluster performance data in a lightweight manner through online monitoring | [**msMonitor**](https://gitcode.com/Ascend/msmonitor/blob/master/README.md) | One-stop monitoring that supports data dumping to the drive and online collection, providing cluster-oriented monitoring and issue location |
