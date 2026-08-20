# **Inference Development Toolchain Quick Start**

<br>

## 1. Overview

### 1.1 Features

MindStudio Inference Tools (msIT) is an open-source inference development toolchain for the Ascend AI platform. It provides developers with full-process tool support from model deployment to online operation and maintenance (O&M). The toolchain covers five phases: pre-check, quantization, accuracy debugging, performance tuning, and online monitoring. It includes multiple specialized tools to help developers efficiently complete inference development and optimization for LLMs (including dense models and MoE architectures), multimodal models, and traditional models on Ascend hardware.

### 1.2 Core Capabilities

msIT provides the following core capabilities for the key phases of inference development:

- **Deployment pre-check**: Performs environment verification, connectivity detection, and inference result comparison before formal deployment, identifying configuration anomalies in advance and reducing deployment risks.
- **Model compression**: Provides quantization and compression capabilities, supports one-click quantization and automated accuracy iteration, and reduces inference resource overhead while ensuring model accuracy.
- **Accuracy debugging**: Supports all-scenario accuracy collection and comparison analysis, helping developers quickly identify the root cause of accuracy deviations.
- **Performance tuning**: Covers data collection, bottleneck analysis, visual diagnosis, serving tuning, and modeling and simulation, providing multi-level performance optimization methods from a single operator to service clusters.
- **Online monitoring**: Provides cluster-oriented real-time performance monitoring and fault localization to ensure the stable operation of inference services.

### 1.3 Purpose of This Document

This document summarizes the functional positioning and quick start entry of each tool in the toolchain, helping you quickly select the appropriate tool and get started based on your current development phase and business requirements.

## 2. Quick Tool Navigation

| Category | Tool | Description | Quick Start Link |
|:--:|:--|:--|:--:|
| Pre-check | **msPrechecker** | **Pre-check tool**: Supports environment pre-check, connectivity pre-check, and dump and comparison of inference data, helping users identify anomalies before deployment. | [Click to view](../../../msprechecker/README.md#快速入门) |
| Quantization | **msModelSlim** | **Model compression**: Provides inference optimization techniques such as quantization and compression, supporting dense LLMs, MoE models, multimodal models, and so on. | [Click to view](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/docs/en/quick_start/quantization_quick_start.md) |
| Accuracy | **msProbe** | **Accuracy debugging**: an all-scenario accuracy tool for Ascend, used for accuracy debugging and issue localization | [Click to view](https://gitcode.com/Ascend/msprobe/blob/26.1.0/docs/en/user_guide/dump/atb_data_dump_instruct.md#%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8) |
| Performance | **msProf** | **Model tuning**: the all-scenario performance tuning foundation that collects full-stack hardware and software performance data to improve device tuning efficiency | [Click to view](https://gitcode.com/Ascend/msprof/blob/26.1.0/docs/en/quick_start/msprof_quick_start.md) |
| Performance | **msprof-analyze** | **Performance analysis**: Performs performance analysis based on collected data to quickly identify performance bottlenecks. | [Click to view](https://gitcode.com/Ascend/msprof-analyze/blob/26.1.0/docs/en/quick_start/msprof-analyze_quick_start.md) |
| Performance | **msServiceProfiler** | **Serving tuning**: Supports visualization of request scheduling and model execution processes, improving the efficiency of serving performance analysis. | [Click to view](https://gitcode.com/Ascend/msserviceprofiler/blob/26.1.0/docs/en/quick_start.md) |
| Performance | **msMemScope** | **Memory tuning**: a dedicated memory tuning tool that provides network-wide multi-dimensional accelerator memory collection, automatic diagnosis, and optimization analysis | [Click to view](https://gitcode.com/Ascend/msmemscope/blob/26.1.0/docs/en/quick_start/quick_start.md) |
| Performance | **msInsight** | **Visualized tuning**: Provides visualized performance analysis covering system, operator, serving, and other scenarios, assisting in performance diagnosis. | [Click to view](https://gitcode.com/Ascend/msinsight/blob/26.1.0/docs/en/quick_start/system_tuning_quick_start.md) |
| Performance | **msModeling** | **Modeling and simulation**: a neural network inference performance simulation framework that helps developers predict performance, identify bottlenecks, and optimize configurations without hardware or before deployment | [Click to view](https://gitcode.com/Ascend/msmodeling/blob/26.1.0/docs/en/quick_start/tensorcast_throughput_optimizer_quick_start.md) |
| Monitoring | **msMonitor** | **Online monitoring**: one-stop monitoring that supports trace data dumping to the drive and online collection, providing cluster-oriented monitoring and issue localization | [Click to view](https://gitcode.com/Ascend/msmonitor/blob/26.1.0/docs/en/quick_start/msmonitor_quick_start.md) |
