# Version Description

## Version Matching Description

### Product version information

| Product Name | Product Version | Version Type     |
| ------------ | --------------- | ---------------- |
| msIT         | 8.3.0           | Official version |

### Related Product Versions

| msIT version | CANN version     | PyTorch Version | torch_npu version | Python version      |
| ------------ | ---------------- | --------------- | ----------------- | ------------------- |
| 8.3.0        | 8.2.RC1 or later | 2.1 or later    | 2.1 or later      | Python 3.8 or later |

## Version Compatibility

None

## Feature Updates

### 8.3.0

#### 1. New description

##### Performance tool

###### msMonitor (MindStudio Monitor)

1. Lightweight data flushing capability is added to the msMonitor NPU Monitor component. Data persistence can be performed in JSONL format, providing efficient and easy-to-process data format support for subsequent data analysis and system integration.
2. The msMonitor command line interaction is optimized. The status subcommand can query the status of the current step in real time, improving the tool convenience.

###### msServiceProfiler (MindStudio Service Profiler)

1. Supports data collection and parsing by Torch Profiler.
2. Supports interconnection with the OpenTelemetry open-source ecosystem for trace data tracing.
3. Supports non-intrusive automatic instrumentation to collect service-oriented performance data of the vLLM framework.
4. Supports the automatic optimization plug-in mode.

###### msprechecker (MindStudio Prechecker Tool)

1. The msprechecker supports the function of flushing check results to disks.

###### msprof-analyze (MindStudio Profiler Analyze)

1. The module_statistic analysis capability is added. The module_statistic analysis capability can automatically parse the model structure of the PyTorch model, helping accurately locate performance bottlenecks.

###### msInsight (MindStudio Insight)

1. Optimized the UI of the tb_graph_ascend component: Some typesetting and option styles are adjusted to improve the GUI cleanliness and operation experience.
2. Shortcut key description is added to help users quickly master common operations and improve efficiency.

##### Precision tool

###### msProbe (MindStudio Probe)

1. The msProbe supports automatic cross-framework comparison between MindSpeed and Mindformers.

###### msMemScope (MindStudio MemScope)

1. The msMemScope supports Python API collection.
2. The msMemScope supports memory snapshot collection in the PyTorch framework.
3. The msMemScope supports the identification of the memory page table attributes and flushing to disk.
4. The msMemScope supports the interface for obtaining the new video memory allocation of the driver.

##### Quantification tool

###### msModelSlim (MindStudio ModelSlim)

1. The msModelSlim supports automatic optimization of the quantization precision feedback and can automatically search for the optimal quantization configuration based on precision requirements.
2. The msModelSlim supports self-quantized multi-modal understanding models and quantitative access to multi-modal understanding models.
3. The msModelSlim supports multi-card quantization and distributed layer-by-layer quantization, improving the quantization efficiency of large models.
4. msModelSlim supports DeepSeek-V3.2 W8A8 quantization. A single card can be executed with 64 GB video memory and 100 GB memory.
5. msModelSlim supports DeepSeek-V3.2-Exp W4A8 quantization. A single card can be executed with 64 GB video memory and 100 GB memory.
6. msModelSlim supports Qwen3-VL-235B-A22B W8A8 quantization.
7. The msModelSlim model adaptation supports plug-in, configuration registration, and dependency pre-check.
8. The msModelSlim supports Qwen3-235B-A22B W4A8, Qwen3-30B-A3B W4A8 quantization. The vLLM Ascend supports quantitative model inference deployment.
9. The msModelSlim supports DeepSeek-V3.2-Exp W8A8 quantization. A single card has 64 GB video memory and 100 GB memory can be executed.
10. MsModelSlim has fixed the problem that Qwen3-235B-A22B frequently appears abnormal tokens such as "game copy" under W8A8 quantization.
11. msModelSlim supports DeepSeek R1 W4A8 per-channel quantization \[Prototype\].
12. The msModelSlim supports large model quantitative sensitive layer analysis.

#### 2. Deleted Description

##### Performance tool

###### msMonitor (MindStudio Monitor)

1. Remove redundant GPU instructions from the msMonitor NPU Trace component.

###### msInsight (MindStudio Insight)

1. Removed the Customization of Precision Color option: The original precision color represented by digits is now changed to "pass", "warning", and "error" status identifiers, which are more intuitive and clear.

#### 3. Bugfix

##### Performance tool

###### msInsight (MindStudio Insight)

1. Some precision overflow data in the earlier version is adapted to prevent misjudgment and ensure accurate judgment.

##### Precision tool

###### msMemScope (MindStudio MemScope)

1. During source code compilation, a certificate error is reported when the wget obtains the sqlite package.
2. Rectify the error that the unzip command does not exist during decompression.
3. Fixed an issue where data is flushed to the wrong card in the visible card scenario.
4. When both the kernel and trace files are enabled to collect db files, the trace and dump files are flushed to disks in disorder.
