# 版本说明

## 版本配套说明

### 产品版本信息

| 产品名称 | 产品版本 | 版本类型 |
|---------|---------|---------|
| msIT    | 8.3.0   | 正式版本 |

### 相关产品版本配套说明

| msIT版本 | CANN版本        | PyTorch版本 | torch_npu版本 | Python版本     |
|----------|----------------|-------------|---------------|----------------|
| 8.3.0    | 8.2.RC1及以上版本 | 2.1以上      | 2.1以上        | Python 3.8以上  |

## 版本兼容性说明

无

## 特性变更说明

### 8.3.0

#### 一、新增说明

##### 性能工具

###### msMonitor（MindStudio Monitor）

1. msMonitor NPU Monitor组件新增轻量化数据落盘能力：支持以 Jsonl 格式进行数据持久化，为后续数据分析与系统集成提供更高效、易处理的数据格式支持。
2. msMonitor 命令行交互体验优化：status 子命令支持实时查询当前执行的 step 状态，提升工具的使用便捷性。

###### msServiceProfiler（MindStudio Service Profiler）

1. 支持Torch Profiler数据采集，解析。
2. 支持与OpenTelemetry开源生态对接，进行Trace数据追踪。
3. 支持无侵入式自动插桩采集vLLM框架服务化性能数据。
4. 支持自动寻优插件化模式。

###### msprechecker（MindStudio Prechecker Tool）

1. msprechecker新增支持检查结果落盘功能。 

###### msprof-analyze（MindStudio Profiler Analyze）

1. 新增module_statistic分析能力：提供针对PyTorch模型自动解析模型层级结构的分析能力，帮助精准定位性能瓶颈。

###### msInsight（MindStudio Insight）

1. tb_graph_ascend 组件 UI 优化：调整了部分排版与选项样式，提升界面整洁度与操作体验。
2. 新增快捷键说明：便于用户快速掌握常用操作，提高使用效率。

##### 精度工具

###### msProbe（MindStudio Probe）

1. msProbe新增支持MindSpeed、Mindformers跨框架自动化比对能力。

###### msMemScope（MindStudio MemScope）

1. msMemScope支持Python API采集方式使用。
2. msMemScope支持PyTorch框架下采集内存快照。
3. msMemScope支持识别显存页表属性并进行落盘。
4. msMemScope支持获取Driver新增的显存分配接口。

##### 量化工具

###### msModelSlim（MindStudio ModelSlim）

1. msModelSlim 支持量化精度反馈自动调优，可根据精度需求自动搜索最优量化配置。
2. msModelSlim 支持自主量化多模态理解模型，支持多模态理解模型的量化接入。
3. msModelSlim 一键量化支持多卡量化，支持分布式逐层量化，提升大模型量化效率。
4. msModelSlim 支持 DeepSeek-V3.2 W8A8 量化，单卡64G显存、100G内存即可执行。
5. msModelSlim 支持 DeepSeek-V3.2-Exp W4A8 量化，单卡64G显存、100G内存即可执行。
6. msModelSlim 支持 Qwen3-VL-235B-A22B W8A8 量化。
7. msModelSlim 模型适配支持插件化和配置注册，支持依赖预检。
8. msModelSlim 支持 Qwen3-235B-A22B W4A8、Qwen3-30B-A3B W4A8 量化。vLLM Ascend已支持量化模型推理部署。
9. msModelSlim 支持 DeepSeek-V3.2-Exp W8A8 量化，单卡64G显存，100G内存即可执行。
10. msModelSlim 现已解决Qwen3-235B-A22B在W8A8量化下频繁出现“游戏副本”等异常token的问题。
11. msModelSlim 支持DeepSeek R1 W4A8 per-channel 量化【Prototype】。
12. msModelSlim 支持大模型量化敏感层分析。

#### 二、删除说明

##### 性能工具

###### msMonitor（MindStudio Monitor）

1. 移除 msMonitor NPU Trace 组件中冗余的 GPU 相关指令。

###### msInsight（MindStudio Insight）

1. 移除“精度颜色自定义配置”选项：原以数字表示的精度颜色，现统一调整为使用"pass""warning""error"三类状态标识，更加直观清晰。

#### 三、Bugfix

##### 性能工具

###### msInsight（MindStudio Insight）

1. 修复旧版本中部分精度溢出数据的适配问题，避免系统误判，确保判断准确性。

##### 精度工具

###### msMemScope（MindStudio MemScope）

1. 修复源码编译时，wget获取sqlite压缩包，证书报错问题。
2. 修复解压时不存在unzip命令的报错。
3. 修复可见卡场景下，数据落盘错卡问题。
4. 修复同时开启kernel和trace采集db文件，出现trace和dump文件落盘错乱的问题。
