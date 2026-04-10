# **msIT**

## 最新消息

- [2026.1.12]：[msit仓库License变更通知](https://gitcode.com/Ascend/msit/discussions/1)。

- [2025.12.31]：昇腾平台MindStudio推理工具链全面开源，涉及如下代码仓。

    - [MindStudio-Profiler](https://gitcode.com/Ascend/msprof)  
    构建昇腾全场景性能调优基础能力，支持采集CANN和NPU性能数据，提升昇腾设备性能调优效率。

    - [MindStudio-Profiler-Analyze](https://gitcode.com/Ascend/msprof-analyze)  
    昇腾性能分析工具，基于采集的性能数据进行分析，提供昇腾设备性能瓶颈快速识别能力。

    - [MindStudio-MemScope](https://gitcode.com/Ascend/msmemscope)  
    针对昇腾显存调试调优场景的专用工具，提供整网级多维度显存数据采集、自动诊断、优化分析能力。

    - [MindStudio-Service-Profiler](https://gitcode.com/Ascend/msserviceprofiler)  
    昇腾亲和的服务化性能调优工具，支持请求调度、模型执行过程可视化，提升服务化性能分析效率。

    - [MindStudio-Monitor](https://gitcode.com/Ascend/msmonitor)  
    一站式在线监控工具，支持落盘和在线性能数据采集，提供集群场景性能监测及定位能力。

    - [MindStudio-ModelSlim](https://gitcode.com/Ascend/msmodelslim)  
    昇腾模型压缩工具，一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。包含量化和压缩等一系列推理优化技术，支持大语言稠密模型、MoE模型、多模态理解模型、多模态生成模型等。

    - [MindStudio-Insight](https://gitcode.com/Ascend/msinsight)  
    MindStudio Insight可视化工具，支持系统级、算子级、服务化等多场景多维度性能分析，深度剖析性能数据，帮助开发者完成性能诊断。
    
    - [MindStudio-Probe](https://gitcode.com/Ascend/msprobe)  
    模型开发精度调试环节使用的工具包，是针对昇腾提供的全场景精度工具链，帮助用户提高模型精度定位效率。

## 简介

MindStudio Inference Tools（MindStudio昇腾推理工具链，msIT），为用户提供大模型与传统模型推理开发中常用的模型压缩、模型调试调优等功能，支持推理服务化场景下的性能调优能力，帮助用户达到最优的推理性能。

## 目录结构  

关键目录如下。

```tex
|—————— msit                     # msit推理工具
|—————— msmodelslim              # msmodelslim量化工具
|—————— msprechecker             # 预检工具
|—————— msserviceprofiler        # 服务化调优工具
|—————— test                     # UT测试
|—————— README.md                # 总仓介绍
```

## 快速入门

快速入门是以一个简单模型为例，介绍大模型推理工具链中的模型量化、数据dump、精度比对、性能调优等工具的使用，具体内容请参见《[快速入门](./docs/zh/msit_quick_start.md)》。

## 功能介绍

作为昇腾平台的统一推理开发工具链，包含模型量化、精度调试和性能调优等工具，可根据下方的工具介绍，选择相应工具查看具体信息，进行模型推理。

### 性能工具

- [**msProf（MindStudio Profiler）**](https://gitcode.com/Ascend/msprof)<br>
    **数据采集工具**：构建昇腾全场景性能调优基础能力，支持采集CANN和NPU性能数据，提升昇腾设备性能调优效率。

- [**msMonitor（MindStudio Monitor）**](https://gitcode.com/Ascend/msmonitor)<br>
    **在线监控工具**一站式在线监控工具，支持落盘和在线性能数据采集，提供集群场景性能监测及定位能力。

- [**msServiceProfiler（MindStudio Service Profiler）**](https://gitcode.com/Ascend/msserviceprofiler)<br>
    **服务化性能调优工具**：昇腾亲和的服务化性能调优工具，支持请求调度、模型执行过程可视化，提升服务化性能分析效率。

- [**msprechecker（MindStudio Prechecker Tool）**](https://gitcode.com/Ascend/msit/tree/master/msprechecker)<br>
    **预检工具**：msprechecker提供推理场景的预检能力，支持环境预检，连通性预检，推理过程中的落盘和比对功能。帮助用户在推理业务部署前，提前发现异常问题。推理时，提高推理性能，快速复现基线。

- [**msprof-analyze（MindStudio Profiler Analyze）**](https://gitcode.com/Ascend/msprof-analyze)<br>
    **昇腾性能分析工具**：基于采集的性能数据进行分析，提供昇腾设备性能瓶颈快速识别能力。

- [**msInsight（MindStudio Insight）**](https://gitcode.com/Ascend/msinsight)<br>
    **MindStudio Insight可视化工具**：支持系统级、算子级、服务化等多场景多维度性能分析，深度剖析性能数据，帮助开发者完成性能诊断。

### 精度工具
    
- [**msProbe（MindStudio Probe）**](https://gitcode.com/Ascend/msprobe)<br>
    **精度调试工具**：模型开发精度调试环节使用的工具包，是针对昇腾提供的全场景精度工具链，帮助用户提高模型精度定位效率。

- [**msMemScope（MindStudio MemScope）**](https://gitcode.com/Ascend/msmemscope)<br>
    **内存工具**：针对昇腾显存调试调优场景的专用工具，提供整网级多维度显存数据采集、自动诊断、优化分析能力。

### 量化工具

- [**msModelSlim（MindStudio ModelSlim）**](https://gitcode.com/Ascend/msmodelslim)<br>
    **模型压缩工具**：昇腾模型压缩工具，一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。包含量化和压缩等一系列推理优化技术，支持大语言稠密模型、MoE模型、多模态理解模型、多模态生成模型等。

## 安全声明  

描述msIT相关的安全信息，公网地址以及通信矩阵等信息，具体内容请参见《[安全声明](./docs/zh/security_statement.md)》。

## 免责声明

- 本工具仅供调试和开发之用，不适用于生产环境。使用者需自行承担使用风险，并理解以下内容：

  - [x] 仅限调试开发使用：此工具设计用于辅助开发人员进行调试，不适用于生产环境或其他商业用途。对于因误用本工具而导致的数据丢失、损坏，本工具及其开发者不承担责任。

  - [x] 数据处理及删除：用户在使用本工具过程中产生的数据（包括但不限于dump的数据）属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防泄露或不必要的信息泄露。

  - [x] 数据保密与传播：使用者了解并同意不得将通过本工具产生的数据随意外泄或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本工具及其开发者概不负责。

  - [x] 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入命令行不当所导致的问题，本工具及其开发者概不负责。

- 免责声明范围：本免责声明适用于所有使用本工具的个人或实体。使用本工具即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本工具。

- 在使用本工具之前，请**谨慎阅读并理解以上免责声明的内容**。对于使用本工具所产生的任何问题或疑问，请及时联系开发者。

## License

msIT工具的使用许可证，具体请参见《[LICENSE](./LICENSE)》。

msIT工具docs目录下的文档适用CC-BY 4.0许可证，具体请参见《[LICENSE](./docs/LICENSE)》。

## 贡献声明

1. **提交错误报告**：如果您在msIT中发现了一个不存在安全问题的漏洞，请在msIT仓库中的Issues中搜索，以防该漏洞已被提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应该包含完整信息。
2. **安全问题处理**：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。
3. **解决现有问题**：通过查看仓库的Issues列表可以发现需要处理的问题信息, 可以尝试解决其中的某个问题。
4. **如何提出新功能**：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. **开始贡献**：
    1. Fork本项目的仓库。
    2. Clone到本地。
    3. 创建开发分支。
    4. 本地测试：提交前请通过所有单元测试，包括新增的测试用例。
    5. 提交代码。
    6. 新建Pull Request。
    7. 代码检视：您需要根据评审意见修改代码，并重新提交更新。此流程可能涉及多轮迭代。
    8. 当您的PR获得足够数量的检视者批准后，Committer会进行最终审核。
    9. 审核和测试通过后，CI会将您的PR合并到项目的主干分支。

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[Issues](https://gitcode.com/Ascend/msit/issues)，我们会尽快回复。感谢您的支持。

## 致谢

msIT由华为公司的下列部门联合贡献：

- 昇腾计算MindStudio开发部

感谢来自社区的每一个PR，欢迎贡献msIT！
