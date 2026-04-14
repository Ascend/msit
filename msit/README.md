# **msit**

## 简介

MindStudio Inference Tools(msit)作为昇腾统一推理工具，提供一体化开发功能，帮助用户进行模型迁移以及性能与精度的调试调优。目前，该工具包括benchmark、debug、analyze、convert、profile、llm、tensor-view等组件。

**注意**

msit中的llm组件，debug组件功能在[msprobe](https://gitcode.com/Ascend/msprobe)中已有对应的优化实现，msit中相关功能已停止演进，建议前往使用msprobe工具。

## 注意事项

- 工具读写的所有路径，只允许包含大小写字母、数字、下划线、斜杠、点和短横线。

- 出于安全性及权限最小化角度考虑，msit工具不应使用root等高权限账户使用，建议使用普通用户权限安装执行。

- 使用msit工具前请确保执行用户的umask值大于等于0027，否则可能会导致工具生成的精度数据文件和目录权限过大。

- 用户须自行保证使用最小权限原则，如给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。

- 使用msit命令行时，请检查当前环境是否有可用且唯一的Python环境。

## 命令格式

msit工具通过命令行方式启动。

```bash
msit <TASK> <SUB_TASK> [OPT] [ARGS]
```

|参数|说明|
|-----|-----|
|TASK|任务类型，当前支持 debug、benchmark、analyze、convert、profile、llm、graph、tensor-view，具体任务介绍可参见 [各组件功能介绍章节](#各组件功能介绍)。<br> 也可通过执行```msit -h```命令查看当前支持的任务列表。|
|SUB_TASK| `<TASK>` 下包含的子任务类型，以 `debug` 任务为例，可以通过执行```msit debug -h```m命令，查看每个任务支持的子功能列表。|
|OPT和ARGS|可选项及参数，每个任务下的可选项和参数可能不同，以 `debug` 任务下的 `compare` 子任务为例，可以通过执行```msit debug compare -h```命令获取可选项和参数。|

例如使用llm的dump功能，启动命令为：

```bash
msit llm dump <options>
```

## 各组件功能介绍

### llm

提供[**MindIE**](https://www.hiascend.com/software/mindie)和[**torchair**](./docs/glossary/README.md)框架下的[大模型推理调试工具](./docs/llm/README.md)，包含模块参见下表。

|模块|说明|
|-----|-----|
|dump|提供了大模型推理过程的数据dump功能，包含ATB和PyTorch场景。使用指南请分别参见[atb dump用户指南](./docs/llm/工具-DUMP加速库数据使用说明.md)和[Pytorch dump用户指南](./docs/llm/工具-Pytorch场景数据dump.md)。|
|compare|提供大模型推理的自动比对功能，快速定位算子精度问题。使用指南请参见[compare用户指南](./docs/llm/工具-大模型精度比对.md)。|
|opcheck|提供加速库（ATB）的单算子精度预检功能，检测加速库算子精度是否达标。使用指南请参见[opcheck用户指南](./docs/llm/工具-精度预检使用说明.md)。|
|errcheck|提供异常检测能力，目前仅支持算子溢出检测。使用指南请参见[异常检测使用说明](./docs/llm/工具-异常检测使用说明.md)。|

### debug

提供一站式调试功能，用于传统小模型下定位用户推理过程中的问题，确保模型的正确性。包含模块参见下表。

|模块|说明|
|-----|-----|
|dump|提供了传统小模型场景下的数据dump功能，适用于TensorFlow、TensorFlow 2.0、ONNX、Caffe、MindIE-Torch框架。使用指南请参见[dump用户指南](./docs/debug/dump/README.md)。|
|compare|提供了传统小模型推理场景下的自动化比对功能，用于定位问题算子，适用于TensorFlow、TensorFlow 2.0、ONNX、Caffe、MindIE-Torch框架。使用指南请参见[compare用户指南](./docs/debug/compare/README.md)和[融合pass比对](./docs/debug/compare/融合算子匹配对应pass使用说明.md)。|
|opcheck|提供了传统小模型场景下精度预检功能，支持对经过GE推理后dump落盘数据进行算子精度预检，检测kernel级别的算子精度是否达标。使用指南请参见[opcheck用户指南](./docs/debug/opcheck/README.md)。<br> 目前只支持TensorFlow 2.6.5。|
|surgeon|使能ONNX模型在昇腾芯片的优化，并提供基于ONNX的改图功能。使用指南请参见[surgeon用户指南](./docs/debug/surgeon/README.md)。|

### analyze

提供模型从其他平台迁移至昇腾平台的支持度分析功能，分析算子支持情况、算子定义是否符合约束条件和算子输入是否为空。使用指南请参见[analyze用户指南](./components/analyze/README.md)。

### convert

提供将ONNX、TensorFlow、Caffe、MindSpore等框架的模型文件转化为OM类型文件的功能，并支持调优。使用指南请参见[convert用户指南](./docs/convert/README.md)。

### profile

提供性能分析功能，面向OM类型文件（由onnx等文件转换为的离线模型）在昇腾设备上进行模型推理性能分析，提供整网详细的性能数据及相关信息。使用指南请参见[profile用户指南](./docs/profile/README.md)。

### benchmark

针对指定的推理模型运行推理程序，并能够测试推理模型的性能（包括吞吐率、时延），帮助用户评估推理模型的表现。使用指南请参见[benchmark用户指南](./docs/benchmark/README.md)。

### graph

提供基于GE（Graph Engine，图引擎）的图统计、压缩、截取、性能分析等功能。使用指南请参见[graph用户指南](./docs/graph/README.md)。

### tensorview

提供了查看tensor的接口，并能够对tensor进行链式切片、转置操作。使用指南请参见[tensor-view用户指南](./docs/tensor_view/README.md)。

### elb

提供Deepseek模型在静态/动态场景下负载均衡亲和专家寻优策略。使用指南请参见[负载均衡算法用户指南](./docs/expert_load_balancing/工具-负载均衡亲和专家寻优.md)。

## FAQ

msit工具收集了msit安装常见问题和其它问题，可分别参见对应的FAQ查看。

- [msit使用以及安装常见问题](https://gitcode.com/Ascend/msit/wiki/msit%E7%9A%84%E5%AE%89%E8%A3%85%E4%B8%8E%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%2Fmsit%E5%AE%89%E8%A3%85.md)

- [更多FAQ请点击](./docs/FAQ.md)

## 免责声明

- msit仅提供在昇腾设备上的一体化开发工具，支持一站式调优，不对其质量或维护负责。如果您遇到了问题，请在GitCode/Ascend/msit提交Issues，我们将根据您的issues跟踪解决。衷心感谢您对我们社区的理解和贡献。

- 部分msit依赖包的某些版本存在已知安全漏洞，请及时使用安全补丁进行修复，或在满足业务需求的情况下，将依赖包升级至以下推荐版本。如需安装MindStudio提供的软件外的第三方软件（如三方依赖等），请注意及时升级第三方软件的最新版本，关注并修复存在的漏洞，尤其是已公开的CVSS打分大于7分的高危漏洞。

| 依赖包      | 安全版本                 |
|----------|----------------------|
| torch    | 2.7.1rc1             |
| protobuf | 4.25.8、5.29.5、6.31.1 |
| numpy    | 1.22.0版本以上           |

> [!NOTE] 说明  
> 安全版本依赖包可能不满足业务需求，请根据实际场景，选择合适的版本依赖。
