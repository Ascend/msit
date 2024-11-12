# msit convert功能使用指南

## 简介

convert模型转换工具依托ATC（Ascend Tensor Compiler），AOE（Ascend Optimization Engine），AIE（Ascend Inference Engine）推理引擎，提供由ONNX、TensorFlow、Caffe、MindSpore模型至om模型的转换及调优功能。

* ATC（Ascend Tensor Compiler）
> 昇腾张量编译器（Ascend Tensor Compiler，简称ATC）是异构计算架构CANN体系下的模型转换工具， 它可以将开源框架的网络模型以及Ascend IR定义的单算子描述文件（json格式）转换为昇腾AI处理器支持的.om格式离线模型。
>
> 模型转换过程中，ATC会进行算子调度优化、权重数据重排、内存使用优化等具体操作，对原始的深度学习模型进行进一步的调优，从而满足部署场景下的高性能需求，使其能够高效执行在昇腾AI处理器上。
>
> [更多说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/devaids/auxiliarydevtool/atlasatc_16_0005.html)
* AOE（Ascend Optimization Engine）
> AOE（Ascend Optimization Engine）是一款自动调优工具，作用是充分利用有限的硬件资源，以满足算子和整网的性能要求。
>
> AOE通过生成调优策略、编译、在运行环境上验证的闭环反馈机制，不断迭代出更优的调优策略，最终得到最佳的调优策略，从而可以更充分利用硬件资源，不断提升网络的性能，达到最优的效果。
>
> [更多说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/devaids/auxiliarydevtool/aoe_16_001.html)
* AIE（Ascend Inference Engine）
> 是针对昇腾AI处理器的推理加速引擎，提供AI模型推理场景下的商业化部署能力，能够将不同的深度学习框架（PyTorch、ONNX等）上完成训练的算法模型统一为计算图表示，具备多粒度模型优化、整图下发以及推理部署等功能。
>
> [更多说明](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/ascendIE/ascendIE_0009.html)

## 工具安装

- 如果使用aie做模型转换，需要在安装convert前安装AIE，AIE安装请参考AIE[安装指导](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/AscendIE/AscendIE/readme.md#%E5%AE%89%E8%A3%85%E5%92%8C%E9%83%A8%E7%BD%B2)

  注：aie要求python3.9，在python3.7下安装aie会提示如下错误，请忽略，不影响模型转换。
  ```shell
  AscendIE python api Install failed, please install python3.9 firstly!
  ```

- 工具安装请见 [msit一体化工具使用指南](../../README.md)

## 工具使用

一站式msit工具使用命令格式说明如下：
```shell
msit convert [subcommand]
```
msit convert目前支持以下3种子命令：

| subcommand | 说明                      |
| ---------- | ------------------------- |
| atc        | 使用atc进行模型转换       |
| aoe        | 使用aoe进行模型转换及调优 |
| aie        | 使用aie进行模型转换       |

### atc命令
使用ATC后端进行模型转换，命令格式如下：
```shell
msit convert atc [args]
```
参数定义严格遵从ATC的参数定义，由于参数较多，详情可参考：https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/inferapplicationdev/atctool/atctool_000041.html

使用示例：
```shell
msit convert atc --model resnet50.onnx --framework 5 --soc_version <soc_version> --output resnet50
```
### aoe命令
使用AOE后端进行模型转换，命令格式如下：
```shell
msit convert aoe [args]
```
参数定义严格遵从AOE的参数定义，由于参数较多，详情可参考：https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/devtools/auxiliarydevtool/aoepar_16_001.html

使用示例：
```shell
msit convert aoe --model resnet50.onnx --job_type 2 --output resnet50
```
### aie命令
使用AIE后端进行模型转换，目前仅支持ONNX模型的转换，命令格式如下：
```shell
msit convert aie [args]
```
参数说明如下：

| 参数                  | 说明                                                      | 是否必选 |
|---------------------|---------------------------------------------------------|------|
| -gm, --golden-model | 标杆模型输入路径，支持onnx模型                                       | 是    |
| -of, --output-file  | 输出文件，需要有后缀 .om, 当前支持基于 AIE(Ascend Inference Engine) 的模型转换 | 是    |
| -soc, --soc-version | 芯片类型                 | 是    |

命令示例如下：

```shell
msit convert aie --golden-model resnet50.onnx --output-file resnet50.om --soc-version <soc_version> 
```

#### 使用案例
更多关于aie子命令的介绍请移步[convert工具使用示例](../../examples/cli/convert/01_basic_usage)


## FAQ
使用convert组件进行模型转换时如遇问题，请先行查阅[FAQ](FAQ.md)