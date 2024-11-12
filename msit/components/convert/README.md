# msit convert功能使用指南

## 简介

convert模型转换工具依托ATC（Ascend Tensor Compiler），AOE（Ascend Optimization Engine），MindIE-RT（Ascend MindIE Runtime）推理引擎，提供由ONNX、TensorFlow、Caffe、MindSpore模型至om模型的转换及调优功能。

## 工具安装

- 如果使用MindIE-RT做模型转换，需要在安装convert前安装MindIE，MindIE安装请参考MindIE[官方网站](https://www.hiascend.com/software/mindie)

  注：MindIE-RT的python API调用要求python为3.10版本，在其他python版本下安装会提示如下错误，请忽略，不影响模型转换。
  ```shell
  AscendIE python api Install failed, please install python3.10 firstly!
  ```

- 工具安装请见 [msit一体化工具使用指南](../../docs/install/README.md)

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
| aie        | 使用mindie-rt进行模型转换(aie现已合入MindIE)|

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
使用MindIE-RT后端进行模型转换，目前仅支持ONNX模型的转换，命令格式如下：
```shell
msit convert aie [args]
```
参数说明如下：

| 参数                  | 说明                                                      | 是否必选 |
|---------------------|---------------------------------------------------------|------|
| -gm, --golden-model | 标杆模型输入路径，支持onnx模型                                       | 是    |
| -of, --output-file  | 输出文件，需要有后缀 .om, 当前支持基于 MindIE-RT(Ascend MindIE Runtime) 的模型转换 | 是    |
| -soc, --soc-version | 芯片类型                 | 是    |

命令示例如下：

```shell
msit convert aie --golden-model resnet50.onnx --output-file resnet50.om --soc-version <soc_version>
```

#### 使用案例
更多关于aie子命令的介绍请移步[convert工具使用示例](../../examples/cli/convert/01_basic_usage)


## FAQ
使用convert组件进行模型转换时如遇问题，请先行查阅[FAQ](FAQ.md)