# User Guide to the msit convert Function

## Brief Introduction

Based on the Ascend Tensor Compiler (ATC) and Ascend Optimization Engine (AOE), the convert tool converts ONNX, TensorFlow, Caffe, and MindSpore models to OM models and optimizes them.

 * ATC (Ascend Tensor Compiler)

> Ascend Tensor Compiler (Ascend) is a model conversion tool in the heterogeneous computing architecture CANN. It can convert the network model of the open-source framework and the single operator description file (in JSON format) defined by Ascend IR into the .om offline model supported by the Ascend AI processor.
> 
> During model conversion, ATC performs operations such as operator scheduling optimization, weight data rearrangement, and memory usage optimization to further optimize the original deep learning model to meet the high performance requirements in deployment scenarios and efficiently execute the model on the Ascend AI processor.
> 
> [More instructions](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasatc_16_0005.html)    

 * AOE (Ascend Optimization Engine)

> Ascend Optimization Engine (AOE) is an automatic optimization tool. It fully utilizes limited hardware resources to meet the performance requirements of operators and the entire network.
> 
> AOE generates, compiles, and verifies the closed-loop feedback mechanism in the running environment to continuously iterate optimization policies and finally obtain the optimal optimization policies. In this way, AOE makes full use of hardware resources, continuously improves network performance, and achieves the optimal effect.
> 
> [More instructions](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/auxiliarydevtool_0014.html)    

## Tool Installation

 * For details about how to install the tool, see.[Msit Integrated Tool User Guide](../install/README.md)    

## Tool Usage

The command format for using the one-stop msit tool is as follows:

```shell
msit convert [subcommand]
```

Currently, the msit convert command supports the following sub-commands:

| subcommand | Description                              |
| ---------- | ---------------------------------------- |
| atc        | Use ATC to convert models.               |
| aoe        | Using AOE to Convert and Optimize Models |

### ATC command

Use the ATC backend to convert models. The command format is as follows:

```shell
msit convert atc [args]
```

The parameter definition strictly complies with the ATC parameter definition. For details, see the following:[https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasatc_16_0039.html\#ZH-CN_TOPIC_0000001949484154__section6351244132417](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasatc_16_0039.html#ZH-CN_TOPIC_0000001949484154__section6351244132417)    Example:

```shell
msit convert atc --model resnet50.onnx --framework 5 --soc_version <soc_version> --output resnet50
```

### aoe command

Use the AOE backend to convert models. The command format is as follows:

```shell
msit convert aoe [args]
```

The parameter definition strictly complies with the AOE parameter definition. For details, see the following:https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/auxiliarydevtool_0014.html

Example:

```shell
msit convert aoe --model resnet50.onnx --job_type 2 --output resnet50
```

## FAQ

If you encounter any problem when using the convert component to convert models, see[FAQ](FAQ.md)    
