# msit convert Usage Guide

## Introduction

The convert model conversion tool is based on ATC (Ascend Tensor Compiler) and AOE (Ascend Optimization Engine). It provides conversion and tuning functions from ONNX, TensorFlow, Caffe, and MindSpore models to om models.

* ATC (Ascend Tensor Compiler)

> The Ascend Tensor Compiler (ATC) is a model conversion tool under the heterogeneous computing architecture CANN system. It can convert network models from open-source frameworks and single operator description files (in json format) defined by Ascend IR into .om format offline models supported by the Ascend AI processor.
>
> During model conversion, ATC performs operator scheduling optimization, weight data rearrangement, and memory usage optimization to further tune the original deep learning model, thereby meeting the high-performance requirements in deployment scenarios and enabling it to execute efficiently on the Ascend AI processor.
>
> [More Information](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasatc_16_0005.html)

* AOE (Ascend Optimization Engine)

> AOE (Ascend Optimization Engine) is an automatic tuning tool. Its purpose is to fully utilize limited hardware resources to meet the performance requirements of operators and the entire network.
>
> AOE continuously iterates better tuning strategies through a closed-loop feedback mechanism of generating tuning strategies, compiling, and verifying on the running environment. It ultimately obtains the best tuning strategy, thereby making fuller use of hardware resources, continuously improving network performance, and achieving optimal results.
>
> [More Information](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/auxiliarydevtool_0014.html)

## Tool Installation

### Installation Prerequisites

- Install the Ascend AI inference-related driver, firmware, and CANN package for the development and running environment. Refer to [CANN](https://www.hiascend.com/cann/download).
- After installation, you can specify the installed CANN version path by **setting the CANN_PATH environment variable**, for example: export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest.
- If not set, the tool attempts to obtain the CANN version from the environment variable ASCEND_TOOLKIT_HOME and the path /usr/local/Ascend/ascend-toolkit/latest respectively.

### Installation Methods

The installation methods include: **source code installation** and **pip source installation**. You can choose as needed.

- [Source code installation](#source-code-installation): Install from source code to ensure the latest msit functionality.
- [pip source installation](#pip-source-installation): Install the msit package through pip. Packages are released once a quarter.

#### Source Code Installation

```shell
git clone https://gitcode.com/Ascend/msit.git
# 1. git pull origin to update the latest code
cd msit/msit

# 2. Install the msit package
pip install .

# 3. Install convert:
msit install convert

# 4. After installation, use the msit check command to verify the installation:
msit check convert
```

#### pip Source Installation

```shell
# 1. Install the msit package
pip install msit

# 2. Install convert:
msit install convert

# 3. After installation, use the msit check command to verify the installation:
msit check convert
```

## Tool Usage

The command format for the one-stop msit tool is as follows:

```shell
msit convert [subcommand]
```

msit convert currently supports the following 2 subcommands:

| subcommand | Description                      |
| ---------- | ------------------------- |
| atc        | Use atc for model conversion       |
| aoe        | Use aoe for model conversion and tuning |

### atc Command

Use the ATC backend for model conversion. The command format is as follows:

```shell
msit convert atc [args]
```

The parameter definitions strictly follow the ATC parameter definitions. For details, refer to: <https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasatc_16_0039.html#ZH-CN_TOPIC_0000001949484154__section6351244132417>

Usage sample:

```shell
msit convert atc --model resnet50.onnx --framework 5 --soc_version <soc_version> --output resnet50
```

### aoe Command

Use the AOE backend for model conversion. The command format is as follows:

```shell
msit convert aoe [args]
```

The parameter definitions strictly follow the AOE parameter definitions. For details, refer to: <https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/auxiliarydevtool_0014.html>

Usage sample:

```shell
msit convert aoe --model resnet50.onnx --job_type 2 --output resnet50
```

## FAQ

- For issues during installation, first refer to the [FAQ](../install/FAQ.md).
