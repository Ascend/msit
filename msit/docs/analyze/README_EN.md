# msit analyze Feature Usage Guide

## Introduction

The model support analysis tool provides operator support analysis, operator definition constraint validation, and operator input emptiness checking.

## Tool Installation

### Prerequisites

- Install the Ascend AI inference driver, firmware, and CANN package for the development and running environment. Refer to [CANN](https://www.hiascend.com/cann/download).
- After installation, set the **CANN_PATH environment variable** to specify the CANN version path, for example: `export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest`.
- If you do not set this variable, the tool attempts to obtain the CANN version from the `ASCEND_TOOLKIT_HOME` environment variable and the `/usr/local/Ascend/ascend-toolkit/latest` path respectively.

### Installation Methods

The installation methods include **source code installation** and **pip installation**. Select one as needed.

- [Source code installation](#source-code-installation): Install from source to obtain the latest msit features.
- [pip installation](#pip-installation): Install the msit package through pip, which is released quarterly.

#### Source Code Installation

```shell
git clone https://gitcode.com/Ascend/msit.git
# 1. git pull origin to update the latest code
cd msit/msit

# 2. Install the msit package
pip install .

# 3. Install analyze:
msit install analyze

# 4. After installation, use the msit check command to verify the installation:
msit check analyze
```

#### pip Installation

```shell
# 1. Install the msit package
pip install msit

# 2. Install analyze:
msit install analyze

# 3. After installation, use the msit check command to verify the installation:
msit check analyze
```

## Tool Usage

The command format for the msit tool is as follows:

```shell
msit analyze [OPTIONS]
```

The OPTIONS parameters are described as follows:

| Parameter | Description | Required |
|-----------|-------------|----------|
| -gm, --golden-model | Golden model input path. Supports ONNX, Caffe, and TensorFlow models. | Yes |
| -o, --output | Output path. The analysis result **result.csv** is generated in this path. | Yes |
| --framework | Model type. Consistent with the [atc](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/atc/atlasatc_16_0005.html) parameter. 0: Caffe, 3: TensorFlow, 5: ONNX. | No |
| -w, --weight | Weight file. Required when the input model is a Caffe model. | No |
| -soc, --soc-version | Chip type. If not specified, the tool obtains it through the ACL interface. | No |
| -h, --help | Command-line parameter help information. | No |

**Note:** When using the analyze tool for model support analysis on Atlas 200/500 A2 inference products, manually specify the `-soc` parameter to the specific chip type.

The command sample and output are as follows:

```shell
msit analyze -gm /tmp/test.onnx -o /tmp/out
```

After execution, the last line prints `analyze model finished`.

The output result is in `result.csv`, which records the information and support status of each operator in the model. The result is as follows:

| ori_op_name           | ori_op_type        | op_name | op_type         | soc_type  | engine  | is_supported | details                                                      |
| --------------------- | ------------------ | ------- | --------------- | --------- | ------- | ------------ | ------------------------------------------------------------ |
| Reshape_46            | Reshape            |         | Reshape         | Ascend310 | AICORE  | TRUE         |                                                              |
| Cast_47               | Cast               |         | Cast            | Ascend310 | AICORE  | TRUE         |                                                              |
| Pad_49                | Pad                |         | PadV3           | Ascend310 | AICORE  | TRUE         |                                                              |
| Conv_52               | Convx              |         |                 | Ascend310 | UNKNOWN | FALSE        | No Op registered for Convx with domain_version of 11;Op is unsupported. |
| Transpose_53          | Transpose          |         | PartitionedCall | Ascend310 | AICORE  | TRUE         |                                                              |
| LeakyRelu_54          | LeakyRelu          |         | LeakyRelu       | Ascend310 | AICORE  | TRUE         |                                                              |
| BatchNormalization_60 | BatchNormalization |         | BatchNorm       | Ascend310 | AICORE  | TRUE         |                                                              |
| Shape_61              | Shape              |         | Shape           | Ascend310 | AICORE  | TRUE         |                                                              |

Output data description:

| Column | Description |
| ------ | ----------- |
| ori_op_name  | Original operator name |
| ori_op_type  | Original operator type |
| op_name      | Operator name after model migration |
| op_type      | Operator type after model migration |
| soc_type     | Chip type |
| engine       | Operator execution engine |
| is_supported | Whether the operator is supported. TRUE: supported. FALSE: unsupported. Possible reasons include the operator is not supported by the current hardware platform, the operator definition does not meet constraint conditions, or the operator input is empty. For specific reasons, refer to the details field. |
| details      | Problem description of the operator support status, including whether the operator is supported, whether the operator definition meets constraint conditions, and whether the input is empty. |

## FAQ

- For installation issues, refer to [Installation FAQ](../install/FAQ.md) first.
- For usage issues, refer to [Usage FAQ](FAQ.md) first.
