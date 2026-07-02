# msit analyze user guide

## Brief Introduction

The model support analysis tool analyzes the operator support, whether the operator definition meets the constraint conditions, and whether the operator input is empty.

## Tool Installation

 * For details about how to install the msit tool, see.[MSIT Integrated Tool User Guide](../../docs/install/README.md)    
 * After the msit is installed, run the msit install analyze command to install the analyze component.

## Tool Usage

The command format for using the one-stop msit tool is as follows:

```shell
msit analyze [OPTIONS]
```

OPTIONS parameters are described as follows:

| Parameter           | Description                                                                                                                                                                                          | Mandatory or not |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| -gm, --golden-model | Input path of the benchmark model. The ONNX, Caffe, and TensorFlow models are supported.                                                                                                             | Yes              |
| -o, --output        | Output path. The analysis result result.csv file is generated in the path.                                                                                                                           | Yes              |
| --framework         | Model type, and[atc](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/atc/atlasatc_16_0005.html)    The parameter values are the same. 0: Caffe; 3: TensorFlow; 5: ONNX. | No.              |
| -w, --weight        | Weight file. This file is required when the input model is Caffe.                                                                                                                                    | No.              |
| -soc, --soc-version | Indicates the chip type. If the chip type is not specified, the value is obtained through the ACL interface.                                                                                         | No.              |
| -h, --help          | Command line parameter help information                                                                                                                                                              | No.              |

**Note: When using the analyzer tool to analyze model support on the Atlas 200/500 A2 inference product, manually specify the -soc parameter to the specific chip type.**

The following is an example command and output:

```shell
msit analyze -gm /tmp/test.onnx -o /tmp/out
```

After the execution is complete, analyze model finished is displayed in the last line.

The output result is stored in the result.csv file, which records the information and support status of each operator in the model. The result is as follows:

| ori_op_name           | ori_op_type        | op_name | op_type         | soc_type  | engine  | is_supported | details                                                                 |
| --------------------- | ------------------ | ------- | --------------- | --------- | ------- | ------------ | ----------------------------------------------------------------------- |
| Reshape_46            | Reshape            |         | Reshape         | Ascend310 | AICORE  | TRUE         |                                                                         |
| Cast_47               | Cast               |         | Cast            | Ascend310 | AICORE  | TRUE         |                                                                         |
| Pad_49                | Pad                |         | PadV3           | Ascend310 | AICORE  | TRUE         |                                                                         |
| Conv_52               | Convx              |         |                 | Ascend310 | UNKNOWN | FALSE        | No Op registered for Convx with domain_version of 11;Op is unsupported. |
| Transpose_53          | Transpose          |         | PartitionedCall | Ascend310 | AICORE  | TRUE         |                                                                         |
| LeakyRelu_54          | LeakyRelu          |         | LeakyRelu       | Ascend310 | AICORE  | TRUE         |                                                                         |
| BatchNormalization_60 | BatchNormalization |         | BatchNorm       | Ascend310 | AICORE  | TRUE         |                                                                         |
| Shape_61              | Shape              |         | Shape           | Ascend310 | AICORE  | TRUE         |                                                                         |

Output data description:

| Title        | Description                                                                                                                                                                                                                                                                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ori_op_name  | Original operator name.                                                                                                                                                                                                                                                                                                                           |
| ori_op_type  | Original operator type.                                                                                                                                                                                                                                                                                                                           |
| op_name      | Operator name after model migration                                                                                                                                                                                                                                                                                                               |
| op_type      | Operator type after model migration                                                                                                                                                                                                                                                                                                               |
| soc_type     | Chip Type                                                                                                                                                                                                                                                                                                                                         |
| engine       | Operator execution engine                                                                                                                                                                                                                                                                                                                         |
| is_supported | Indicates whether the operator is supported. The options are TRUE (supported) and FALSE (not supported). The possible causes are as follows: The operator is not supported by the current hardware platform, the operator definition does not meet the constraint conditions, or the operator input is empty. For details, see the details field. |
| details      | Problem description about the operator support, including whether the operator is supported, whether the operator definition meets the constraint conditions, and whether the input is empty.                                                                                                                                                     |

## FAQ

 * If you have any problems during use, you can check them first.[FAQ](FAQ.md)    
