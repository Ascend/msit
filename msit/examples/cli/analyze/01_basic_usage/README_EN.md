# Basic Usage #

## Introduction ##

Analyze supports Caffe, TensorFlow, and ONNX models migrated to the Ascend hardware platform. The tool supports ATC support analysis, Ascend operator quick query analysis, and ONNX Checker operator constraint analysis. It determines whether operators are supported, whether operator definitions meet constraints, and whether operator input is empty, and provides a detailed analysis report.

## Operating Principle ##

1.  The Analyze tool invokes the ATC conversion tool to convert the model to be evaluated into the JSON format that contains information about all operators in the model.
2.  Perform different analysis based on the conversion result returned by the ATC conversion tool.
    
     *  EVAL_ATC_SUCCESS: After the ATC model is successfully converted, the support degree information about all fusion operators and non-fusion operators is updated based on the OM model information.
     *  EVAL_ATC_UNSUPPORTED_OP_ERR: The ATC conversion result shows that the model contains unsupported operators. Update the information about the unsupported operators, and then use the Ascend operator quick reference tool to analyze whether the operators are supported by the Ascend device. If the input model is in ONNX format, the ONNXChecker class is invoked to check the operator constraints in the model.
     *  EVAL_ATC_Other_ERR: For other types that are not supported, the Ascend operator quick query tool is used to analyze the model. If the model is in ONNX format, the operator constraint check continues.
3.  The tool running results are saved in the specified CSV file. For details, see.`使用示例`chapters

## Use Example ##

```
msit analyze [OPTIONS]
```

OPTIONS parameters are described as follows:

| Parameter           | Description                                                                                                                                                                                                           | Mandatory or not |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| -gm, --golden-model | Input path of the benchmark model. The ONNX, Caffe, and TensorFlow models are supported.                                                                                                                              | Yes              |
| -o, --output        | Output path, in which the analysis result.csv file is generated.                                                                                                                                                      | Yes              |
| --framework         | Model type, and[atc](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/atctool/atlasatcparam_16_0014.html)	The parameter values are the same. 0: Caffe; 3: TensorFlow; 5: ONNX.                | No.              |
| -w, --weight        | Weight file, which is required when the input model is Caffe.                                                                                                                                                         | No.              |
| -soc, --soc-version | Indicates the chip type. If the chip type is not specified, the check is passed.[acl](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclpythondevg_01_0009.html)	Interface Obtaining | No.              |

**Note: When using the analyzer tool to analyze model support on the Atlas 200/500A2 inference product, manually specify the -soc parameter to the specific chip type.**

## Running an Example ##

```
msit analyze -gm /tmp/test.onnx -o /tmp/out
```

```
2023-05-11 11:23:25,824 INFO : convert model to json, please wait...
2023-05-11 11:23:28,210 INFO : convert model to json finished.
2023-05-11 11:23:29,997 INFO : try to convert model to om, please wait...
2023-05-11 11:23:35,127 INFO : try to convert model to om finished.
2023-05-11 11:23:36,321 INFO : analysis result has bean written in /tmp/result.csv
2023-05-11 11:23:36,321 INFO : analyze model finished.
```

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

| Title        | Description                                                                                                                                                                                                                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ori_op_name  | Original operator name.                                                                                                                                                                                                                                                                             |
| ori_op_type  | Original operator type.                                                                                                                                                                                                                                                                             |
| op_name      | Operator name after model migration                                                                                                                                                                                                                                                                 |
| op_type      | Operator type after model migration                                                                                                                                                                                                                                                                 |
| soc_type     | Chip Type                                                                                                                                                                                                                                                                                           |
| engine       | Operator execution engine                                                                                                                                                                                                                                                                           |
| is_supported | Indicates whether the operator is supported. The options are as follows: TRUE: yes; FALSE: no. The possible causes are as follows: The operator is not supported by the current hardware platform, the operator definition does not meet the constraint conditions, or the operator input is empty. |
| details      | Problem description about the operator support, including whether the operator is supported, whether the operator definition meets the constraint conditions, and whether the input is empty.                                                                                                       |

