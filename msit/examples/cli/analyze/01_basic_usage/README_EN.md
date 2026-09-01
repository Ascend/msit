# Basic Usage

## Introduction

The analyze tool provides model support analysis for migrating Caffe, TensorFlow, and ONNX models to the Ascend hardware platform. The tool covers three analysis methods: **ATC support analysis**, **Ascend operator quick lookup tool analysis**, and **ONNXChecker operator constraint analysis**. It determines operator support status, whether operator definitions meet constraint conditions, and whether operator inputs are empty. The tool provides a detailed analysis report.

## Working Principle

1. The analyze tool first calls the ATC conversion tool to convert the given model to be evaluated into json format containing all operator information of the model.
2. Different analyses are performed based on the conversion result returned by the ATC conversion tool:
    - EVAL_ATC_SUCCESS: The ATC model conversion is successful. The support information of all fused operators and non-fused operators is then updated based on the om model information.
    - EVAL_ATC_UNSUPPORTED_OP_ERR: The ATC conversion result shows that the model contains unsupported operators. The unsupported operator information is updated first, and then the Ascend operator quick lookup tool is used to analyze the operator support status on the current Ascend device. If the input model is in ONNX format, the ONNXChecker class is called to check the operator constraint conditions in the model.
    - EVAL_ATC_OTHER_ERR: Other unsupported types. The Ascend operator quick lookup tool is used to analyze operator support for the model. If the model is in ONNX format, operator constraint condition checking is continued.
3. The tool running result is saved to the csv file specified by the user. For detailed interpretation, refer to the `Usage Samples` chapter.

## Usage Samples

```shell
msit analyze [OPTIONS]
```

The OPTIONS parameters are described as follows:

| Parameter             | Description                                                         | Required |
|----------------| ------------------------------------------------------------ | -------- |
| -gm, --golden-model | Benchmark model input path. Supports onnx, caffe, and tensorflow models.            | Yes       |
| -o, --output   | Output path. The analysis result **result.csv** is generated under this path.             | Yes       |
| --framework    | Model type. Consistent with the [atc](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/atctool/atlasatcparam_16_0014.html) parameter. 0: caffe, 3: tensorflow, 5: onnx. | No       |
| -w, --weight   | Weight file. Required when the input model is caffe.                  | No       |
| -soc, --soc-version | Chip type. If not specified, it is obtained through the [acl](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclpythondevg_01_0009.html) interface. | No       |

**Special Note**: When using the analyze tool for model support analysis on Atlas 200/500 A2 inference products, manually specify the -soc parameter as the specific chip type.

## Running Samples

```shell
msit analyze -gm /tmp/test.onnx -o /tmp/out
```

```shell
2023-05-11 11:23:25,824 INFO : convert model to json, please wait...
2023-05-11 11:23:28,210 INFO : convert model to json finished.
2023-05-11 11:23:29,997 INFO : try to convert model to om, please wait...
2023-05-11 11:23:35,127 INFO : try to convert model to om finished.
2023-05-11 11:23:36,321 INFO : analysis result has been written in /tmp/result.csv
2023-05-11 11:23:36,321 INFO : analyze model finished.
```

The output result is in result.csv. It records the information and support status of each operator in the model. The result is as follows:

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

| Field         | Description                                                             |
| ------------ |----------------------------------------------------------------|
| ori_op_name  | Original operator name                                                         |
| ori_op_type  | Original operator type                                                         |
| op_name      | Operator name after model migration                                                      |
| op_type      | Operator type after model migration                                                      |
| soc_type     | Chip type                                                           |
| engine       | Operator execution engine                                                         |
| is_supported | Whether the operator is supported. TRUE: supported, FALSE: not supported. Possible reasons include the operator is not supported by the current hardware platform, the operator definition does not meet constraint conditions, or the operator input is empty. |
| details      | Description of the operator support issue, including whether the operator is supported, whether the operator definition meets constraint conditions, and whether the input is empty.                        |
