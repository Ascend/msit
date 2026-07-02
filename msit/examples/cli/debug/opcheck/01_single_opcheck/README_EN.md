# Precision precheck of a single operator #

## Introduction ##

This tool can pre-check the precision of operators that are not optimized by compilation. It can be used to quickly demarcate error accumulation problems and check whether the precision of operators at the kernel level meets the requirements.

## Environment Dependency ##

```
pip install torch==1.11.0
```

## Running an Example ##

### Input Data ###

Run the following command to dump the tensor data during model inference:`msit debug dump -m /home/HwHiAiUser/prouce_data/resnet_offical_saved_model -dp npu -is {input_shape} -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test`, of which`{input_shape}`is the input shape of the model and needs to be modified based on the site requirements.

 *  Reference for Related Parameters[Reference for Using the saved_model dump Function](../../dump/06_saved_model/README.md)	
 *  The generated tensor information is flushed to the disk in the directory specified by the -o parameter.

### Using the entrance ###

The pre-check function can be enabled by using the msit command line tool. The startup mode is as follows: The path in the command example must be an absolute path. The tensor path specified by the -i parameter is flushed to disks by running the msit debug dump command.

```
msit debug opcheck -i /home/HwHiAiUser/result/test/{timestamp} -o /home/HwHiAiUser/result/test/opcheck_test
```

## Output Description ##

The precheck result file result_\{timestamp\}.xlsx contains two sheets: opcheck_results (operators that are successfully prechecked) and addition_failed_cases (operators that fail to be prechecked).

### Precheck Result Column Name Description ###

| Column Name           | Description                                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| op_type               | Operator type, in the format of operator class name contained in list.                                                        |
| op_name               | Operator name, which is separated by underscores (_).                                                                         |
| op_param              | Operator parameter                                                                                                            |
| tensor_path           | File name of the operator input and output intensor.                                                                          |
| out_tensor_id         | Sequence number of the operator output extensor. (Some operator outputs may have multiple extensors.)                         |
| rel_precision_rate(%) | Actual accuracy pass rate (using relative error, 100% if all passes)                                                          |
| max_rel_error         | Maximum relative error                                                                                                        |
| abs_precision_rate(%) | Actual absolute error precision pass rate                                                                                     |
| max_abs_error         | Maximum absolute error                                                                                                        |
| cosine_similarity     | cosine similarity                                                                                                             |
| kl_divergence         | kl divergence                                                                                                                 |
| fail_reason           | Failure cause, including the precision does not reach the specified standard or the cause of the algorithm execution failure. |

The related calculation formula is:

Relative error: rel_error = abs(actual_output - golden_output) / abs(golden_output) precision: Pass rate: rel_precision_rate = sum(rel_error <= etol) / size(rel_error) \* 100 precision: precision_result = bool(rel_precision_rate >= pass_rate) precision

## Supported precision precheck operators ##

| Operator name (A-G) | Operator name (H-M) | Operator name (N-R) | Operator name (S-Z) |
| ------------------- | ------------------- | ------------------- | ------------------- |
| Add                 | LogicalOr           | Pad                 | SoftMaxV2           |
| Adds                | LogicalAnd          | PadD                | Sub                 |
| BatchNorm           | Less                | Pack                | Sigmoid             |
| BatchMatMulV2       | Mul                 | ReduceMean          | StrideSlice         |
| BiasAdd             | Minimum             | Rsqrt               | Tanh                |
| BNInfer             | Mul                 | ReduceMean          | Tile                |
| ConcatV2            | MatMulV2            | ReduceSum           | Transpose           |
| Conv2D              |                     | Relu                |                     |
| ClipByValue         |                     |                     |                     |
| GatherV2            |                     |                     |                     |

## FAQ ##

1.  Why do I need to use PyTorch 1.11.0?
    
    The benchmark implementation uses PyTorch 1.11.0. If PyTorch of another version is used, the operator implementation may be different.
2.  Why Is the BatchNorm Operator Precheck Abnormal?
    
    The benchmark implementation of the BatchNorm operator uses the torch.ops.aten.native_batch_norm implementation of the PyTorch. During the calculation, exceptions such as inf and south may occur.

