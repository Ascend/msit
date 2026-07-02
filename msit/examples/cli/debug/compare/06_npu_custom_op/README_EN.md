# Npu Custom op #

## Introduction ##

Some Ascend models have NPU custom operator, such as[Retinanet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet)	In, the BatchMultiClassNMS post-processing operator exists. This operator cannot run on the onnxruntime. As a result, the precision of the model cannot be compared using the msit debug compare function. In this case, you can add the --custom-op parameter to specify the custom operator type name. The tool deletes these operators so that the operators can be properly inferred and dump data can be obtained.

## Application Scenario Constraints ##

1. The benchmark model must be an ONNX file. The input parameter \[-gm, --golden-model\] must be an ONNX model.

2. Currently, the value range of custom-op is BatchMultiClassNMS, DeformableConv2D, and RoiExtractor.

3. Do not disable --dump or enable advanced functions such as --locat and --single-op.

## Operating Principle ##

 *  Delete all custom operator in the original model, and add the customized operator output node to the inputs node of the overall model.
 *  On the NPU, run the OM model to obtain the dump data of the NPU.
 *  Obtain the output data of the NPU custom operator through the NPU dump data and transfer the data to the ONNX model.
 *  Run the ONNX model whose custom operator is deleted on the CPU to obtain the benchmark dump data.
 *  Invoke the precision comparison tool in the CANN package to compare the ONNX dump data with the NPU dump data.

## Running an Example ##

In the name of[Retinanet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet)	For example, after the ONNX model is obtained, run the following command to check whether the custom operator type in the model is BatchMultiClassNMS:

```
msit debug compare -gm ./model.onnx -om ./model.om -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test  --custom-op='BatchMultiClassNMS'
```

 *  `--custom-op`Indicates the name of the custom operator type in the ONNX model.

Multiple custom operator types are also supported and separated by commas (,).

```
msit debug compare -gm ./model.onnx -om ./model.om -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test  --custom-op='BatchMultiClassNMS,RoiExtractor'
```

