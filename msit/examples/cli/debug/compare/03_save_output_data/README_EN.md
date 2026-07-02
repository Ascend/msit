# Save Output Data #

## Introduction ##

By default, the output is saved in the timestamp (for example, . / 20230601115623) directory in the current directory. The output directory can be specified using -o or --output.

## Running an Example ##

```
msit debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -i /home/HwHiAiUser/result/test/input_0.bin -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
```

In this scenario, the output is saved in /home/HwHiAiUser/result/test/\{timestamp\}.

