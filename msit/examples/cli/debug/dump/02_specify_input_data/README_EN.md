# Specify Input Data #

## Introduction ##

If you want to specify a model input file, you can use parameters.`--input`Designated

 *  **Note: If you are importing for a folder, make sure that all input files used are .bin files.**

## Running an Example ##

**Specify the model input command example. The path must be an absolute path.**

```
msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
-i /home/HwHiAiUser/result/test/input_0.bin -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
```

 *  `-i, –-input`Input data path of the model. By default, the input data path is randomly generated based on the input parameter of the model. Separate multiple input data paths with commas (,). For example:`/home/input_0.bin,/home/input_1.bin`. During inference, the batch size is calculated based on the input shape and model definition shape. Ensure that the shape of the input file is different from that of the model definition only in the batch dimension.

```
msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
-i /home/HwHiAiUser/result/test/input_0.npy -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
```

 *  `-i, –-input`If the npy file exists, the Npy file can be automatically converted to a bin file. For example:`/home/input_0.npy,/home/input_1.npy`

