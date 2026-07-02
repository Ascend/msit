# Specify Input Data #

## Introduction ##

By default, data whose values are all 0s is constructed and sent to the model for precision comparison. File input or folder input can be specified.

 *  **Note: If entering for a folder, make sure that all input files used are .bin files.**

## Running an Example ##

**Example of the input command for a specified model. The path must be an absolute path.**

```
msit debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
-i /home/HwHiAiUser/result/test/input_0.bin -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
```

 *  `-i，–-input`Input data path of the model. By default, the input data path is randomly generated based on the input parameter of the model. Separate multiple input data paths with commas (,). For example:`/home/input_0.bin,/home/input_1.bin`. During inference, the batch size is calculated based on the input shape and model definition shape. Ensure that the shape of the input file is different from that of the model definition only in the batch dimension.

```
msit debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
-i /home/HwHiAiUser/result/test/input_0.npy -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
```

 *  `-i，–-input`If the npy file is specified, the Npy file can be automatically converted to a .bin file without manual conversion. For example:`/home/input_0.npy,/home/input_1.npy`.

