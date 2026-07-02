# Basic Usage #

## Introduction ##

The dump function can be enabled by running the msit command. You can use the -m parameter to specify a model file. Currently, save_model models saved by ONNX, OM, and TensorFlow are supported.

## Running an Example ##

 *  **An example of an input command without specifying a model. The path must be an absolute path.**
    
    ```
    msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
    -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
    ```
    
     *  `-m, –-model`Specify the original offline model (.onnx) path.
     *  `-dp, --device-pattern`Specifies the type of the device to be dumped. The value can be CPU or NPU. Currently, only the saved_model model is supported in NPU mode.
     *  `-c，–-cann-path`(Optional) Specify`CANN`Path after the package is installed. If the path is not specified, the system environment variables are obtained by default.`ASCEND_TOOLKIT_HOME`Obtain the value in. If the environment variable does not exist, the default value is`/usr/local/Ascend/ascend-toolkit/latest`
     *  `-o, –-output`(Optional) Output file path. The default value is the current path.

