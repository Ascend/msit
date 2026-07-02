# Basic Usage #

## Introduction ##

Supports the ONNX and OM model precision comparison scenario. The compare precision comparison function can be enabled by running the msit command.

## Running an Example ##

 *  **Note: When using ATC to convert the ONNX model to the OM model, ensure that the input data type of the converted OM model is the same as that of the original ONNX model. (For example, the input_fp16_nodes parameter cannot be used during ATC conversion of the ONNX model input by fp32.)**
 *  **An example of an input command without specifying a model. The path must be an absolute path.**
    
    ```
    msit debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
    -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
    ```
    
     *  `-om, –-om-model`Specifies the offline model (.om) path of the Ascend AI processor.
     *  `-gm, --golden-model`Specifies the path of the model file (PB model, ONNX model, or Caffe model).
     *  `-c，–-cann-path`(Optional) Specify`CANN`Path after the package is installed. If the path is not specified, the system environment variables are obtained by default.`ASCEND_TOOLKIT_HOME`Obtained from`CANN`Package path. If the package path does not exist, the default value is`/usr/local/Ascend/ascend-toolkit/latest`
     *  `-o, –-output`(Optional) Output file path. The default value is the current path.

### Output Result Description and Analysis Procedure Reference ###

For details, see:[Procedure for Comparing Result Analysis](../result_analyse/README.md)	

