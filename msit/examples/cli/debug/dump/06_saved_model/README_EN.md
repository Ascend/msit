# Saved model dump #

## Introduction ##

Tensor data can be dumped for models in save_model format saved using the TensorFlow framework.

## Use Example ##

 *  1. Example of the saved_model npu dump command. The path must be an absolute path.
    
    ```
    msit debug dump -m /home/HwHiAiUser/prouce_data/resnet_offical_saved_model -dp npu
    -is "模型的输入shape" -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
    ```
    
    2. Example of the saved_model cpu dump command. The path must be an absolute path. The ge_xxx.json file in the model folder must be transferred to the cpu after the npu dump as the --tf-json parameter.
    
    ```
    msit debug dump -m /home/HwHiAiUser/prouce_data/resnet_offical_saved_model -dp cpu
    -is "模型的输入shape" -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test 
    --tf-json  /home/HwHiAiUser/result/test/{date}/model/ge_xxx.json
    ```
    
     *  `-m, –-model`Specify the original saved_model model path.
     *  `-dp, --device-pattern`Specifies the type of the device to be dumped. The value can be CPU or NPU. Currently, only the saved_model model is supported in NPU mode.
     *  `-is, --input-shape`(Optional) Input shape of the model
     *  `-c，–-cann-path`(Optional) Specify`CANN`Path after the package is installed. If the path is not specified, the system environment variables are obtained by default.`ASCEND_TOOLKIT_HOME`Obtained from`CANN`Package path. If the package path does not exist, the default value is`/usr/local/Ascend/ascend-toolkit/latest`
     *  `-o, –-output`(Optional) Output file path. The default value is the current path.
     *  `--tf-json`Specifies the JSON file related to the GE graph generated during the NPU dump. This parameter is mandatory for the CPU dump.

