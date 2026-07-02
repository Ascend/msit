# Alone Compare #

## Introduction ##

You can also directly transfer the dump data path to the model for precision comparison without using parameters.

## Running an Example ##

 *  Example command
 *  **Note: The -mp and -gp paths must be the upper-level folder of the data.**
    
    ```
    msit debug compare -mp /home/HwHiAiUser/onnx_prouce_data/resnet_offical_om -gp /home/HwHiAiUser/onnx_prouce_data/model/resnet_offical_onnx
    --ops-json /home/HwHiAiUser/onnx_prouce_data/resnet_offical/ops.json -o /home/HwHiAiUser/result/test
    ```
    
     *  `-mp, --my-path`Specifies the dump data path on the NPU.
     *  `-gp, --golden-path`Specifies the dump data path on the CPU.
     *  `--ops-json`Specifies the JSON file path of the matching rule between the CPU and NPU operators.
     *  `-o, –-output`(Optional) Output file path. The default value is the current path.
 *  For details about how to obtain dump data on the CPU and NPU, see.[Msit debug dump function](../../../../../docs/debug/dump/README.md)	

### Output Result Description and Analysis Procedure Reference ###

For details, see:[Procedure for Comparing Result Analysis](../result_analyse/README.md)	

