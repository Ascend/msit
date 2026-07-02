# Accuracy Error Location #

## Introduction ##

After the model comparison is complete, demarcate and locate the error of the first precision problem node, determine whether the error is a single-layer error or a accumulated error, and output the error range. The error range is stored in the error_interval_info.txt file in the output directory. This function is disabled by default. When this function is used, run the --locat=True command to set this function.

 *  **Note that dump must be set to True. Currently, the custom-op function does not support parallel operation. Do not enable the custom-op function when using this function.**

## Running an Example ##

**Prerequisites: Currently, dynamic shape models are not supported. During the running, the npy file may fail to be loaded.**`--onnx-fusion-switch False`To resolve.

 *  For a dynamic shape model with a fixed shape, run the onnxsim command to fix all shape information in the model.

```
onnxsim {your_old_onnx_model} {your_new_output_onnx_model}
```

If the ONNX is static, skip this step.

 *  Run the following command to convert the ONNX model to OM:

```
atc --framework=5 --output={your_model_name} --soc_version=<soc_version> --model={your_onnx_model_path}
```

 *  Run the following command to compare the precision:

```
msit debug compare -gm {onnx_model_path} -om {om_model_path} -i {input_data_path} -o {output_file_path} --locat True
```

 *  > onnx_model_path indicates the path of the ONNX file.
 *  > om_model_path indicates the path of the OM file converted by ATC.
 *  > input_data_path indicates the path of the model input file. If no input is specified, -i is not required.
 *  > output_file_path indicates the path of the comparison result.

 *  **Output Result Reference**[01_basic_usage](../01_basic_usage/README.md)	, where the error demarcation and locating information is output in the \{output_path\}/\{timestamp\}/error_interval_info.txt directory.

## The result ##

![content](./说明.png)	

 *  **Result explanation: Every two nodes are the start nodes of a group of error intervals. If the model has multiple inputs, there are multiple error interval node pairs. For example:**

```
Node(Mul_28): 
 inputs=['232', '233']
 outputs=['234']
 attrs = {}
:Node(Gather_1186): 
 inputs=['last_hidden_state', '270']
 outputs=['1632']
 attrs = {'axis': 1}
```

Indicates the error range from the start node Mul_28 to Gather_1186.

