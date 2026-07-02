# Extract Command #

## Introduction ##

Perform subgraph division on the model.

```
msit debug surgeon extract [OPTIONS] [REQUIRED]
```

extract can be abbreviated as ext.

Parameter description:

| Parameter | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Mandatory or not |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| OPTIONS   | Specifies an additional parameter. The options are as follows: -ck/--is-check-subgraph: indicates whether to verify the submap. When this option is enabled, the split submap is verified. -sis/--subgraph-input-shape: extra parameter. You can specify the input shape after the submap is captured. The input shape of multiple nodes is specified in the following format: "input1:n1,c1,h1,w1;input2:n2,c2,h2,w2". -sit/--subgraph_input_dtype: extra parameter. You can specify the input dtype after the submap is captured. The input dtype for multiple nodes is specified in the following format, "input1:dtype1;input2:dtype2". -h, --help: tool help information. | No.              |
| REQUIRED  | -in/--input: Input the ONNX model to be optimized. It must be an .onnx file. -of/--output-file: indicates the ONNX model name of the split submap. It is user-defined and must be an .onnx file. -snn/--start-node-names: name of the start node. You can specify multiple input nodes, which are separated by commas (,). -enn/--end-node-names: indicates the name of the end node. You can specify multiple output nodes, which are separated by commas (,).                                                                                                                                                                                                                | Yes              |

Note: To ensure that the subimage segmentation function works properly and does not affect the inference performance, do not specify an input or output node with a parent-child relationship as a segmentation parameter.

## Running an Example ##

```
msit debug surgeon extract --input=origin_model.onnx --output-file=sub_model.onnx --start-node-names="s_node1,s_node2" --end-node-names="e_node1,e_node2" --subgraph_input_shape="input1:1,3,224,224" --subgraph_input_dtype="input1:float16"
```

An example output is as follows:

```
2023-04-27 14:32:33,378 - 984068 - msit_debug_logger - INFO - Extract the model completed, model was saved in sub_model.onnx
```

