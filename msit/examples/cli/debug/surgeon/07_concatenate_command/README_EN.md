# Concatenate Command #

## Introduction ##

Combine the two ONNX diagrams based on the specified input and output mapping.

Note: If a name conflict is detected between two ONNX diagrams to be spliced, the system automatically adds the prefix "pre_" to the names of all components in the first ONNX diagram.

```
msit debug surgeon concat [OPTIONS]
```

concatenate can be abbreviated as concat.

Parameter description:

| Parameter | Description                                                                                                                                                                                                                                                                                                                                              | Mandatory or not |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| OPTIONS   | Additional parameter. The options are as follows: -cgp/--combined-graph-path: name of the structure diagram after the combination. The default value is the name of two graphs connected by underscores (_).-pref/--prefix: prefix character string added to the first ONNX graph. The default value is pre_.-h, --help: tool help information.          | No.              |
| REQUIRED  | -g1/--graph1: indicates the first ONNX model to be imported. The model must be an .onnx file. -g2/--graph2: indicates the first ONNX model to be imported. The model must be an .onnx file. -io/--io-map: mapping between the output of the first image and the input of the second image during stitching. For example, "g1_out1,g2_in1;g1_out2,g2_in2" | Yes              |

## Running an Example ##

```
msit debug surgeon concat -g1 sub1.onnx -g2 sub2.onnx -io "g1_out1,g2_in1;g1_out2,g2_in2"
```

An example output is as follows:

```
2023-07-19 10:45:31,237 - 984068 - msit_debug_logger - WARNING - Cant merge two graphs with overlapping names. Found repeated nodes names: conv4_10/x1/bn_1_QuantizeLinear,Add
_nc_rename_45_quant,conv4_24/x1/bn_1_QuantizeLinear,Add_nc_rename_257_quant,concat_4_7_1_DequantizeLinear,Add_nc_rename_415_quant,BatchNormalization_nc_rename_453,Add_nc
_rename_67_quant...
2023-07-19 10:45:31,240 - 984068 - msit_debug_logger - INFO - A prefix `pre_` will be added to graph1
2023-07-19 10:45:31,510 - 984068 - msit_debug_logger - INFO - Concatenate ONNX model: densenet-12-int8.onnx and ONNX model: densenet-12-int8.onnx completed. Combined model sa
ved in res.onnx
```

