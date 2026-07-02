# Introduction to the ONNX Combination Tool #

The ONNX merge tool can merge two ONNX files into one, and can be used multiple times to merge several ONNX files into one ONNX file.

Based on the original onnx.compose function, this tool eliminates the restrictions on the custom operator check and ir version, and encapsulates parameters.

The following is an example of using the command line of the tool:

```
python combine.py --previous_model_path sub_model1.onnx --following_model_path sub_model2.onnx --merge_model_path merged_model.onnx --previous_model_outputs output1 --following_model_inputs input1
```

The following parameters are contained:

1.  previous_model_path: ONNX file path of submap A in the topological view
2.  following_model_path: ONNX file path of submap B in the topology view
3.  merge_model_path: path of the merged ONNX file
4.  previous_model_outputs: output name of submap A
5.  following_model_inputs: name of inputs in submap B

The combination tool combines outputs of A and inputs of B in one-to-one mapping. The output and input names can be obtained by using the visualization tool.

For example, output1 of sub_model1.onnx is connected to input1 of sub_model2.onnx to merge sub_model1.onnx and sub_model2.onnx into one file. If the subnet has multiple inputs and outputs, you can enter multiple names in the previous_model_outputs and following_model_inputs parameters, for example,

```
--previous_model_outputs output1 output2 --following_model_inputs input1 input2
```

