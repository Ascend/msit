# ONNX Merge Tool Introduction

The ONNX merge tool can merge two ONNX files into one. By using it multiple times, several ONNX files can be merged into a single ONNX file.

This tool is based on the original onnx.compose functionality. It removes the restrictions on custom operator checks and ir version, and encapsulates the parameters.

The command line usage sample of the tool is:

```bash
python combine.py --previous_model_path sub_model1.onnx --following_model_path sub_model2.onnx --merge_model_path merged_model.onnx --previous_model_outputs output1 --following_model_inputs input1
```

It contains five parameters:

1. previous_model_path: The ONNX file path of subgraph A, which is earlier in the topological graph.
2. following_model_path: The ONNX file path of subgraph B, which is later in the topological graph.
3. merge_model_path: The file path of the merged ONNX file.
4. previous_model_outputs: The outputs name of subgraph A.
5. following_model_inputs: The inputs name of subgraph B.

The merge tool matches the **outputs of A** with the **inputs of B** one by one for concatenation. The outputs and inputs names can be obtained through visualization tools.

Take the parameters in the usage sample as an example: the output1 of sub_model1.onnx is connected with the input1 of sub_model2.onnx, thereby merging sub_model1.onnx and sub_model2.onnx into one file. If the subnet has multiple inputs and outputs, you can enter multiple names in the previous_model_outputs and following_model_inputs parameters, for example:

```bash
--previous_model_outputs output1 output2 --following_model_inputs input1 input2
```
