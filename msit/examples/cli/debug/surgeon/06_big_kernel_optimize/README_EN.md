# Large Kernel Optimization for Transform Models #

## Introduction ##

During model conversion, the ATC supports a standard pattern fusion pass for the decoder structure of transform models. The entire attention structure is fused into two operators AttentionLnQKV and AttentionScore, improving the model inference performance. Therefore, during the inference migration of transform models, the attention structure of the model is converted into a standard pattern by using certain rules to improve the inference performance of the model. In this scenario, the Surgeon component provides the KnowledgeBigKernel knowledge base optimized by the big kernel, which can automatically convert the original transform model attention structure into the standard pattern.

## Application Scenario Constraints ##

1. The big kernel optimization scenario is valid only for Atlas inference series products.

2. During big kernel optimization, the start node and end node of the first attention structure of the model need to be provided.

3. Models verified: gpt2, bert-base, and bert-large;

## Operating Principle ##

1. Based on the start node and end node of the first attention structure, capture the attention subgraph, construct the attention pattern based on the attention subgraph, and search for the pattern on the entire network to obtain all attention subgraphs.

2. Perform the following operations for each attention submap:

2.1 Parse the attention parameters, including calculating the matmul and add parameters of qkv, and obtain the params parameter.

2.2 Initialize a standard submap.

2.3 Use params to update the parameters in the standard submap.

2.4 Replace the original attention subgraph with the standard subgraph.

3. After all attention submaps are replaced, perform the following post-processing operations:

3.1 The input shape of standard attention and layernorm can only be two dimensions. Therefore, you need to insert the reshape operator before the first layernorm to convert the input into two dimensions, and insert the reshape node after the last layernorm. Convert the output shape to the original shape.

3.2 The standard attention subgraph has an Add node, which adds the result of multiplying qk and mask. It is called qk_mask_add temporarily. In the standard pattern, the qk_mask_add node requires that only the first dimension of input2 be broadcast during addition. If the shapes of other dimensions, such as 0, 2, and 3, are different from those of input1, expand the operation.

3.3 Delete unnecessary nodes and parameters from the graph and save the model.

## Running an Example ##

```
msit debug surgeon opt --input=bert-base-chinese.onnx --output-file=bert-base-chinese_opt.onnx -bk -as attention_start_name -ae attention_end_name
```

