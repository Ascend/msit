# Custom Ops #

## Introduction ##

This case applies to the custom operator development scenario. Based on the Surgeon component, you can add customized operators to the ONNX diagram.

Note: The added custom operator in this case also adapts to all functions of the image modification API of the Surgeon component, but it is not guaranteed that the inference test can be performed. This case aims to provide a simple and feasible solution for adding custom operators.

## Running an Example ##

The following code example shows the process of replacing an operator in an ONNX diagram with custom operator. You can add only customized operators as required.

```
from auto_optimizer import OnnxGraph, OnnxNode

g = OnnxGraph.parse("model.onnx")

#Find the node to be replaced based on the operator name.
ori_node = g.get_node("ori_node_name", node_type=OnnxNode)

#Added the custom operator.
custom_op = g.add_node(
    "custom_op",          #Operator name
    "CustomOpType"        #Operator type (customized type)
)

#After the new custom operator is connected to the original operator
g.insert_node("ori_node_name", custom_op, refer_index=0) #refer_index=0 indicates that refer_index is inserted after the reference operator.

#Delete the original operator.
g.remove("ori_node_name")

#Saving Models
g.update_map()
g.save("model_with_custom_op.onnx")
```

