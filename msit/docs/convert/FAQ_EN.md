# FAQ

## 1. An error is reported when the msit convert atc command is used for conversion (ONNX format)

**Error Message E16005: The model has \[2\] \[--domain version\] fields, but only one is allowed**

**Error Cause: A model that contains custom operator is introduced by third-party library development. As a result, multiple domain versions may appear in the ONNX model. As a result, an error is reported during ATC model conversion.**

**The solution uses the surgeon component to traverse all nodes in the model and leaves the domain version of the node different from the domain version of the model blank. The following scripts are for reference only:**

```python
from auto_optimizer import OnnxGraph

g = OnnxGraph.parse("model.onnx")

for node in g.nodes:
    if node.domain:
        node.domain = ""

g.save("modified_model.onnx")
```
