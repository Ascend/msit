# MindIE Torch Scenario - Network-wide Operator Data Dumping #

## Introduction ##

Tensor data dump can be performed on models optimized by MindIE Torch compilation.

## 1. Dependency ##

 *  CANN (8.0RC3 or later)
 *  MindIE (1.0RC3 or later)

## 2. Install the msit. ##

 *  For details about how to install the msit, see.[Integrated Installation Guide](../../../../../docs/install/README.md)	
 *  Ensure that the compare and llm components under msit are installed.
    
    ```
    msit install compare llm
    ```

## 3. Dump data ##

Before precision comparison, dump CPU and NPU data. The following uses the ResNet50 model as an example. Detailed scripts are provided for your reference.

### 3.1 NPU Data Dump (TorchScript Route) ###

 *  To prepare`resnet_inference.py`, is the forward inference script of the model. Currently, MindIE Torch supports data dump of TS and export routes.
 *  The MindIE Torch inference interface is an asynchronous interface. To ensure that the inference is complete and data is flushed to disks, the synchronization operation is required. Currently, MindIE Torch and torch_npu provide two different APIs. The synchronization method is as follows:
    
    ```
    #The torch_npu interface is recommended.
    import torch_npu
    npu_result = compiled_module(input_data.to("npu:0"))
    torch_npu.npu.synchronize()
    mindietorch.finalize()
    #If the torch_npu dependency package is not installed in the environment, the MindIE Torch API can be used as follows:
    npu_stream = mindietorch.npu.Steam()
    npu_result = compiled_module(input_data.to("npu:0"))
    with mindietorch.npu.stream(npu_stream):
      npu_stream.synchronize()
    ```

#### 3.1.1 TorchScript Route (jit.ScriptModule) ####

```
#Import the torch first and then the mindietorch.
import torch
import mindietorch
import torchvision.models as models

model = models.resnet50()
model.eval()
torch.manual_seed(88)   #Set random seeds to ensure that the model input on the CPU side is consistent with that on the NPU side.
input_data = torch.randn(1, 3, 224, 224)

input_info = [mindietorch.Input((1, 3, 224, 224))]
mindietorch.set_device(0)  #Designated Inference Card

traced_model = torch.jit.trace(model, input_data)
compiled_module = mindietorch.compile(traced_model, inputs=input_info, soc_version={具体的芯片型号}) #You can run the npu-smi info command to view the chip model.
#Data synchronization
import torch_npu
npu_result = compiled_module(input_data.to("npu:0"))
torch_npu.npu.synchronize()

mindietorch.finalize()
```

#### 3.1.2 Torch.export Route (export.ExportedProgram or nn.Module) ####

```
import torch
from torch._export import export
import mindietorch
import torchvision.models as models

model = models.resnet50()
model.eval()
input_data = torch.randn(1, 3, 224, 224)

input_info = [mindietorch.Input((1, 3, 224, 224))]
model_ep = export(model, args=(input_data,))  #If export is not used, the nn.Module model is used.
compiled_module = mindietorch.compile(model_ep, inputs=input_info, soc_version={具体的芯片型号}, ir="dynamo")    #The ir must be set to dynamo.

mindietorch.set_device(0)
import torch_npu
result = compiled_module(input_data.to("npu:0"))
torch_npu.npu.synchronize()
mindietorch.finalize()
```

#### 3.1.3 Torch.export Route (fx.GraphModule) ####

```
import torch
from torch._export import export
from torch import fx
import mindietorch
import torchvision.models as models

def fx_transform(m: torch.nn.Module, tracer_class=fx.Tracer):
    graph = tracer_class().trace(m)
    graph.lint()
    new_gm = fx.GraphModule(m, graph)
    return new_gm

input_data = torch.randn(1, 3, 224, 224)
model = models.resnet50()
model.eval()
fx_model = fx_transform(model)

inputs = [mindietorch.Input((1, 3, 224, 224))] 
compiled_module = mindietorch.compile(fx_model, inputs=inputs, soc_version={具体的芯片型号}, ir="dynamo") #The ir must be set to dynamo.

mindietorch.set_device(0)
import torch_npu
npu_result = compiled_module(input_data.to("npu:0"))
torch_npu.npu.synchronize()
mindietorch.finalize()
```

 *  After editing the resnet_inference.py file based on the actual scenario, run the msit command to dump the NPU data.
    
    ```
    msit debug dump --exec "python resnet_inference.py" [--output /path/to/dump] [--operation-name MatMulv2_1,trans_Cast_0]
    ```
 *  Parameter Description

| Parameter name            | Description                                                                                                                                                                                                                                    | Mandatory. |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| --exec                    | Execution Command of the MindIE Torch Inference Script                                                                                                                                                                                         | Yes        |
| --output                  | Specifies the dump data output path. The default path is the current path.                                                                                                                                                                     | No.        |
| -opname, --operation-name | Indicates the operator that needs to be dumped. The default value is all, indicating that all ops in the model will be dumped. The element is the MindIE Torch operator type. If operation-name is set, only the specified ops will be dumped. | No.        |

### 3.2 Benchmark Data Dump ###

 *  Generally, the benchmark data used for precision comparison is the inference data in Torch single-operator mode. For details about the dump process, see.[Precision Data Collection in the PyTorch Scenario](../../../../../docs/llm/工具-Pytorch场景数据dump.md)	

## Attention. ##

 *  When the NPU data is dumped, inference must be performed online. If inference is performed after the model is saved and then loaded, the JSON file lacks necessary information and cannot be compared.
 *  After the NPU data is dumped, two operator type mapping files are generated in the current directory. The file names are as follows:`mindie_torch_op_mapping.json`And to the`mindie_rt_op_mapping.json`.
 *  The torch_npu is imported to the transformers library. Therefore, if the transformers library is imported and torch_npu is installed in the environment, the torch_npu synchronization interface must be used to flush data to disks during inference. Otherwise, an error is reported.

