# PyTorch 场景的精度数据采集

## 简介

msit llm dump 工具主要通过在推理脚本内添加 dump 接口启动推理的方式采集精度数据。

## 使用前准备

**工具安装**

需要安装msit工具，软件安装见 [msit工具安装](../install/README.md)。

安装好msit后需要安装msit中的 llm 组件，执行 msit install llm。

**版本要求**

若使用transformers 的 AutoModelForCausalLM 类加载预训练模型，请确保 transformers>=4.43.2（或官方给出的依赖版本）。

**磁盘空间要求**

落盘数据时要预留足够的磁盘空间大小，以下为给出的不同大模型所需空间大小示例。

|    模型    | 落盘数据大小(单个 token 整网数据) |
| ---------- | ---------- | 
| Llama3-8B  |    20M   | 
| qwen1.5-14B |   660MB |

## 接口介绍

### DumpConfig

 **功能说明**：配置Dump参数实现自定义模型数据采集。

**原型**：

```Python
DumpConfig(dump_path=None, token_range=None, seed=None)
```

参数说明

|参数名|可选/必选|使用说明|含义|版本|
|-----------|----------------------| ------------------------------------------------------------|--------|--|
|dump_path|可选|数据类型：str，默认为当前目录。|设置dump的数据路径。| 
|token_range|可选|数据类型：list。默认为[0]，只dump第0个token的数据。|需要dump的token列表。|
|module_list|可选|数据类型：list，默认为[]，即dump所有module的数据。|指定要hook的module类型。|
|analyze|可选|数据类型：bool，默认为false，即不生成分析文件。|分析dump的数据，生成一个csv文件用于记录数据类型等。|
|api_list|可选|数据类型：list，默认为[]，即dump所有api的数据。|指定要hook的api类型。|
|tensor_part|可选|数据类型：int，默认为2。当tensor_part=0时，只dump输入数据；当tensor_part=1时，只dump输出数据；当tensor_part=2时，dump输入和输出的数据。|指定要dump哪部分数据。|
|device_id|可选|数据类型：int，默认为None 表示不限制 device。如指定 device_id=1，将跳过其他 device 的 dump。|指定要dump的device_id 。|
|dump_last_logits|可选|数据类型： bool，默认为False，当开启后，仅 Dump 模型最后输出的 logits，模型中间layer 不会再输出。可参考 [《logits精度比对》](加速库场景-输出Token的logits精度比对.md) |是否需要Dump 模型最后的输出logits|
|mode|可选|可以选择dump api 还是 module，默认是module，也可以传入数组['api', 'module']，表示两种都dump。|设置dump的模式。|7.0.0b530|
|dump_weight|可选|数据类型：bool，默认是False，不dump。|设置是否需要dump权重。|7.0.0b530|
|layer_name|可选|数据类型：str，可以通过该参数控制dump 的权重和tensor。支持 * 表示匹配0或多个随意字符，不支持其他的模式匹配。|指定需要dump的layer名字。|7.0.0b530|
|seed|可选|数据类型：int，可以通过该参数确定是否要启动确定性计算，输入的值表示固定随机性的种子值。|设定启动确定性计算的种子。|7.0.0rc730|
|dump_statistics_mode|可选|数据类型： int，默认为0，不进行统计量dump。当dump_statistics_mode=1时，只dump统计量；当dump_statistics_mode=2时，同时dump统计量和tensor。|设置统计量Dump模式。|7.0.0rc1230|

### register_hook

**原型**：

```Python
register_hook(model, config, hook_type="dump_data")
```

| 参数名    | 含义           | 使用说明                                                | 是否必填 |
| --------- | -------------- | ------------------------------------------------------- | -------- |
| model     | 需要hook的模型 | 数据类型：torch.nn.Module，建议设置为最外层的torch模型  | 是       |
| config    | Hook配置       | 数据类型：DumpConfig                                    | 是       |
| hook_type | hook类型       | 数据类型：str，默认值为dump_data，当前仅支持dump_data。 | 否       |

## 示例代码

### 快速上手

这个示例调用了transformers库的Llama-2-7b模型，在进行数据采集时使用原型函数 DumpConfig 传入 dump_path 参数、 token_range 参数和 seed 参数。是需要在模型推理前配置好工具数据采集接口并开启数据dump即可，实际使用场景可根据自己的模型进行调整。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM

"""
下面两行是需要添加的
"""
# 导入工具的数据采集接口
from msit_llm import DumpConfig, register_hook

# 在模型推理开始前实例化DumpConfig
config = DumpConfig(dump_path='./torch_dump/', token_range=[0,1,2,3], seed=2345)

# 需自行保证模型配置代码文件安全可靠。在确保其安全性的前提下，可以使用以下代码。否则，请将'trust_remote_code'置为False
tokenizer = AutoTokenizer.from_pretrained("/home/data/Llama-2-7b-hf", trust_remote_code=True)

# 需自行保证模型配置代码文件安全可靠。在确保其安全性的前提下，可以使用以下代码。否则，请将'trust_remote_code'置为False
model = LlamaForCausalLM.from_pretrained("/home/data/Llama-2-7b-hf", trust_remote_code=True).to('npu')

if __name__ == "__main__":
    # 在推理前开启数据 dump
    register_hook(model, config)

    # 下面可以替换为自己的推理代码
    with torch.no_grad():
        inputs = tokenizer(
                    "What's deep learning?",
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=10).to('npu')
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=10)

```

## dump 结果文件介绍

推理结束后，工具将 dump 的数据保存在 dump_path 参数指定的目录下。目录结构示例如下：

```lua
├── dump_path
│   ├── msit_dump_20250220_083154 # 命名格式为 msit_dump_{PID}
│   |   ├── torch_tensors 
│   |   │   ├── npu5_7624  # 命名格式为 {device_id}_{进程号}
|   |   |   |   ├── 3  # 命名格式为 {token_id}
|   |   |   |   |   ├── root    
|   |   |   |   |   ├── root.lm_head
|   |   |   |   |   ├── root.model.embde_tokens         # 命名格式为{root}.{model}.{module_name}
|   |   |   |   |   ├── root.model.layers0   
|   |   |   |   |   ├── root.model.layers0.mlp          # 命名格式为{root}.{model}.{layers_id}.{module_name}
|   |   |   |   |   ├── root.model.layers0.mlp.act_fn   
|   |   |   |   |   ...
|   |   |   |   └── model_tree.json    


* `npu`：设备卡号，每张卡的数据保存在对应的 `npu{ID}` 目录下。
* `{token id}`：每一个 token 采集到的数据会保存在对应的 token id 下。
