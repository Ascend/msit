## 大模型量化

大模型量化工具将高位浮点数转为低位的定点数，例如16bit降低到8bit，直接减少模型权重的体积，生成量化参数和权重文件。在无需训练成本的前提下，完成大模型的训练后压缩并最大程度保障其精度。

### 前提条件

- 仅支持在以下产品中使用。
    - Atlas 推理系列产品（Atlas 300I Duo 推理卡）。
    - Atlas 训练系列产品。
    - Atlas A2训练系列产品/Atlas 800I A2推理产品。

- 已参考环境准备，完成CANN开发环境的部署、以及PyTorch 2.1.0及以上版本的框架和npu插件、Python环境变量配置。
- 大模型量化工具须执行命令安装如下依赖。
  如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。
```
pip3 install numpy==1.25.2
pip3 install transformers        #需大于等于4.29.1版本，LLaMA模型需指定安装4.29.1版本
pip3 install accelerate==0.21.0  #若需要使用NPU多卡并行方式对模型进行量化，需大于等于0.28.0版本
pip3 install tqdm==4.66.1
```
- （可选）如果需要在大模型量化工具中使用NPU多卡并行的方式对模型进行量化，需关闭NPU设备中的虚拟内存，并手动配置量化将会执行的设备序列环境。
```
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False # 关闭NPU的虚拟内存
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 #配置量化将会执行的设备序列环境
```
说明
仅Atlas 训练系列产品和Atlas A2训练系列产品/Atlas 800I A2推理产品支持此功能。

### 已验证模型列表

[已验证模型列表](../../../docs/大模型已验证列表.xlsx)  

说明

大模型压缩技术主要针对常规大语言模型进行量化压缩，但在量化拥有特殊结构的模型时，msModelSlim工具可能存在以下限制：

MOE模型支持W8A8_per-token量化场景、W8A16 per-channel量化场景和W8A16 per-group量化场景，不支持lowbit稀疏量化场景。
多模态模型仅支持W8A16量化场景，不支持W8A8量化场景和lowbit算法稀疏量化场景。

### 功能实现流程
图1 量化接口调用流程

![量化接口调用流程](量化接口调用流程.png)

关键步骤说明如下：

1. 用户准备原始模型和校准数据。

2. 可选：使用离群值抑制功能对LLM模型进行离群值抑制，可参考精度保持策略选择是否启用。
    - 使用AntiOutlierConfig生成离群值抑制配置。
    - 调用AntiOutlier接口，将模型、校准数据等传入，生成抑制器。
    - 调用抑制器的process()方法对原始模型进行离群值抑制。

3. 使用QuantConfig生成量化配置。

4. 根据原始模型、量化配置和校准数据，调用Calibrator接口构建量化校准对象。

5. 调用生成的量化校准对象的run()方法对原始模型进行量化。

6. 调用生成的量化校准对象的save()接口保存量化后的模型，包括模型量化权重和模型相关参数，用于后续量化模型的部署任务，，具体请参见MindIE的“加速库支持模型列表”章节中已适配量化的模型。

### 量化步骤（以ChatGLM2-6B为例）

1. 用户自行准备模型、权重文件和校准数据，本样例以ChatGLM2-6B为例，目录示例如下：
```
├── config.json
├── configuration chatglm.py
├── modeling_chatglm.py
├── pytorch_model-00001-of-00007.bin
├── pytorch_model-00002-of-00007.bin
├── pytorch_model-00003-of-00007.bin
├── pytorch_model-00004-of-00007.bin
├── pytorch_model-00005-of-00007.bin
├── pytorch_model-00006-of-00007.bin
├── pytorch_model-00007-of-00007.bin
├── pytorch_model.bin.index.json
├── quantization.py
├── README.md
├── tokenization_chatglm.py
├── tokenizer.model
├── tokenizer_config.json
```

2. ChatGLM2-6B模型进行量化前请执行如下命令安装所需依赖，若运行量化工具过程中提示缺失某个依赖，请根据提示安装。
```
pip3 install protobuf==4.24.1
pip3 install sentencepiece==0.1.99
pip3 install sympy==1.11.1
```

3. 新建模型的quant.py量化脚本，编辑quant.py文件，根据实际的量化场景导入样例代码，参考加粗字体信息提示，并根据实际情况进行修改。

    - W8A8 per_channel量化场景导入的样例代码如下，kvcache、lowbit算法以及per_token算法量化场景导入的代码样例请参考[w8a8量化场景](量化及稀疏量化场景导入代码样例.md)。

```
# 导入相关依赖
import torch 
import torch_npu   # 若需要cpu上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', trust_remote_code=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2',
    trust_remote_code=True,
  ).npu()    # 若在npu上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto',创建model时需去掉.npu()；若在cpu上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)   
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])     
        return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=8, 
    w_bit=8,       
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'], 
    dev_id=model.device.index， 
    dev_type='npu',   # 在cpu进行量化时，需配置参数dev_type='cpu'，并取消dev_id=model.device.index参数的配置
    act_method=3,
    pr=0.5, 
    mm_tensor=False
  )  
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```

    - W8A16或W4A16 per_channel量化场景导入的样例代码如下，MinMax算法、HQQ算法、GPTQ算法、AWQ算法以及w4a16 per-group量化场景导入的代码样例请参考w8a16或w4a16量化场景。
```
# 导入相关依赖
import torch
import torch_npu   # 若需要cpu上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', trust_remote_code=True) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2',
    trust_remote_code=True,
    ).npu()    # 若在npu上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在cpu上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改，W8A16 Label-Free模式下请忽略此步骤
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=8,     # W4A16场景下，w_bit值需配置为4。在W4A16的per_group场景下，需参考W4A16的per_group量化场景参数进行设置。
    a_bit=16,         
    disable_names=[], 
    dev_id=model.device.index， 
    dev_type='npu',   # 在cpu进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=False, 
    mm_tensor=False
  )  
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```

4. 启动模型量化任务，并在指定的输出目录获取模型量化参数，量化后权重文件的介绍请参见量化后权重文件，若使用MindIE进行后续的推理部署任务，请保存为safetensors格式，具体请参见MindIE的“加速库支持模型列表”章节中已适配量化的模型。
```
python3 quant.py
```
量化任务完成后，可能会存在模型精度下降的情况，可以参考精度保持策略进行配置优化减少精度损耗。
### 量化及稀疏量化场景导入代码样例
其他场景样例可参考[此处](量化及稀疏量化场景导入代码样例.md)
### 量化后权重文件
- npy格式
当save_type设置为['numpy']或不设置时，量化权重会保存为npy文件，npy储存格式为字典，其中key值为各层Linear的名字，例如ChatGLM2-6B模型的transformer.encoder.layers.0.self_attention.query_key_value，value值为第0层query_key_value的Linear权重。
```
├── anti_fp_norm.npy   #LLaMA模型且已启用离群抑制功能，具体操作请参见使用离群值抑制功能，将会生成此文件。antioutlier算法生成浮点权重中的norm层权重文件，用于量化层的input和post norm的权重适配
├── deq_scale.npy      #W8A8量化和稀疏量化的量化参数权重文件，Tensor数据类型为int64，deq_scale已针对量化算子进行数据类型转换，可直接适配算子。在量化BF16模型情况下，数据类型不会转换为int64，仍然为float32
├── input_offset.npy   #W8A8量化和稀疏量化的激活值量化偏移值权重文件，Tensor数据类型为float32
├── input_scale.npy    #W8A8量化和稀疏量化的激活值量化缩放因子权重文件，Tensor数据类型为float32
├── kv_cache_offset.npy    #kv cache量化参数文件，kv linear激活值量化偏移值权重文件，Tensor数据类型为float32
├── kv_cache_scale.npy   #kv cache量化参数文件，kv linear激活值量化缩放因子权重文件，Tensor数据类型为float32
├── quant_bias.npy     #W8A8量化和稀疏量化的量化参数权重文件，Tensor数据类型为int32，quant_bias已考虑原始浮点模型linear层的bias值
├── quant_weight.npy   #量化权重文件，Tensor数据类型为int8
├── weight_offset.npy  #w8a16和w4a16权重量化参数文件，Tensor数据类型为float32
├── weight_scale.npy   #w8a16和w4a16权重量化参数文件，Tensor数据类型为float32
```
推理部署时读取上述文件的示例代码：quant_param_dict = np.load("xxx.npy", allow_pickle=True).item()。

- safetensors格式
当save_type设置为['safe_tensor']时，量化权重会保存为safetensors文件和json描述文件，

    - safetensors中储存格式为字典，包含量化权重和量化不修改的浮点权重。其中量化权重的key值为各层Linear的名字加上对应权重的名字，module.weight和module.bias对应anti_fp_norm.npy，weight对应quant_weight.npy，quant_bias对应quant_bias.npy等以此类推。例如ChatGLM2-6B模型的transformer.encoder.layers.0.self_attention.query_key_value.deq_scale对应npy格式权重中deq_scale.npy中的transformer.encoder.layers.0.self_attention.query_key_value。
```
# llama模型稀疏量化生成的权重文件部分内容
{
  "model.embed_tokens.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_scale": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_offset": tensor([...]),
  "model.layers.0.self_attn.q_proj.quant_bias": tensor([...]),
  "model.layers.0.self_attn.q_proj.deq_scale": tensor([...]),
  "model.layers.0.self_attn.k_proj.weight": tensor([...]),
 ...
}
```
    - json描述文件中储存的量化权重的总体类型model_quant_type，是否启用kvcache量化kv_cache_type，和其中各个权重的类型，来自原始浮点权重则为FLOAT，来自W8A8量化则为W8A8，来自稀疏量化则为W8A8S，来自压缩则为W8A8SC。
```
# llama模型稀疏量化生成的json描述文件部分内容
{
  "model_quant_type": "W8A8S",                               # 整体量化类型为稀疏量化
  "model.embed_tokens.weight": "FLOAT",                      # 来自原始浮点模型的embed_tokens权重
  "model.layers.0.self_attn.q_proj.weight": "W8A8S",         # 量化新增的第0层self_attn.q_proj的quant_weight
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8S",    # 量化新增的第0层self_attn.q_proj的input_scale
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8S",   # 量化新增的第0层self_attn.q_proj的input_offset
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8S",     # 量化新增的第0层self_attn.q_proj的quant_bias
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8S",      # 量化新增的第0层self_attn.q_proj的deq_scale
  "model.layers.0.self_attn.k_proj.weight": "W8A8S",         # 量化新增的第0层self_attn.k_proj的quant_weight
 ...
}
```

### 精度保持策略

在量化权重生成后，可以使用伪量化模型进行推理，检验伪量化精度是否正常。伪量化是指通过torch，通过浮点运算完成量化模型运算逻辑，运算过程中的数据和真实量化的数据差异只在算子精度上，同时可以规避接入推理框架时引入的精度误差。如果伪量化精度不满足预期，真实量化结果也将无法满足预期。在调用Calibrator.run()方法后，构建Calibrator时传入的model会被替换为伪量化模型，可以直接调用进行前向推理，用来测试对话效果。如果伪量化结果不理想，可以参考以下手段进行调优。一般来说，W8A16的精度调优较为容易，W8A8和稀疏量化的精度调优相对复杂。

#### 数据集精度掉点严重，对话乱码或胡言乱语

1. 对于label free校准场景，确认浮点模型使用torch npu推理是否正常。量化校准依赖浮点模型推理，如果浮点推理异常，量化校准时获取到的数据分布信息不对，校准结果自然不对。

2. 适当增加回退层。某些模型中的部分Linear层对精度的影响比较显著。例如ChatGlm2-6B模型W8A8量化时的layers.0.mlp.dense_4h_to_h层，依据调优经验以及相关论文数据，模型靠前和靠后的decoder layer、各个decoder layer的mlp down层对精度的影响一般较大，可以优先考虑回退这些层。如果回退效果不理想的话，可以尝试较为激进的回退策略，例如回退掉所1/4或者1/2的Linear层，直到完全回退成浮点模型，模型的精度也完全回退成浮点模型的精度。退回越多，精度越高，性能越差。

#### 数据集精度部分掉点，对话正常

1. 调整量化参数。例如W8A8量化调整act_method，W8A16量化更换使用的w_method。act_method默认为1。该参数可以选 ‘1’ ‘2’ ‘3’：1代表min-max量化方式；2代表histogram量化方式；3代表min-max和histogram自动混合量化方式。LLM大模型场景下推荐使用3。（稀疏量化的情况下只支持1和2的方式）

2. 稀疏量化可以调整fraction，该参数的含义为限制异常值的保护范围，建议在0.01~0.1之间将相应的值调大来增加精度。Lowbit场景下，除了上述参数微调，还可以使用sigma调整sigma_factor，该参数的含义也为限制异常值的保护范围，建议在3.0~4.0之间将相应的值调小来增加精度。

3. 使用异常值抑制算法，将do_smooth设置为True。W8A8量化使用anti_method=m1或m2，W8A16量化使用m3，通过抑制量化过程中的异常值，从而提高量化模型的精度。Lowbit场景下只需要开启即可，无需设置方法类型。

4. 调整校准数据。校准数据的数量一般为20-40条。在选取时需要考虑模型部署时的具体推理场景，例如中文模型需要使用中文输入作为校准集；英文模型需要使用英文输入；代码生成类模型则使用代码生成类任务；中英文兼顾的模型考虑使用中英文混合的校准集合。正常情况下，可以增加数据得到精度提升，但是到一定数据后，提高数据对精度影响有限。有些场景下，减少数据反而能得到精度提升。（例如长数据场景）

5. 增加回退层，可以使用disable_level自动回退功能按照一定的标准自动回退对精度影响比较大的Linear层，或者按照一定的经验，通过disable_name手动设置回退层。