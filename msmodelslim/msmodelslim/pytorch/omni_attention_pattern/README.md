# Omni attention pattern 搜寻工具


本工具用于搜索attention pattern用于后续的omni attention的插入，可以定制化地适配不同的sparsity(0~1)要求：sparsity为0.0的时候是模型原型不作任何改变，sparsity为1.0的时候是Omni attention的极致性能，但精度可能会随着sparsity的增加而下降。
需要注意的是pattern的搜索很重要，一个好的pattern不仅能有效地降低推理成本，而且推理的精度也能得到保障，本方案实现的时候已为（Qwen2.5-7B、Qwen2.5-72B、Llama3-8B、Llama3-70B）搜索到合适的pattern，有需要的请联系对应人员。




表1 已验证模型列表
|模型名称|框架|
|----|-----|
|Qwen2.5-7B|PyTorch|
|Llama3-8B|PyTorch|
|Qwen2.5-72B|PyTorch|
|Llama3-70B|PyTorch|



### 前提条件
已参考环境准备，完成CANN开发环境的部署、PyTorch 2.1.0及以上版本的安装及Python环境变量的配置。
执行命令安装如下依赖。
```
pip3 install numpy
pip3 install transformers>=4.45.2
pip3 install torch>=2.1.0
pip3 install torch_npu>=2.1.0
```

### 搜索步骤（以Qwen2.5-7B为例）

#### 步骤1：用户准备原始模型。

用户需要自行准备模型、权重文件。本样例以Qwen2.5-7B-Instruct为例，从HuggingFace下载权重文件，放置在目录`/path/to/qwen/7b`。目录示例如下：
```
.gitattributes
README.md
chat_template.json
config.json
generation_config.json
merges.txt
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
model.safetensors.index.json
tokenizer_config.json
tokenizer.json
tokenizer_config.json
vocab.json
```

#### 步骤2：在omni_attention_pattern文件夹下创建omni_run.py，并传入模型权重路径。

```
from msmodelslim.pytorch.omni_attention_pattern.omni_config import OmniAttentionConfig
from msmodelslim.pytorch.omni_attention_pattern.omni_tools import OmniAttentionGeneticSearcher

config = OmniAttentionConfig(model_path="/path/to/qwen/7b", pool_size=50)
print(config)

searcher = OmniAttentionGeneticSearcher(config)
searcher.search_on_this_sparsity(sparsity=50)
```

参数`pool_size`控制遗传算法初始化个体的数量。参数`sparsity`控制得到的pattern的稀疏度，`sparsity`越大，则压缩力度越大，pattern中的压缩头的数量越多，也就是说，推理时的性能越快，对精度的影响也可能会更大。

#### 步骤3：检查输出
搜索出的最佳pattern会保存在`omni_attention_pattern/output`文件夹下，每种模型有一个自己的子文件夹。例如:

```
- omni_attention_pattern/
-- output/
--- Qwen2.5-7B-Instruct/
----- genetic_rowwise_sparsity_10.tsv
----- genetic_rowwise_sparsity_20.tsv
```

#### 步骤4：使用pattern
在MindIE中，可以通过使用环境变量来开启OMNI Attention，例如：
```
export ATB_LLM_OMNI_ATTENTION_ENABLE=1
export ATB_LLM_OMNI_SHIFT_WINDOWS_ENABLE=1
export ATB_LLM_OMNI_ATTENTION_PATTERN_FILE=/path/to/pattern/tsv/file
```
只要如上所示指定pattern对应的tsv文件，MindIE会读取该pattern并用于推理加速。