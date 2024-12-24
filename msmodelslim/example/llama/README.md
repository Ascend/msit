# LLAMA 量化案例

## 模型介绍

- [LLaMA（Large Language Model Meta AI）](https://github.com/facebookresearch/llama/tree/llama_v1)和 [LLaMA2（Large Language Model Meta AI 2）](https://github.com/facebookresearch/llama)，是由 Meta AI 发布的一个开放且高效的大型基础语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | msit下载后放置的目录                  |
| modelslim_path | `${working_dir}/msmodelslim` |                         |

#### Llama3.1-70B w8a8c8 量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/Llama/convert_quant_weights_w8a8c8.py \
    --model_path {浮点权重路径} \
    --save_path {W8A8C8量化权重路径}
    ```
    - 注意：`model_path`和`save_path`请勿使用同一个文件夹，避免浮点权重和量化权重混淆

- 从浮点权重路径下复制以下文件到量化权重路径
    - config.json
    - gitattributes
    - special_tokens_map.json
    - tokenizer.json
    - tokenizer.model
    - tokenizer_config.json


- 修改量化权重的 config.json 文件 加入quantize
    ```
    "quantize": "w8a8",
    "quantization_config":{
      "kv_quant_type":"C8"
    }
    ```

#### Llama3.1-70B w8a8c8 无回退量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/Llama/convert_quant_weights_norollback.py \
    --model_path {浮点权重路径} \
    --save_path {W8A8C8量化权重路径}
    ```
    - 注意：`model_path`和`save_path`请勿使用同一个文件夹，避免浮点权重和量化权重混淆

- 从浮点权重路径下复制以下文件到量化权重路径
    - config.json
    - gitattributes
    - special_tokens_map.json
    - tokenizer.json
    - tokenizer.model
    - tokenizer_config.json


- 修改量化权重的 config.json 文件 加入quantize
    ```
    "quantize": "w8a8",
    "quantization_config":{
      "kv_quant_type":"C8"
    }
    ```

#### Llama3.1-70B w8a8 pertoken-pertensor 量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/Llama/convert_quant_weights_pdmix.py \
    --model_path {浮点权重路径} \
    --save_path {W8A8_PDMIX量化权重路径}
    ```
    - 注意：`model_path`和`save_path`请勿使用同一个文件夹，避免浮点权重和量化权重混淆

- 从浮点权重路径下复制以下文件到量化权重路径
    - config.json
    - gitattributes
    - special_tokens_map.json
    - tokenizer.json
    - tokenizer.model
    - tokenizer_config.json


- 修改量化权重的 config.json 文件 加入quantize
    ```
    "quantize": "w8a8_pdmix"
    ```

#### Llama3.1-8B w8a8 parcomp 量化权重请使用以下指令生成 (此量化权重只能在800IA2机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/Llama/convert_quant_weights_parcomp.py \
    --model_path {浮点权重路径} \
    --save_path {W8A8量化权重路径}
    ```
    - 注意：`model_path`和`save_path`请勿使用同一个文件夹，避免浮点权重和量化权重混淆

- 从浮点权重路径下复制以下文件到量化权重路径
    - config.json
    - gitattributes
    - special_tokens_map.json
    - tokenizer.json
    - tokenizer.model
    - tokenizer_config.json


- 修改量化权重的 config.json 文件 加入quantize
    ```
    "quantize": "w8a8"
    ```

#### Llama3.1-8B w8a8 kmeans量化权重请使用以下指令生成 (此量化权重只能在800IA2机器上生成)
  - 环境准备
  ```
    pip install ckmeans_1d_dp
  ```
  
  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/Llama/convert_quant_weights_lut.py \
    --model_path {浮点权重路径} \
    --save_path {W8A8S量化权重路径}
    ```
    - 注意：`model_path`和`save_path`请勿使用同一个文件夹，避免浮点权重和量化权重混淆

- 从浮点权重路径下复制一下文件到量化权重路径
    - config.json
    - gitattributes
    - special_tokens_map.json
    - tokenizer.json
    - tokenizer.model
    - tokenizer_config.json


- 修改量化权重的 config.json 文件 加入quantize
    ```
    "quantize": "w8a8s"
    ```
  
#### Llama3.1-70B w8a8 pertoken-pertensor 仅回退down层量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/Llama/convert_llama3.1_70b_pdmix_revert_down_only.py \
    --model_path {浮点权重路径} \
    --save_path {W8A8_PDMIX量化权重路径}
    ```
    - 注意：`model_path`和`save_path`请勿使用同一个文件夹，避免浮点权重和量化权重混淆

- 从浮点权重路径下复制以下文件到量化权重路径
    - config.json
    - gitattributes
    - special_tokens_map.json
    - tokenizer.json
    - tokenizer.model
    - tokenizer_config.json


- 修改量化权重的 config.json 文件 加入quantize
    ```
    "quantize": "w8a8_pdmix"
    ```
