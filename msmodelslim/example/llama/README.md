# LLAMA 量化案例

## 模型介绍

- [LLaMA（Large Language Model Meta AI）](https://github.com/facebookresearch/llama/tree/llama_v1)
  和 [LLaMA2（Large Language Model Meta AI 2）](https://github.com/facebookresearch/llama)，是由 Meta AI
  发布的一个开放且高效的大型基础语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

## 环境配置

1. 设置CANN包的环境变量

  ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

2. 下载安装开源版本msModelSlim

- git clone下载本仓代码
- 运行安装脚本
  ```shell
    cd msmodelslim
    bash install.sh
  ```

## 量化权重生成

### W8A8 PDMIX

- LLaMa3.1 70B推荐使用以下配置进行PDMIX量化
  ```shell
  python3 convert_llama3.1_70b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- LLaMa3.1 70B无回退权重生成，对话正常，建议仅用于基准性能测试
  ```shell
  python3 convert_llama3.1_70b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径} --no_disable
  ```
- LLaMa3.1 70B可用以下配置进行PDMIX量化，不只回退down层
  ```shell
  python3 convert_llama3.1_70b_pdmix_kqv.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```

### W8A8 KMEANS

- LLaMa3.1 8B推荐使用以下配置进行KMEANS混合量化
  ```shell
  python3 convert_llama3.1_8b_kmeans_mix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- LLaMa3.1 8B可用以下配置进行KMEANS量化
  ```shell
  python3 convert_llama3.1_8b_kmeans.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- LLaMa3.1 8B可用以下配置进行KMEANS+C8量化
  ```shell
  python3 convert_llama3.1_8b_kmeans_c8.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- LLaMa3.1 8B可用以下配置进行KMEANS+PERTILLING量化
  ```shell
  python3 convert_llama3.1_8b_pertilling.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | msit下载后放置的目录                  |
| modelslim_path | `${working_dir}/msmodelslim` |                         |

#### Llama3.1-70B w8a8c8 部分down层回退 BF16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/llama/convert_quant_weights_w8a8c8.py \
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

#### Llama3.1-70B w8a8c8 无回退 FP16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/llama/convert_quant_weights_norollback.py \
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

#### Llama3.1-70B w8a8 pertoken-pertensor 无回退 BF16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/llama/convert_quant_weights_pdmix.py \
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

#### Llama3.1-70B w8a8 pertoken-pertensor 仅回退down层 BF16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/llama/convert_llama3.1_70b_pdmix_revert_down_only.py \
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

#### Llama3.1-8B w8a8 parcomp 任意回退 BF16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/llama/convert_quant_weights_parcomp.py \
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

#### Llama3.1-8B w8a8 kmeans 任意回退 BF16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2机器上生成)
  - 环境准备
  ```
    pip install ckmeans_1d_dp
  ```
  
  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/llama/convert_quant_weights_lut.py \
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

