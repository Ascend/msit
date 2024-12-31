# Qwen 量化案例

## 模型介绍

- 千问（qwen）语言大模型是阿里巴巴集团推出的大型语言模型，具备强大的自然语言处理能力，能够理解和生成文本，应用于智能客服、内容生成、问答系统等多个场景，助力企业智能化升级。

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

- Qwen2.5 7B推荐使用以下配置进行PerTensor量化
  ```shell
  python3 convert_qwen2.5_7b_pertensor.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径} --mix_select_layer --mix_layer_alpha 0.5 --mix_layer_beta 1
  ```
- Qwen2.5 72B推荐使用以下配置进行PDMIX量化
  ```shell
  python3 convert_qwen2.5_72b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- Qwen2.5 72B 无回退权重生成，对话正常，建议仅用于基准性能测试
  ```shell
  python3 convert_qwen2.5_72b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径} --no_disable
  ```
- Qwen2.5 14B推荐使用以下配置进行PDMIX量化
  ```shell
  python3 convert_qwen2.5_14b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- Qwen2.5 14B 无回退权重生成，对话正常，建议仅用于基准性能测试
  ```shell
  python3 convert_qwen2.5_14b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径} --no_disable
  ```
  
- Qwen2.5 72B可用以下配置进行PDMIX量化，不只回退down层
  ```shell
  python3 convert_qwen2.5_72b_pdmix_kqv.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```

### W8A8 PERTILLING

- Qwen2 7B推荐使用以下配置进行PERTILLING+KMEANS量化
  ```shell
  python3 convert_qwen2_7b_pertilling.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
  
- Qwen2 7B推荐使用以下配置进行PERTILLING+KMEANS+C8量化
  ```shell
  python3 convert_qwen2_7b_pertilling_c8.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```

### W8A8 PERTENSOR

- Qwen2.5 7B推荐使用以下配置进行PERTENSOR+KMEANS量化
  ```shell
  python convert_qwen2.5_7b_pertensor.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径} --mix_select_layer --mix_layer_alpha {alpha} --mix_layer_beta {beta}
  ```
- Qwen2.5 7B可用以下配置进行KMEANS+C8量化
  ```shell
  python convert_qwen2.5_7b_kmeans_c8.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径} --mix_select_layer
  ```
- Qwen2.5 72B可用以下配置进行PERTENSOR+C8量化
  ```shell
  python convert_qwen2.5_72b_pertensor.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | msit下载后放置的目录                  |
| modelslim_path | `${working_dir}/msmodelslim` |                         |



#### Qwen2-7B W8A8 parcomp 任意回退 BF16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/qwen/convert_quant_weights_parcomp.py \
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

#### Qwen2-7B W8A8 kmeans 任意回退 BF16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2机器上生成)
  - 环境准备
  ```
    pip install ckmeans_1d_dp
  ```

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/qwen/convert_quant_weights_lut.py \
    --model_path {浮点权重路径} \
    --save_path {W8A8s量化权重路径}
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
    "quantize": "w8a8s"
    ```

#### Qwen2-72B W8A8 pertoken-pertensor 无回退 BF16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)
  - 假设当前位于`${working_dir}`目录下
  - 使用下列命令进行W8A8 pertoken-pertensor量化权重导出：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  cd ${modelslim_path}
  python example/qwen/convert_quant_weights_pdmix.py --model_path ${浮点权重路径} --save_path ${量化权重保存路径}

  ```

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
  
#### Qwen2-72B w8a8c8 部分down层回退 BF16模型量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/qwen/convert_quant_weights_w8a8c8.py \
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
  #### Qwen2-72B w8a8c8 无回退 FP16模型的量化权重请使用以下指令生成 (此量化权重只能在800IA2 64G机器上生成)

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${modelslim_path}
    python example/qwen/convert_quant_weights_norollback.py \
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