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
以下量化样例脚本均基于bf16浮点模型

### W8A8 PDMIX

- LLaMa3.1 70B推荐使用以下配置进行PDMIX量化，只回退down层
  ```shell
  python3 convert_llama3.1_70b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- LLaMa3.1 70B无回退权重生成，对话正常，建议仅用于基准性能测试
  ```shell
  python3 convert_llama3.1_70b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径} --no_disable
  ```
- LLaMa3.1 70B可用以下配置进行PDMIX量化，回退涉及k,q,v,o,down线性层
  ```shell
  python3 convert_llama3.1_70b_pdmix_kqv.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```

### W8A8 KMEANS

- LLaMa3.1 8B推荐使用以下配置进行KMEANS+PERTENSOR/W8A8/FLOAT混合量化
  ```shell
  python3 convert_llama3.1_8b_kmeans_mix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- LLaMa3.1 8B可用以下配置进行KMEANS+PERTENSOR量化
  ```shell
  python3 convert_llama3.1_8b_kmeans.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- LLaMa3.1 8B可用以下配置进行KMEANS+PERTENSOR+C8量化
  ```shell
  python3 convert_llama3.1_8b_kmeans_c8.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- LLaMa3.1 8B可用以下配置进行KMEANS+PERTILLING量化
  ```shell
  python3 convert_llama3.1_8b_pertilling.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```