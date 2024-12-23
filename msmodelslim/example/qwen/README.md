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

- Qwen2.5 72B推荐使用以下配置进行PDMIX量化
  ```shell
  python3 convert_qwen2.5_72b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径}
  ```
- Qwen2.5 72B 无回退权重生成，对话正常，建议仅用于基准性能测试
  ```shell
  python3 convert_qwen2.5_72b_pdmix.py --model_path {浮点权重路径} --save_path {W8A8PDMIX权重路径} --no_disable
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