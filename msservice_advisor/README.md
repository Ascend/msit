# MindStudio Service Profiling Advisor

## 介绍
- **基本功能** 根据当前的 benchmark 输出结果以及 service 的 config.json 配置，分析提高 TFTT / Throughput 等的优化点
- **安装**
  ```sh
  pip install msservice-advisor
  ```
- **执行 mindie benchmark，输出结果的 instance 文件夹**
- **执行调参建议**
  ```sh
  msservice_advisor -i ../instance/
  # msservice_advisor_logger - INFO - <think>
  # msservice_advisor_logger - INFO - num_mem_size_checker
  # msservice_advisor_logger - INFO - check_input_tokens
  # ...
  # msservice_advisor_logger - INFO - find_best_batch_size
  # msservice_advisor_logger - INFO - ==decode==
  # msservice_advisor_logger - INFO - ==prefill==
  # msservice_advisor_logger - INFO - </think>
  # msservice_advisor_logger - INFO -
  # msservice_advisor_logger - INFO - <answer>
  # msservice_advisor_logger - INFO - [config] maxBatchSize
  # msservice_advisor_logger - INFO - [action] set to range [xx, xx], average is xx
  # msservice_advisor_logger - INFO - [reason] 经过对当前显存信息的计算，建议将maxBatchSize的值设置为average大小，并逐渐向范围最大值调整，以占满整个显存
  # msservice_advisor_logger - INFO -
  # msservice_advisor_logger - INFO - [config] maxBatchSize
  # msservice_advisor_logger - INFO - [action] set bigger
  # msservice_advisor_logger - INFO - [reason] 目前batch样本太小，建议调大点试试
  # msservice_advisor_logger - INFO -
  # msservice_advisor_logger - INFO - [config] maxPrefillBatchSize
  # msservice_advisor_logger - INFO - [action] set bigger
  # msservice_advisor_logger - INFO - [reason] 目前batch样本太小，建议调大点试试
  # msservice_advisor_logger - INFO -
  # msservice_advisor_logger - INFO - </answer>
  ```
- **不安装，使用源码方式执行调参建议**
  ```sh
  git clone https://gitee.com/ascend/msit.git
  cd msit/msservice_advisor
  export PYTHONPATH=$PWD:$PYTHONPATH
  python msservice_advisor/__main__.py -i ../instance/
  ```
- 参数

  | 参数                 | 说明                                                            |
  | -------------------- | --------------------------------------------------------------- |
  | -i, --instance_path  | benchamrk 输出的 instance 路径                                  |
  | -s, --service_config_path  | mindie service config.json path, 默认值为环境变量中的 `MIES_INSTALL_PATH` |
  | -t, --target         | 调参指标, 可选值：ttft, firsttokentime, throughput              |
  | -m, --target_metrics | 调参指标的具体项，可选值：average,max,min,P75,P90,SLO_P90,P99,N |
  | --show  | 指定是否显示拟合曲线的 matplotlib 绘图                                                |
  | -l, --log_level  | 日志级别，可选值 debug, info, warning, error                         |
  | -in, --input_token_num  | 可选参数，请求输入长度。不输入则默认从benchmark的instance结果中获取 |
  | -out, --output_token_num  | 可选参数，请求平均输出长度。不输入则默认从benchmark的instance结果中获取 |
  | -tp, --tp  | 可选参数，tp域大小。不输入则默认从mindie service config.json文件中获取 |