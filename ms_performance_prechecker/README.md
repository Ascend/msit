# MindStudio Performance Prechecker

## 介绍
- **基本功能** 检查当前环境中 log level、cpu 绑核、内核版本等，是否达到最佳要求，并给出相应建议
- **环境要求**
  - Python >= 3.8
- **安装**
  ```sh
  pip install ms-performance-prechecker
  ```
- **执行检查**
  ```sh
  ms_performance_prechecker
  # ms_performance_prechecker_logger - INFO - <think>
  # ms_performance_prechecker_logger - INFO - simple_env_checker
  # ms_performance_prechecker_logger - INFO - linux_kernel_release_checker
  # ms_performance_prechecker_logger - INFO - Got kernel_release: 5.15.167.4-microsoft-standard-WSL2, suggested is 5.10
  # ...
  # ms_performance_prechecker_logger - INFO - </think>
  # ms_performance_prechecker_logger - INFO -
  # ms_performance_prechecker_logger - INFO - <answer>
  # ms_performance_prechecker_logger - INFO - [env] TASK_QUEUE_ENABLE
  # ms_performance_prechecker_logger - INFO - [action] export TASK_QUEUE_ENABLE=2
  # ms_performance_prechecker_logger - INFO - [reason] 配置task_queue 算子下发队列优化登记，可能导致运行中NPU内存峰值上升
  # ms_performance_prechecker_logger - INFO -
  # ms_performance_prechecker_logger - INFO - [env] HCCL_OP_EXPANSION_MODE
  # ms_performance_prechecker_logger - INFO - [action] export HCCL_OP_EXPANSION_MODE=AIV
  # ms_performance_prechecker_logger - INFO - [reason] 配置通信算法的编排展开位置，代表通信算法的编排展开位置在Device侧的AI Vector Core 计算单元（MindIE 2.0.T3 和 MindIE 2.0.T3.1 是能AIV会有崩溃风险，请不要设置它）
  # ...
  # ms_performance_prechecker_logger - INFO -
  # ms_performance_prechecker_logger - INFO - [system] CPU 可能不是高性能模式
  # ms_performance_prechecker_logger - INFO - [action] 开启 CPU 高性能模式：cpupower -c all frequency-set -g performance
  # ms_performance_prechecker_logger - INFO - [reason] 在相同时延约束下，TPS会有~3%的提升
  # ms_performance_prechecker_logger - INFO -
  # ms_performance_prechecker_logger - INFO - </answer>
  ```
- 参数

  | 参数                 | 说明                                                            |
  | -------------------- | --------------------------------------------------------------- |
  | -t, --check_type  | 检查项类型，可选值：basic, deepseek                                  |
  | -s, --save_env  | 保存环境变量相关改动输出路径，默认值 ms_performance_prechecker_env.sh    |
  | -l, --log_level  | 日志级别，可选值 debug, info, warning, error                         |
- **Python 接口调用**
  ```py
  from ms_performance_prechecker import run_precheck, save_env_contents

  run_precheck(mindie_service_config)  # 执行检查
  save_env_contents("foo_env.sh")  # 保存环境变量相关改动到 foo_env.sh
  ```