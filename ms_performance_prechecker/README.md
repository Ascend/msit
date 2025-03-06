# MindStudio Performance Prechecker

## 介绍
- **基本功能** 检查当前环境中 log level、cpu 绑核、内核版本等，是否达到最佳要求，并给出相应建议
- **安装**
  ```sh
  pip install ms-performance-prechecker
  ```
- **执行**
  ```sh
  ms_performance_prechecker
  ```
- 参数

  | 参数                 | 说明                                                            |
  | -------------------- | --------------------------------------------------------------- |
  | -t, --check_type  | 检查项类型，可选值：basic, deepseek                                  |
