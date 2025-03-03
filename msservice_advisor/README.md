# MindStudio profiling advisor

## 介绍
- 执行 mindie benchmark，输出结果的 instance 文件夹
- 执行 `msservice_advisor`
  ```sh
  # 如果没有设置 mindie 环境变量，手动 export 路径
  export MIES_INSTALL_PATH=$HOME/workspace

  # 执行调参建议
  msservice_advisor -i ../instance/
  ```
- 参数

  | 参数                 | 说明                                                            |
  | -------------------- | --------------------------------------------------------------- |
  | -i, --instance_path  | benchamrk 输出的 instance 路径                                  |
  | -t, --target         | 调参指标, 可选值：ttft, firsttokentime, throughput              |
  | -m, --target_metrics | 调参指标的具体项，可选值：average,max,min,P75,P90,SLO_P90,P99,N |