# NPU-Monitor

## 介绍

NPU-Monitor，资源利用率监控。 【Powered by PAE】

对NPU的HBM占用，算力利用率和功耗采样，支持频率小于1s级采样，支持依次执行多任务统计

## 使用说明

将dcmi_interface_api.h传到目录monitor_tool下
```
cp /usr/local/dcmi/dcmi_interface_api.h ./monitor_tool
```

生成可执行文件monitor-info
```
g++ get_monitor_info.cpp /usr/local/dcmi/libdcmi.so -o monitor-info -lm
```

精度性能集采前开启(生成record文件)
```
watch -n 0.1 ./monitor-info
======依次执行ceval、mmlu等数据集测试任务======
```

CTRL+C退出监测，编译执行任务切分(得到task1、task2、task3....文件)
```
g++ task_split.cpp -o task-split
./task-split
```

查看0卡在执行task1的过程中资源利用率情况：
```
g++ calculate.cpp -o calculate
./calculate task1/device_0.csv
```

#### 许可证
[Apache License 2.0](LICENSE)

