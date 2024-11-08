# Collect Ascend AI Processor Data
## 简介
介绍msit profile和采集昇腾AI处理器系统AI数据相关的可选命令
- **命令示例**
```
msit profile --application "msit benchmark -om /home/HwHiAiUser/resnet101_bs1.om" --output=output_data/ --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --sys-pid-profiling=on --dvpp-profiling=on
```

## 涉及可选命令
  | 参数名                    | 描述                                       | 必选   |
  | ------------------------ | ---------------------------------------- | ---- |
  | -o, --output             | 搜集到的profiling数据的存放路径，默认为当前路径下输出output目录                                                                | 否    |
  | --sys-hardware-mem       | 控制DDR，LLC的读写带宽数据采集开关，可选on或off，默认为on。 | 否    |
  | --sys-cpu-profiling      | CPU（AI CPU、Ctrl CPU、TS CPU）采集开关。可选on或off，默认值为off。                           | 否    |
  | --sys-profiling          | 系统CPU usage及System memory采集开关。可选on或off，默认值为off。 | 否    |
  | --sys-pid-profiling      | 进程的CPU usage及进程的memory采集开关。可选on或off，默认值为off。 | 否    |
  | --dvpp-profiling         | DVPP采集开关，可选on或off，默认值为on | 否    |

## 输出文件采集内容介绍

- **分析结果输出的目录结构**
```bash
|--- output_data/
|    |--- profiler/  # 采集的性能数据
|    |    |--- PROF_000001_20230608201922856_LPKNFOADMAQRMDGC/ # msprof保存的数据
|    |    |    |--- host/ # host侧数据
|    |    |    |--- device_0/ #device侧数据，调用多个device推理会有多个和device文件夹
|    |    |    |    |--- timeline/
|    |    |    |    |--- summary/
|    |    |    |    |--- data/
```

- **采集内容（timeline文件夹内）**

| timeline文件名 | 关联参数 | 说明 |
| ----- | ----- | ----- |
|msprof*.json| 所有可生成数据的参数均会在此文件写入数据。|timeline数据总表。对采集到的timeline性能数据按照迭代粒度进行性能展示。详情请参见[timeline数据总表](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0064.html)|
|ddr_*.json|--sys-hardware-mem|DDR内存读写速率。详情请参见DDR[DDR内存读写速率数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0087.html)|
|llc_aicpu_*.json|--sys-hardware-mem|AI CPU的三级缓存使用量，LLC Profiling采集事件设置为capacity时才会导出该文件。文件详情请参见[AI CPU的三级缓存使用量数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0099.html)（Atlas200/300/500推理产品）|
|llc_ctrlcpu_*.json|--sys-hardware-mem|Control CPU三级缓存使用量，LLC Profiling采集事件设置为capacity时才会导出该文件。文件详情请参见[Control CPU三级缓存使用量数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0100.html)（Atlas200/300/500推理产品）|
|llc_read_write_*.json|--sys-hardware-mem|三级缓存读写速率数据。文件详情请参见[三级缓存读写速率数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0097.html)（Atlas推理系列产品）|

- **采集内容（summary文件夹内）**

| summary文件名 | 关联参数 | 说明 |
| ----- | ----- | ----- |
|ddr_*.csv|--sys-hardware-mem|DDR内存读写速率。详情请参见[DDR内存读写速率数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0087.html)|
|llc_aicpu_*.csv|--sys-hardware-mem|AI CPU三级缓存使用量，LLC Profiling采集事件设置为capacity时才会导出该文件。文件详情请参见[AI CPU的三级缓存使用量数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0099.html)（Atlas200/300/500推理产品）|
|llc_ctrlcpu_*.csv|--sys-hardware-mem|Control CPU三级缓存使用量，LLC Profiling采集事件设置为capacity时才会导出该文件。文件详情请参见[Control CPU三级缓存使用量数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0100.html)（Atlas200/300/500推理产品）|
|llc_read_write_*.csv|--sys-hardware-mem|三级缓存读写速率数据。文件详情请参见[三级缓存读写速率数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0097.html)（Atlas推理系列产品）|
|ai_cpu_top_function_*.csv|--sys-cpu-profiling|AI CPU热点函数。文件详情请参见[AI CPU热点函数数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0102.html)|
|ai_cpu_pmu_events_*.csv|--sys-cpu-profiling|AI CPU PMU事件。文件详情请参见[AI CPU PMU事件数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0103.html)|
|ctrl_cpu_top_function_*.csv|--sys-cpu-profiling|Ctrl CPU热点函数。文件详情请参见[Ctrl CPU热点函数数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0104.html)|
|ctrl_cpu_pmu_events_*.csv|--sys-cpu-profiling|Ctrl CPU PMU事件。文件详情请参见[Ctrl CPU PMU事件数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0105.html)|
|ts_cpu_top_function_*.csv|--sys-cpu-profiling|TS CPU热点函数。文件详情请参见[TS CPU热点函数数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0106.html)|
|ts_cpu_pmu_events_*.csv|--sys-cpu-profiling|TS CPU PMU事件。文件详情请参见[TS CPU PMU事件数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0107.html)|
|cpu_usage_*.csv|--sys-profiling|AI CPU、Control CPU利用率。文件详情请参见[AI CPU、Ctrl CPU利用率数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0085.html)|
|sys_mem_*.csv|--sys-profiling|指定device的内存使用情况。详情请参见[系统内存数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0083.html)|
|process_cpu_usage_*.csv|--sys-pid-profiling|进程CPU占用率。生成文件详情请参见[进程CPU占用率数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0086.html)|
|process_mem_*.csv|--sys-pid-profiling|进程内存占用率。文件详情请参见[进程内存占用率数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0084.html)|
|dvpp_*.csv|--dvpp-profiling|DVPP数据。文件详情请参见[DVPP数据](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0101.html)|
