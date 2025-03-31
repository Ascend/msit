# msServiceProfiler 工具介绍

## 概述
msServiceProfiler 是一款基于昇腾平台，支持MindIE Service框架和vLLM框架的服务化调优工具。

其性能采集与数据解析能力已嵌入昇腾CANN工具包，支持MindStudio Insight、Chrome Tracing、Grafana多个平台数据可视化。

目前，扩展能力**服务化性能数据比对工具**和**vLLM服务化性能采集工具**已在本仓库开源。

## 硬件环境

|   类型    | 配置参考 |  服务化性能数据比对工具 | vLLM服务化性能采集工具 |
|:-------:|:-------:|:-------:|:-------:|
|   服务器   | Atlas 800I A2 推理产品   | √ | √ |
|   推理卡   | Atlas 300I Duo 推理卡+Atlas 800 推理服务器（型号：3000）   | √ |  |


## 特性清单

### [服务化性能数据比对工具](docs/服务化性能数据比对工具.md)
支持对使用msServiceProfiler工具采集的性能数据进行差异比对，通过比对快速识别可能存在的问题点。

### ️[vLLM服务化性能采集工具](docs/vLLM服务化性能采集工具.md)
基于Ascend-vLLM，提供性能数据采集能力，结合msServiceProfiler的数据解析与可视化能力，可以vLLM服务化推理调试调优。


## 安装指南
目前工具支持源码下载，脚本调用。
#### 源码下载

```shell
git clone https://gitee.com/ascend/msit.git
cd msit/msserviceprofiler
```

## 支持与帮助
🐛 [Issue提交](https://gitee.com/ascend/msit/issues)

💬 [昇腾论坛](https://www.hiascend.com/forum/forum-0106101385921175006-1.html)