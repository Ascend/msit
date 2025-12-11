# 解析工具支持Torch Profiler

## 概述

msserviceprofiler解析工具是一个用于解析性能数据的命令行工具，支持解析堆栈数据。

## 使用方法

### 基本命令

```bash
python3 -m ms_service_profiler.parse --input-path ${PATH}/prof_dir/
```

## 参数说明

| 参数 | 说明 | 是否必选 |
|------|------|----------|
| `--input-path` | 指定性能数据所在路径，会遍历读取该路径下所有名为 `{worker_name}_{时间戳}_ascend_pt` 的文件夹，进行解析。 | 是 |
| `--output-path` | 指定解析后文件生成路径，默认为当前路径下的 output 目录。堆栈数据会放置到 output 下的 `chrome_tracing.json` | 否 |
| `--log-level` | 设置日志级别，取值为：<br>- `debug`：调试级别。该级别的日志记录了调试信息，便于开发人员或维护人员定位问题。<br>- `info`：正常级别。记录工具正常运行的信息。默认值。<br>- `warning`：警告级别。记录工具和预期的状态不一致，但不影响整个进程运行的信息。<br>- `error`：一般错误级别。<br>- `fatal`：严重错误级别。<br>- `critical`：致命错误级别。 | 否 |
| `--format` | 设置性能数据输出文件的导出格式，取值为：<br>- `csv`：表示只导出 csv 格式的结果文件。一般用于原始落盘数据量过大（通常为>10G）场景，仅导出该格式文件可减少数据解析耗时，该格式文件包含每轮请求调度耗时、模型执行耗时、KVCache显存占用情况等。<br>- `json`：表示只导出 json 格式的结果文件。一般用于仅需使用 trace 数据进行分析的场景，仅导出该格式文件可减少数据解析耗时，该格式文件包含服务化框架推理全过程 timeline 图。<br>- `db`：表示只导出 db 格式的结果文件。一般用于仅通过 MindStudio Insight 工具分析结果数据场景，该格式文件包含全量数据解析结果，可直接在 MindStudio Insight 中完全展示。<br>不使用 format 则默认全部导出，可以配置一个或多个参数。 | 否 |

## 输出文件说明

解析完成后，堆栈数据会存在以下json文件中：

- `chrome_tracing.json`：可用于可视化分析