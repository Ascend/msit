# 蚂蚁 Profiling 数据格式转换解析说明

## 概述

`vllm_data_converter.py` 脚本专门用于处理和转换蚂蚁采集的 vLLM profiling 数据。该脚本将原始 JSON 格式数据转换为适配华为 profiling 解析的 SQLite 数据库文件，实现与华为数据解析工具的无缝对接。




## 功能特点

### 1. 文件处理

* 递归查找目录中的所有 JSON 文件

* 从文件名中提取角色和进程 ID（格式：`{角色}_{PID}.json`）

* 支持批量处理多个文件

### 2. 数据转换

* **方法映射**：将复杂的 vLLM 方法名映射为简化的域名/名称对
  
* **字段重命名**：使用 `MESSAGE_KEY_MAPPING` 转换字段名称
  
* **字段过滤**：通过 `FIELDS_TO_DELETE` 移除不必要的字段
  
* **特殊处理**：处理特定方法如 `processOutputs`、NPU 操作组等
  

### 3. 特殊处理逻辑

**请求 ID 处理**

    # 跟踪批处理和模型执行的迭代计数
    batch_rid_counter = defaultdict(int)
    model_exec_rid = defaultdict(int)

**NPU 模型执行分组**

将相关的 NPU 操作进行分组：

* `npuModelExec`、`npuReqsProc` 和 `forward` 记录
  
* 创建 `preprocess` 和 `postprocess` 记录，修改为对应的时间关系
  

### 4. 输出生成

* 生成结构化 SQLite 数据库，包含两个核心表：
  
  * `Mstx`表：主要跟踪数据
    
    | 字段  | 类型  | 描述  |
    | --- | --- | --- |
    | markId | INTEGER | 连续记录标识符（按时间戳对记录进行排序，自动添加连续的 `markId` 用于记录标识） |
    | role | TEXT | 从文件名提取的角色 |
    | pid | INTEGER | 从文件名提取的进程 ID |
    | method | TEXT | 原始方法路径 |
    | timestamp | REAL | 开始时间戳（纳秒级） |
    | endTimestamp | REAL | 结束时间戳（纳秒级） |
    | tid | INTEGER | 线程 ID |
    | message | TEXT | 转换后的 JSON 格式数据信息 |
    
  * `Meta`表：文件元数据
    
    | 字段  | 类型  | 描述  |
    | --- | --- | --- |
    | file_path | TEXT | 原始文件路径 |
    | role | TEXT | 文件角色 |
    | pid | INTEGER | 进程 ID |
    | name | TEXT | 元数据项名称 |
    | value | TEXT | 元数据项值 |
    



## 使用方法

### 环境要求

* Python >= 3.10
  
* pandas >= 2.2
  
* numpy >= 1.24.3
  
* psutil >= 3.7.5
  
* scipy >= 1.7.2
  

### 安装步骤

    # 1. 获取代码
    git clone https://gitcode.com/Ascend/msit.git -b msserviceprofiler_dev
    cd msit/msserviceprofiler/
    
    # 2. 配置环境
    export PYTHONPATH=$PWD:$PYTHONPATH
    
    # 3. 安装依赖
    pip install pandas numpy psutil scipy

### 使用方法

#### 步骤1. 数据格式转换

##### 基础用法

    python data_transformation/vllm_data_converter.py /path/to/trace/files

##### 高级选项

    python data_transformation/vllm_data_converter.py /path/to/trace/files \
      -o /custom/output/directory \
      --level debug

**命令行参数**

| 参数  | 缩写  | 描述  | 默认值 |
| --- | --- | --- | --- |
| `input_dir` | -   | 包含 JSON 跟踪文件的输入目录 | 必填  |
| `--output` | `-o` | 输出目录路径 | `./prof` |
| `--parse-all` | -   | 处理所有方法（包括默认跳过的），不建议开启 | 关闭  |
| `--level` | `-l` | 日志级别（debug, info, warning, error, critical） | `info` |

##### 输入文件格式

**文件命名约定**

输入文件应遵循命名格式：`{角色}_{PID}.json`

示例：

* `worker_12345.json`
  
* `engine_67890.json`
  

**输入数据格式标准**

每个 JSON 文件包含多行记录，每行为独立 JSON 对象：

    {
      "method_identifier": "vllm.entrypoints.openai.serving_chat.OpenAIServingChat.create_chat_completion",
      "start_ms": 1630000000.123,
      "end_ms": 1630000000.456,
      "thread_id": 12345,
      "trace_data_json": "{\"request_ids\": [\"req-1\"], \"prompt_len\": 100}"
    }

##### 输出文件

**位置**：默认生成在 `./prof` 目录**命名**：`ms_service_YYYYMMDD_HHMMSS.db`（时间戳自动生成）`

输出的SQLite 数据库结构同上述”输出生成“中提及的 Mstx 表（主要跟踪数据）与Meta 表（元数据）

#### 步骤2. 数据解析

    python ms_service_profiler/parse.py \
      --input-path=/absolute/path/to/database/directory \
      --output-path=/custom/parse/output/directory

**参数说明**：

* `--input-path`：**必须为绝对路径**，指向数据库文件目录
  
* `--output-path`：可选，默认输出到 `./output`
  

更多输入参数说明详见：[执行解析-MindStudio8.1.RC1-昇腾社区](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/T&ITools/Profiling/atlasprofiling_16_0033.html)

结果文件说明详见：[解析结果-MindStudio8.1.RC1-昇腾社区](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/T&ITools/Profiling/atlasprofiling_16_0034.html)




## 配置说明

### 方法映射规则

在 `METHOD_MAPPING` 字典中配置方法名映射：

    # 映射规则说明：
    # - domain: 确定在 trace 图中的泳道位置
    # - name: 显示在trace图色块上的简化名称
    METHOD_MAPPING = {
        "原始.方法.完整路径": {
            "domain": "功能域分类",
            "name": "显示名称"
        }
    }

### 方法过滤配置

在 `SKIP_METHODS` 列表中配置默认跳过不需要解析的方法：

    SKIP_METHODS = [
        "不需要处理的方法.路径"
    ]




## 注意事项

1. **文件权限**：确保脚本对输入目录有读取权限，对输出目录有写权限。
  
2. **文件格式**：输入文件必须是有效的 JSON 格式，每行一个 JSON 对象。
  
3. **内存使用**：处理大量文件时，注意内存使用情况。
  
4. **日志文件**：脚本会在当前目录生成 `data_transformation.log` 日志文件。