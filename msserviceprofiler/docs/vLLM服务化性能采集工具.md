# vLLM服务化性能采集工具
## 概述
vLLM是一款广受欢迎的大模型推理框架，具备投机推理和自动前缀缓存等关键功能，使其在学术界和工业界都得到了广泛应用。Ascend-vLLM是华为云针对NPU优化的推理框架，继承了vLLM的优点，并通过特定优化实现了更高的性能和易用性。

### 工具定位
本工具基于Ascend-vLLM，提供性能数据采集能力，结合msServiceProfiler的数据解析与可视化能力，进行vLLM服务化推理调试调优。

### 关键特性
- 多维度指标
  - 请求状态变化
  - KV Cache等存储资源消耗情况
  - 组batch、模型执行过程记录

## 版本配套关系
| vLLM服务化性能采集工具 |    Ascend-vLLM     |  CANN   | vLLM |
|:-------------:|:------------------:|:-------:|:----:|
|     当前版本      | v0.6.3 | 8.0.RC3 |  v0.6.3   |
其他环境依赖查看[版本说明和要求](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_91203.html)。

## 环境准备
#### 1. 参考Ascend-vLLM[准备推理环境](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_91203.html)，成功启动推理服务。
#### 2. 下载msit源码
```shell
git clone https://gitee.com/ascend/msit.git
```

## 快速入门
### 使用前准备
#### 1. 在Ascend-vLLM中导入采集工具接口。
步骤1. 输入`pip show vllm`，查看vllm安装路径，记为`${vllm_install_path}`。

步骤2. 在`${vllm_install_path}/vllm/__init__.py`文件中添加如下代码：
```
import vllm_profiler.vllm_profiler_0_6_3
```
#### 2. 准备性能采集配置文件。
在任意路径下新建json文件，可自定义文件名，此处以ms_service_profiler_config.json为例，假设数据落盘路径为`${logs_prof}`。
```
{
    "enable": 1,
    "prof_dir": "${logs_prof}",
    "profiler_level": "INFO"
}
```
|   参数   | 说明                                                       | 是否必选  | 
|:------:|:---------------------------------------------------------|:-----:|
|   enable   | 是否开启性能数据采集的开关：<br/>0：关闭<br/>1：开启<br/>默认值：0               |   是   |
|   prof_dir   | 采集到性能数据的存放路径，支持用户自定义。<br/>默认值：$HOME/.ms_service_profiler |   否   |
|   profiler_level   | 数据采集等级。默认值为"INFO"，指普通级别的性能数据。                            |   否   |

#### 2. 配置环境变量。
步骤1. 将采集工具代码路径加入PYTHONPATH。
```
export PYTHONPATH=$HOME/msit/msserviceprofiler/ms_service_profiler_ext/:$PYTHONPATH
```
步骤2. 将采集配置文件路径加入SERVICE_PROF_CONFIG_PATH。
```
export SERVICE_PROF_CONFIG_PATH=ms_service_profiler_config.json
```
步骤3.(可选) 指定卡拉起服务，如下为指定1卡运行，可使用`npu-smi info`查看设备情况。
```
export ASCEND_RT_VISIBLE_DEVICES=1
```
### 采集数据
参考Ascend-vLLM资料[启动推理服务](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_91206.html)，此处仅展示基础操作。
#### 1. 服务端拉起vLLM服务
```
python -m vllm.entrypoints.openai.api_server --model ${container_model_path} \
--max-num-seqs=256 \
--max-model-len=4096 \
--max-num-batched-tokens=4096 \
--dtype=float16 \
--tensor-parallel-size=1 \
--block-size=128 \
--host=${docker_ip} \
--port=8080 \
--gpu-memory-utilization=0.8 \
--trust-remote-code
```
 参数   | 说明                                                                                    |
|:------:|:---------------------------------------------------------------------------------------------------|
|   container_model_path   | 模型地址，模型格式是HuggingFace的目录格式。即上传的HuggingFace权重文件存放目录。          |
|   docker_ip   | 服务部署的IP，${docker_ip}替换为宿主机实际的IP地址，默认为None，举例：参数可以设置为0.0.0.0。       |

更多参数说明查看[启动推理服务](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_91206.html)。

#### 2. 客户端发送请求
```
curl -X POST http://${docker_ip}:8080/generate \
-H "Content-Type: application/json" \
-d '{           
      "prompt": "hello",
      "max_tokens": 100,
      "temperature": 0
}'
```
#### 3. 查看采集结果
请求发送结束后，可在`${logs_prof}`路径下，查看落盘的性能原始数据。

调用`msprof --export=on --output=${logs_prof}/PROF_xxx_xxx_xxx`命令行可初步解析该目录下所有的落盘数据，生成msproftx.db文件。

**注意**：使用msprof解析需要替换CANN 8.1.RC1及以上版本，建议在vLLM容器外解析，不要破坏容器环境。

## 结果说明
### 1. 执行推理处理时间数据
① modelExec，表示模型执行时间
```
rid: 请求ID
batch_type: batch类型
batch_size: batch大小
```

② Forward, 模型前向计算时间
```
rid: 请求ID
```

### 2. 请求队列管理状态变化，及组batch过程数据
① Enqueue, Dequeue，表示请求入队、出队
```
rid: 请求ID
QueueSize：当前队列大小
scope：队列名称，通常含有waiting、running队列
```

② BatchSchedule，表示调度信息
```
rid: 当前调度batch中的请求ID列表
QueueSize：当前队列大小
iter_size:当前迭代返回token长度
```

③ ReqState，表示请求状态变化信息
```
rid: 请求ID
WAITING+、RUNNING+、FINIEHED+：请求状态名，值为1表示当前状态，值为-1表示前一状态，值为+1表示后一状态。
```
|   状态名    |   说明    |
|:--------:|:-------:|
| WAITING  | 请求等待状态。 |
| RUNNING  | 请求执行状态。 |
| FINIEHED | 请求结束状态。 |

### 3. kv cache数据
① Allocate，表示请求分配kvcache block字段
```
domain：表示当前为kvcache相关信息
rid: 请求ID
deviceBlock: 分配的block数量
```

② AppendSlot，表示请求过程中新增内存进行缓存的字段
```
domain：表示当前为kvcache相关信息
rid: 请求ID
deviceBlock: 追加的block数量
```

③ Free，请求过程中释放的缓存字段
```
domain：表示当前为kvcache相关信息
rid: 请求ID
deviceBlock: 释放的block数量
```

④ GetCacheHitRate，请求过程中缓存的命中率
```
domain：表示当前为kvcache相关信息
cpuHitCache: cpu缓存命中率
hitCache: gpu缓存命中率
```

### 4. request数据
① httpReq，表示请求到达
```
domain：表示当前为http请求相关信息
rid: 请求ID
```

②  httpRes，表示请求返回
```
domain：表示当前为http请求相关信息
rid: 请求ID
recvTokenSize：表示请求输入长度
replyTokenSize：表示请求输出长度
```
