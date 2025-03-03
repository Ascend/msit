## 降频检测
使用方式 `python freq_analyze.py -d ./profiling_dir`，profiling_dir为解析后的profiling文件路径，profiling文件格式为db  
输出为是否有rank存在空闲时间（aic freq为800mhz），是否有rank存在异常频次（aic不为1800mhz和800mhz）

## 快慢卡慢链路分析
使用方式：
python rank_analyze.py -d ./profiling_dir -o ./output path
profiling_dir为解析后的profiling文件路径，profiling文件格式为db
output path为分析后的文件保存路径
其中各表字段含义为：

vote_result:
| 名称|含义|
| --- | --- |
|rankId  |rank编号 |
|perpetrator_times  |得票数 |
|count_times  |参与投票次数 |

group_rank_map:
| 名称|含义|
| --- | --- |
|rankId  |rank编号 |
|groupName  |通信域名称 |

host_rank_map:
| 名称|含义|
| --- | --- |
|rankId  |rank编号 |
|host  |host名称 |

transmit_time_sum:
| 名称|含义|
| --- | --- |
|rankId  |rank编号 |
|transmit_time  |通信时长 |

slow_link_sum:
| 名称|含义|
| --- | --- |
|opType_relatedRanks_dataSize  |筛选条件 |
|count  |该筛选条件下算子数量 |
|meanNs  |平均值 |
|stdNS  |标准差 |
|minNs  |最小值 |
|q1Ns  |百分之25位数 |
|medianNs  |中位数 |
|q3Ns  |百分之75位数 |
|maxNs  |最大值 |
|sumNs  |总值 |
|offset_ratio  | Z-Score数 |
|max_rank  |此筛选条件下最大值所在卡号 |
|min_rank  |此筛选条件下最小值所在卡号 |

备注：Z-Score 是一种统计方法，用于衡量数据点与均值的标准差距离。 如果某个数据点的 Z-Score 超过阈值（默认为3），则认为它是异常值。
该表用法：该表按照opType、dataSize、related_ranks为筛选条件进行了分组，把分组后数据有Z-Score超过3的数据进行了筛选，按照Z-Score值进行了降序排列。
建议按照表格排序去进行timeline排查。

slow_link_ops:
| 名称|含义|
| --- | --- |
| 名称|含义|
| --- | --- |
|rankId  |rank编号 |
|groupName  |通信域名称 |
|opName  |通信算子名称 |
|host_id  |标识host的唯一id |
|opType  |通信算子类型 |
|dataType  |数据类型 |
|dataSize  |通信量 |
|transmit_time  |传输时间 |
|across_nodes  |跨节点数量 |
|related_ranks  |此通信域关联卡数 |

该表用法：该表是对上述存在异常值的数据的原始数据的汇总，可以拿着上述opType、dataSize、related_ranks为筛选条件去查看原始数据分布。