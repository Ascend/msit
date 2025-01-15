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
|index |序号 |
|rankId  |rank编号 |
|perpetrator_times  |得票数 |
|count_times  |参与投票次数 |

group_rank_map:
| 名称|含义|
| --- | --- |
|index |序号 |
|rankId  |rank编号 |
|groupName  |通信域名称 |

host_rank_map:
| 名称|含义|
| --- | --- |
|index |序号 |
|rankId  |rank编号 |
|host  |host名称 |

transmit_time_sum:
| 名称|含义|
| --- | --- |
|index |序号 |
|rankId  |rank编号 |
|transmit_time  |通信时长 |
