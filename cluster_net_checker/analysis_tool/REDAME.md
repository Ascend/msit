## 降频检测
使用方式 `python freq_analyze.py -d ./profiling_dir`，profiling_dir为解析后的profiling文件路径，profiling文件格式为db  
输出为是否有rank存在空闲时间（aic freq为800mhz），是否有rank存在异常频次（aic不为1800mhz和800mhz）