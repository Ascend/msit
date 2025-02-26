# DeepSeek 量化案例

#### DeepSeek模型当前已验证的量化方法
- W8A8量化：DeepSeek-V2-Lite-Chat-16B, DeepSeek-V2-Chat-236B, DeepSeek-V3, DeepSeek-R1
- W8A16量化：DeepSeek-V2-Lite-Chat-16B, DeepSeek-V2-Chat-236B
 
#### 此模型仓已适配的模型版本
- [Deepseek-V2-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)
- [Deepseek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [Deepseek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)

## 环境配置

- 有关DeepSeek的量化，需要下载安装最新版CANN包及msmodelslim（br_noncom_MindStudio_8.0.0_POC_20251231分支），环境配置以及量化工具安装请参考[使用说明](https://gitee.com/ascend/msit/blob/br_noncom_MindStudio_8.0.0_POC_20251231/msmodelslim/README.md)的方式二安装。
- 如果使用mindie镜像的话，需要下载安装msmodelslim（br_noncom_MindStudio_8.0.0_POC_20251231分支），安装方式参考[使用说明](https://gitee.com/ascend/msit/blob/br_noncom_MindStudio_8.0.0_POC_20251231/msmodelslim/README.md)的方式二安装。

## 运行前必检
在运行脚本前，请根据以下必检项对相关内容进行更改。

- 1、昇腾不支持flash_attn库，运行时需要注释掉权重文件夹中modeling_deepseek.py中的部分代码
- ![img.png](img.png)
- 2、需安装更新transformers版本（>=4.48.2）
- 3、当前昇腾设备不支持FP8格式加载，需要将权重文件夹中config.json中的以下字段删除：
- ![img_1.png](img_1.png)
- 4、修改quant_deepseek_w8a8.py脚本，增加卡数，eg: max_memory={0: "50GiB", 1: "50GiB", "cpu": "1500GiB"}, 可以根据自己的卡数以及显存情况设置，以防OOM显存不够报错。
- ![img_2.png](img_2.png)

## 量化权重生成

- 量化权重可使用[quant_deepseek.py](./quant_deepseek.py)和[quant_deepseek_w8a8.py](./quant_deepseek_w8a8.py)脚本生成，以下提供DeepSeek模型量化权重生成快速启动命令。

#### quant_deepseek.py 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入DeepSeek权重目录路径。 |
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。 |
| a_bit | 激活值量化bit | 8 |大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。 |
| w_bit | 权重量化bit | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。 |
| device_type | device类型 | cpu | 可选值：['cpu', 'npu'] |
| calib_file | 量化校准数据 | teacher_qualification.jsonl | 存放校准数据的json文件。 |
| disable_names | 手动回退的量化层名称 | 默认回退所有down_proj层 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层。 |
| disable_level | L自动回退等级 | L0 | 配置示例如下：<br>'L0'：默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。|
| act_method | 激活值量化方法 | 1 |(1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。|
| anti_method | 离群值抑制参数 | 无默认值 |'m1': SmoothQuant算法。<br>'m2': SmoothQuant加强版算法，推荐使用。<br>'m3': AWQ算法。<br>'m4': smooth优化算法 。<br>'m5': CBQ量化算法。<br>默认为m2。|
| co_sparse	| 是否开启稀疏量化功能 | False | True: 使用稀疏量化功能；<br>False: 不使用稀疏量化功能。 |
| fraction | 模型权重稀疏量化过程中被保护的异常值占比  |0.01| 取值范围[0.01,0.1]|
| use_sigma | 是否启动sigma功能 | False|True: 开启sigma功能；<br>False: 不开启sigma功能。 |
| is_lowbit | 是否开启lowbit量化功能 | False|(1) 当w_bit=4，a_bit=8时，为大模型稀疏量化场景，表示开启lowbit稀疏量化功能。<br>(2) 其他场景为大模型量化场景，会开启量化自动精度调优功能。<br>当前量化自动精度调优框架支持W8A8，W8A16量化。|
| part_file_size | 量化权重文件大小 | 无限制 | 单个量化权重文件大小不超过xGB。|
| use_kvcache_quant | 是否使用kvcache量化功能 | False | True: 使用kvcache量化功能；<br>False: 不使用kvcache量化功能。|
| is_dynamic | 是否使用per-token动态量化功能 | False | True: 使用per-token动态量化；<br>False: 不使用per-token动态量化。 |

#### quant_deepseek_w8a8.py 量化参数说明
| 参数名        | 含义 | 默认值 | 使用方法                       | 
|------------| ---- | --- |----------------------------| 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入DeepSeek权重目录路径。 |
| save_path  | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化结果目录路径。       |
| layer_count  | 量化权重路径 | 无默认值 | 可选参数；<br>用于调试，实际量化的层数。     |
| anti_dataset  | 量化权重路径 | 无默认值 | 可选参数；<br>离群值抑制校准集路径。       |
| calib_dataset  | 量化权重路径 | 无默认值 | 可选参数；<br>量化校准集路径。          |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)


### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

#### DeepSeek-V2/V3/R1 模型量化
##### DeepSeek-V2 w8a16 量化
- 生成DeepSeek-V2模型w8a16量化权重，使用histogram量化方式，在CPU上进行运算
  ```shell
  python3 quant_deepseek.py --model_path {浮点权重路径} --save_directory {W8A16量化权重路径} --device_type cpu --act_method 2 --w_bit 8 --a_bit 16
  ```
- 添加MTP结构到量化模型（可选）
  ```shell
  python3 add_mtp.py --quant_dir {W8A16量化权重路径} --float_dir {浮点权重路径} --new_safetensors_name mtp.safetensors
  ```

##### DeepSeek-V2 w8a8 Dynamic量化
- 生成DeepSeek-V2模型 w8a8 dynamic量化权重，使用histogram量化方式，在CPU上进行运算
```shell
python3 quant_deepseek.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --device_type cpu --act_method 2 --w_bit 8 --a_bit 8  --is_dynamic True
```
- 添加MTP结构到量化模型（可选）
  ```shell
  python3 add_mtp.py --quant_dir {W8A8量化权重路径} --float_dir {浮点权重路径} --new_safetensors_name mtp.safetensors
  ```

##### DeepSeek-V2/V3/R1 w8a8 混合量化(MLA:w8a8量化，MOE:w8a8 dynamic量化)
注：当前量化只支持输入bfloat16格式模型
- 生成DeepSeek-V2/V3/R1模型 w8a8 混合量化权重
  ```shell
  python3 quant_deepseek_w8a8.py --model_path {浮点权重路径} --save_path {W8A8量化权重路径} 
  ```
- 添加MTP结构到量化模型（可选）
  ```shell
  python3 add_mtp.py --quant_dir {W8A8量化权重路径} --float_dir {浮点权重路径} --new_safetensors_name mtp.safetensors
  ```

##### DeepSeek-V2/V3/R1 w4a16 per-group datafree量化(MLA:float，MOE:w4a16量化)
注：当前量化只支持输入bfloat16格式模型
- 生成DeepSeek-V2/V3/R1模型 w4a16 混合量化权重
  ```shell
  python3 convert.py --model_path {浮点权重路径} --save_path {W4A16量化权重路径} 
  ```

##### DeepSeek-V2/V3/R1 w4a16 per-group量化(MLA：float，MOE：w4a16)
注：当前量化只支持输入bfloat16格式模型
- 生成DeepSeek-V2/V3/R1模型 w4a16 量化权重
  ```shell
  python3 quant_deepseek.py \
  --model_path {浮点权重路径} \
  --save_directory {W4A16量化权重路径} \
  --w_bit 4 \
  --a_bit 16 \
  --group_size 64 \
  --is_lowbit True \
  --open_outlier False \
  --device "npu" \
  --calib_file ""
  ```
- 添加MTP结构到量化模型（可选）
  ```shell
  python3 add_mtp.py --quant_dir {W4A16量化权重路径} --float_dir {浮点权重路径} --new_safetensors_name mtp.safetensors
  ```
  
##### DeepSeek量化QA
- Q：报错 This modeling file requires the following packages that were not found in your environment： flash_attn. Run 'pip install flash_attn'
- A: 当前环境中缺少flash_attn库且昇腾不支持该库，运行时需要注释掉权重文件夹中modeling_deepseek.py中的部分代码
- ![img.png](img.png)
- Q：modeling_utils.py报错 if metadata.get("format") not in ["pt", "tf", "flax", "mix"]: AttributeError: "NoneType" object has no attribute 'get';
- A：说明输入的的权重中缺少metadata字段，需安装更新transformers版本（>=4.48.2）
- Q：报错 Unknown quantization type， got fp8 - supported types are：['awq', 'bitsandbytes_4bit', 'bitsandbytes_8bit', 'gptq', 'aqlm', 'quanto', 'eetq', 'hqq', 'fbgemm_fp8']
- A: 由于当前昇腾设备不支持FP8格式加载，需要将权重文件夹中config.json中的以下字段删除：
- ![img_1.png](img_1.png)
- Q: 遇到OOM显存不够报错
- A：修改quant_deepseek_w8a8.py脚本，增加卡数，eg: max_memory={0: "50GiB", 1: "50GiB", "cpu": "1500GiB"}, 可以根据自己的卡数以及显存情况设置。
- ![img_2.png](img_2.png)