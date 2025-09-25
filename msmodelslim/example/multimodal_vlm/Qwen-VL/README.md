# Qwen-VL 量化案例

## 模型介绍

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）。能够以图像、文本和检测框作为输入，生成文本或检测框。该系列模型性能卓越，支持多语言对话、多图交错对话，具备中文开放域定位能力，以及细粒度的图像识别与理解能力。

#### Qwen-VL模型当前已验证的量化方法
| 模型       | 原始浮点权重 | 量化方式 | 推理框架支持情况|
|------------|-------------|---------|----------------|
| Qwen-VL | [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL/tree/main) | W8A8静态量化 | MindIE当前不支持<br>vLLM Ascend当前不支持 |

## 环境配置

- 基础环境配置请参考[安装指南](../../../docs/安装指南.md)

## 生成量化权重

- 量化权重统一使用[quant_qwenvl.py](./quant_qwenvl.py)脚本生成，以下提供Qwen-VL模型量化权重生成快速启动命令。

### 使用案例
- 如果需要使用NPU多卡量化，请先配置环境变量，仅支持1~3卡量化（Atlas 300I Duo 系列产品不支持多卡量化）：
  ```shell
  # 根据实际情况选择多卡，以下3卡量化为例：
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`，让修改后的自定义代码文件能够正确地被加载。(请确保加载的自定义代码文件的安全性)
  
#### 1. Qwen-VL系列
##### Qwen-VL W8A8静态量化
生成Qwen-VL模型量化权重，异常值抑制使用m2算法，在NPU上运行，请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。{校准图片路径}默认为"../calibImages"，用户可根据实际场景替换为其他图片。
  ```shell
  python quant_qwenvl.py  --model_path {浮点权重路径} --calib_images {校准图片路径}  --save_directory {量化权重保存路径} --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True --mindie_format
  ```

### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入Qwen-VL原始浮点权重目录路径。|
| calib_images | 校准集图片路径 | ./calibImages | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/#download)。 示例选取其中2张图片。用户可根据实际场景替换为其他图片。|
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化权重路径。|
| part_file_size | 量化权重文件大小，单位是GB | 默认为None，不限制单个权重文件大小，只生成一个量化权重文件。| 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的大小上限。|
| w_bit | 权重量化bit | 8 | 可选参数;<br>在Qwen-VL量化场景下支持配置为8。|
| a_bit | 激活值量化bit | 8 | 可选参数;<br>在Qwen-VL量化场景下支持配置为8。|
| device_type | 量化运行设备类型 | 'npu' | 可选参数;<br>可选值：['cpu', 'npu']。 |
| trust_remote_code | 是否信任自定义代码 | False | 可选参数;<br>指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。 |
| mindie_format | 多模态理解模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE当前的版本。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../../docs/接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../../docs/接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)