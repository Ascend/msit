# Kohya-ss flux lora

本仓库为Kohya-ss实现的flux lora模型NPU适配版本，在原仓基础上进行了NPU亲和的融合算子替换和精度对齐。源码：[Kohya-ss仓库链接](https://github.com/kohya-ss/sd-scripts/tree/sd3)

## 性能统计

**表 1**  训练性能展示表

|     NAME      | 规模 | 模型      |      SPS       | AMP_Type | Torch_Version |
| :-----------: | ---- | --------- | :------------: | :------: | :-----------: |
| Atlas 800T A2 | 1*8  | flux lora | 6.10 samples/s |   BF16   |     2.5.1     |

# 训练资源准备

## 环境准备

1. 搭建torch环境：请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》，环境要求如表2：

   **表 2**  Python版本说明

   |       Software       |             版本             |
   | :------------------: | :--------------------------: |
   |         CANN         |           8.1.RC1            |
   |        Python        |             3.10             |
   |       PyTorch        |            2.5.1             |
   |      DeepSpeed       |            0.15.4            |

2. 安装三方库：按照原仓指导在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
```shell
git checkout 0e8ac4376054e377ce1f7bf59c471c90616a20b9
```
1. 使能融合算子替换：将本目录代码拷贝到原仓主目录下，修改原代码仓中的train_network.py文件，添加NPU融合算子patch：
```python
from library.utils import setup_logging, add_logging_arguments
import patch_flux
```




## 准备数据集

1. 按照如下格式组织数据集

   ```
   data/
   ├── example_0001.png
   └── example_0001.txt
   ```


## 准备预训练模型权重

| 模型        | 链接                                                |
| ----------- | --------------------------------------------------- |
| Flux.1[dev] | https://huggingface.co/black-forest-labs/FLUX.1-dev |

# 运行指导

运行前修改train_flux_lora.sh中的safetensors路径，运行：


```shell
bash train_flux_lora.sh
```

# 精度对齐（可选）
1. 安装精度对齐依赖：安装msprobe
```shell
pip install msprobe
```
2. 固定随机性：在生成式模型中会用到随机噪声，竞品与NPU的随机数生成算子随机结果会有较大差异，需要将双方随机数生成在CPU上完成，在train_network.py中应用patch：
```python
from library.utils import setup_logging, add_logging_arguments
from msprobe import seed_all
import patch_flux
import patch_deter
seed_all(42) # Replace 42 with your own seed
```
3. 运行：在竞品或者NPU依照上述指导运行后，将数据集文件夹同步到另一方再次运行。