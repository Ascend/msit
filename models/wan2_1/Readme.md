# Wan-Video

本仓库为DiffSynth-Studio实现的Wan-Video模型NPU适配版本，源码可参考：[DiffSynth-Studio仓库链接](https://github.com/modelscope/DiffSynth-Studio/tree/03ea27893ff1fd0fcd3239d69652e5ef8a363c9a)

## 性能统计

**表 1**  训练性能展示表

|     NAME      | 规模 | 模型         |             SPS             | AMP_Type | Torch_Version |
| :-----------: | ---- | ------------ | :-------------------------: | :------: | :-----------: |
|     竞品A     | 1*8  | 1.3B-t2v     |       1.12 samples/s        |   BF16   |     2.1.0     |
| Atlas 800T A2 | 1*8  | 1.3B-t2v     |       0.96 samples/s        |   BF16   |     2.1.0     |
| Atlas 800T A2 | 1*16 | 14B-i2v-480p | 0.16samples/s(双机折算单机) |   BF16   |     2.1.0     |

# 训练资源准备

## 环境准备

1. 搭建torch环境：请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》，环境要求如表2：

   **表 2**  Python版本说明

   |       Software       |             版本             |
   | :------------------: | :--------------------------: |
   |         CANN         |            8.0.1             |
   |        Python        |             3.10             |
   |       PyTorch        |            2.1.0             |
   | Python其他三方库依赖 | 参考根目录下requirements.txt |

2. 安装三方库：在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。

   ```shell
   pip install -e .                    # 安装requirement中的三方库与diffsynth
   ```

3. 修改三方库代码：对于文件`site-packages/lightning/fabric/accelerators/cuda.py` 155行的函数`_is_ampere_or_later`，将其返回修改为True




## 准备数据集

1. 按照如下格式组织数据集

   ```
   data/example_dataset/
   ├── metadata.csv
   └── train
       ├── video_00001.mp4
       └── image_00002.jpg
   ```
2. 构建`metadata.csv`文件，包含对数据的描述:
   ```
   file_name,text
   video_00001.mp4,"video description"
   image_00002.jpg,"video description"
   ```
   模型训练支持图像与视频，图像作为单帧视频处理

3. 使用scripts目录下的对应任务脚本处理数据，注意脚本中的相关信息修改：

   ```shell
   bash scripts/process_data_i2v_1p.sh
   bash scripts/process_data_t2v_1p.sh
   ```

最终输出数据文件夹格式：

```
data/example_dataset/
├── metadata.csv
└── train
    ├── video_00001.mp4
    ├── video_00001.mp4.tensors.pth
    ├── video_00002.mp4
    └── video_00002.mp4.tensors.pth
```



## 准备预训练模型权重

| 模型           | 链接                                              |
| -------------- | ------------------------------------------------- |
| t2v，1.3B      | https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B     |
| i2v，14B，480P | https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P |
| i2v，14B，720P | https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P |

# 运行指导

启动脚本均在模型根目录scripts中，运行前注意修改对应数据、权重、输出路径等信息。

## t2v-1.3B, 8卡

```shell
bash scripts/train_full_1.3B_8p.sh
```



## i2v-14B, 32卡

在主节点与副节点执行如下命令：注意  MATER_ADDR与NODE_RANK修改

```
bash scripts/train_full_14B_32p.sh
```
