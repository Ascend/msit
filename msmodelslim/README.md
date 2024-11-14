
# msModelSlim

## 介绍

MindStudio ModelSlim，昇腾模型压缩工具。 【Powered by MindStudio】

昇腾压缩加速工具，一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。支持训练加速和推理加速，包括模型低秩分解、稀疏训练、训练后量化、量化感知训练等功能，昇腾AI模型开发用户可以灵活调用Python API接口，对模型进行性能调优，并支持导出不同格式模型，在昇腾AI处理器上运行。

## 使用说明

msModelSlim当前处于逐步开源过程中，计划通过630,930,1230三个版本进行过渡。  

630、930版本支持通过CANN或开源方式使用，两边版本将保持一致，后续功能优化、新增将更新在开源版本中。  
630版本CANN，下载链接  
[arm 版本](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC2/Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run?response-content-type=application/octet-stream)  
[x86 版本](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC2/Ascend-cann-toolkit_8.0.RC2_linux-x86_64.run?response-content-type=application/octet-stream) 
**注意** 该版本存在已知问题，使用modelslim调用接口时，部分功能存在异常。请使用msmodelslim调用。 

版本交替期间提供两种方式使用msModelSlim工具：  
1. 下载安装CANN并配置环境变量  
2. 下载安装开源版本msModelSlim  
    **操作步骤：**
    - git clone下载本仓代码；
    - 进入到刚刚clone下来的msmodelslim的目录 `cd msit/msmodelslim`；
    - 设置CANN环境变量；
    - 进入msmodelslim目录，运行安装脚本 `bash install.sh`;
    - 进入python环境路径 `cd 环境路径/env/.../site-packages/msmodelslim/pytorch/weight_compression/compress_graph/`
    - 给build文件夹相关权限 `sudo chown -R 750 build`
    - 编译weight_compression组件 `sudo bash build.sh {CANN包安装路径}/ascend-toolkit/latest`


### 环境准备

- 使用msModelSlim工具前，需参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha003/devaids/auxiliarydevtool/nottoctopics/zh-cn_topic_0000001800373652.html)》搭建开发环境。
- 安装CANN软件后，需要以CANN运行用户登录环境，执行如下示例命令配置环境变量。

```
source {CANN包安装路径}/ascend-toolkit/set_env.sh
```
- 使用非root用户运行调优任务时，需要管理员用户将运行用户加入驱动运行用户组（例如：HwHiAiUser）中，保证普通用户对run包的lib库有读权限。
- msModelSlim工具依赖Python，以Python3.7.5为例，请以运行用户执行如下命令设置Python3.7.5的相关环境变量。

```
#用于设置python3.7.5库文件路径
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH
#如果用户环境存在多个python3版本，则指定使用python3.7.5版本
export PATH=/usr/local/python3.7.5/bin:$PATH
```
- 用户根据实际需要自行安装以下AI框架包（请注意CANN与MindSpore、PyTorch的版本配套关系）：
  - MindSpore：
    请参考[MindSpore官网](https://www.mindspore.cn/install)安装MindSpore框架。
  - PyTorch：
        - 请参考《[Ascend Extension for PyTorch 配置与安装](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/configandinstg/instg/insg_0001.html)》，安装PyTorch框架、torch_npu插件、Torchvision依赖和Apex混合精度模块。
        - 若PyTorch下需要统计模型的参数量信息，则执行如下命令安装依赖thop。
          如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install thop --user。
## 特性清单
- msModelSlim针对开发者的差异化需求，提供了以下模型压缩方案：

| 功能名称                          | 功能简介                                                                                                                                                  |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [模型低秩分解](msmodelslim/pytorch/low_rank_decompose)                       | 低秩分解是一种矩阵分解技术，它可以将一个大型矩阵分解为若干个较小矩阵的乘积，这些较小矩阵的秩相对较低。低秩分解在很多领域都有应用，如数据分析、机器学习、图像处理等。                                                                    |
| [模型稀疏加速训练](msmodelslim/pytorch/sparse)                      | 稀疏加速算法是一种旨在通过减少模型参数的数量来提高计算效率的训练方法。这种算法基于网络扩增训练的思想，通常涉及到在训练过程中引入额外的参数，然后通过某种方式对这些参数进行筛选或修剪，以实现模型的稀疏化。                                                 |
| [模型蒸馏](msmodelslim/common/knowledge_distill)                          | 蒸馏调优是一种模型压缩技术，它将一个大型、复杂的教师模型的知识转移到一个小型的学生模型中。在这个过程中，学生模型试图模仿教师模型的输出，通常是通过训练学生模型来匹配教师模型的输出或中间层的激活。                                                     |
| [大模型量化](msmodelslim/pytorch/llm_ptq)                         | 大模型量化是一种模型压缩技术，它通过减少模型权重和激活的数值表示的精度来降低模型的存储和计算需求。量化工具通常会将高位浮点数转换为低位定点数，从而直接减少模型权重的体积。                                                                 |
| [大模型稀疏量化](msmodelslim/pytorch/llm_sparsequant)和[权重压缩](msmodelslim/pytorch/weight_compression)                  | 大模型稀疏量化工具结合了模型量化与模型稀疏化两种技术，旨在通过减少模型体积和降低内存及带宽消耗来提升模型的性能。                                                                                              |
| 训练后量化（[onnx](msmodelslim/onnx)/[pytorch](msmodelslim/pytorch/quant/ptq_tools)/[mindspore](msmodelslim/mindspore/quant/ptq_quant)) | 训练后量化不需要重新训练模型，而是在模型训练完成后直接对模型进行量化。                                                                                                                   |
| [量化感知训练](msmodelslim/pytorch/quant/qat_tools)                        | 量化感知训练是一种在模型训练过程中模拟量化效果的训练方法。通过在训练过程中加入量化操作，模型可以适应量化带来的精度损失，从而在量化后的模型上保持较高的性能。                                                                        |
| [Transformer类模型权重剪枝调优](msmodelslim/pytorch/prune/transformer_prune)          | 模型权重剪枝是一种通过移除模型中不重要的权重（即那些对模型性能影响较小的权重）来减少模型复杂度的技术。剪枝后的模型权重更少，从而可以减少模型的存储需求，并可能加快模型的推理速度。                                                             |
| [基于重要性评估的剪枝调优](msmodelslim/pytorch/prune)                  | 基于重要性评估进行剪枝调优是一种常用的方法，它涉及到评估模型中每个权重的重要性，并据此决定哪些权重应该被剪枝。  基于重要性评估的剪枝调优可以显著减少模型的大小，提高模型的推理效率，同时尽量保持模型的性能。这种方法在深度学习模型压缩和加速中非常有用，特别是在需要部署模型到资源受限的环境中的情况下。 |

### [Python API接口说明](docs/Python-API接口说明) 
#### 许可证
[Apache License 2.0](LICENSE)
