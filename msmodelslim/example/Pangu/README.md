# Pangu 量化案例

## 模型介绍

- [pangu_ultra_moe]
  我们在此发布基于昇腾NPU从零训练的盘古 Ultra MoE模型。盘古 Ultra MoE总参数量为718B，激活参数量为39B。
  盘古 Ultra MoE 预训练阶段训练了约18T数据。为保证训练稳定性，我们采用了Depth-Scaled Sandwich-Norm技术。
  同时我们采用了基于EP-Group的负载均衡策略，该方法在保证MoE计算均衡的同时，可以显著提升模型的路由灵活性和领域特化，提升模型效果。

详细报告参见：
* 中文技术报告地址：[盘古 Ultra MoE：昇腾原生的混合专家模型](https://gitcode.com/ascend-tribe/pangu-ultra-moe/blob/main/Pangu_Ultra_MoE_CN_Report.pdf)
* 英文技术报告地址：[Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs](https://arxiv.org/abs/2505.04519)

#### Pangu模型当前已验证的量化方法

- W8A8量化：pangu_ultra_moe

#### 此模型仓已适配的模型版本

- [pangu_ultra_moe]

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重可使用[quant_pangu_ultra_moe_w8a8.py](./quant_pangu_ultra_moe_w8a8.py) 脚本生成，以下提供盘古 Ultra MoE模型量化权重生成快速启动命令。


#### quant_pangu_ultra_moe_w8a8.py 量化参数说明

| 参数名           | 含义           | 默认值  | 使用方法                          |
|---------------|--------------|------|-------------------------------|
| model_path    | 浮点权重路径       | 无默认值 | 必选参数；<br>输入Pangu权重目录路径。    |
| save_path     | 量化权重路径       | 无默认值 | 必选参数；<br>输出量化结果目录路径。          |
| layer_count   | 量化权重路径       | 无默认值 | 可选参数；<br>用于调试，实际量化的层数。        |
| anti_dataset  | 量化权重路径       | 无默认值 | 可选参数；<br>离群值抑制校准集路径。          |
| calib_dataset | 量化权重路径       | 无默认值 | 可选参数；<br>量化校准集路径。             |
| batch_size     | 输入batch size | 4(quant_pangu_ultra_moe_w8a8.py)  | 可选参数；<br>生成量化校准数据时使用的batch size。batch size越大，校准速度越快，但也要求更多的显存和内存，如资源受限，请降低batch size。  |
| from_bf16     | 指定原模型为BF16权重 | 不开启  | 可选参数；<br>开启即指定，不可与from_fp8共存。  |
| mindie_format | 非多模态模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE 2.1.RC1及之前的版本。 |
| quant_mtp | 指定量化模式 | none | 可选参数；<br>none: 不保存mtp权重；<br>float: 保存mtp浮点权重；<br>mix: 保存mtp混合量化权重。|
| dynamic      | 指定动态量化  | 不开启 | 可选参数；<br/>开启即指定。     |
| disable_anti | 关闭异常值抑制 | 不开启 | 可选参数；<br/>开启即指定。     |
| anti_method  | 离群值抑制方法 | m4 | 可选参数；<br/> 可选项：m4,m6 |

注：在量化脚本里面通过transformers库对模型进行加载时，调用`from_pretrained`函数时会指定`trust_remote_code=True`让修改后的modeling文件能够正确的被加载。(请确保加载的modeling文件的安全性)


更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
以及量化参数配置类 [Calibrator](https://gitee.com/ascend/msit/blob/dev/msmodelslim/docs/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)

### 使用案例

- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

##### 运行前必检

pangu_ultra_moe模型较大，且存在需要手动适配的点，为了避免浪费时间，还请在运行脚本前，请根据以下必检项对相关内容进行更改。

- 1、昇腾不支持flash_attn库，运行时需要注释掉权重文件夹中modeling_pangu.py中的部分代码
- ![img.png](img.png)
- 2、需安装更新transformers版本（>=4.48.2）


##### pangu_ultra_moe w8a8 动态量化

```shell
python3 quant_pangu_ultra_moe_w8a8.py --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --dynamic
```

##### pangu_ultra_moe w8a8 混合量化 + mtp 量化

- 生成 pangu_ultra_moe模型 w8a8 mtp 量化权重
```shell
python3 quant_pangu_ultra_moe_w8a8.py --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --dynamic --quant_mtp mix
```

##### Pangu模型量化QA

- Q：报错 This modeling file requires the following packages that were not found in your environment： flash_attn. Run '
  pip install flash_attn'
- A: 当前环境中缺少flash_attn库且昇腾不支持该库，运行时需要注释掉权重文件夹中modeling_pangu.py中的部分代码
- ![img.png](img.png)
- Q：modeling_utils.py报错 if metadata.get("format") not in ["pt", "tf", "flax", "mix"]: AttributeError: "NoneType"
  object has no attribute 'get';
- A：说明输入的权重中缺少metadata字段，需安装更新transformers版本（>=4.48.2）
