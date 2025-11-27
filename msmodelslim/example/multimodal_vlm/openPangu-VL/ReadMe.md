# PanguVL 量化案例

## 模型介绍

- [openPangu-VL-7B]
openPangu-VL-7B 是基于昇腾 NPU ，基于openPangu-Embedded-7B-V1.1语言基模和RICE-600M视觉编码器训练的高效多模态模型。openPangu-VL-7B 训练了约 3T tokens，具备通用视觉对话、文档理解、目标定位与计数、视频理解、视觉高阶推理等能力。该模型为快思考模型。



#### Pangu_VL模型当前已验证的量化方法

- W8A8量化：openPangu-VL-7B

#### 此模型仓已适配的模型版本

- [openPangu-VL-7B]

## 环境配置

- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

## 量化权重生成

- 量化权重可使用[quant_pangu_vl.py](./quant_pangu_vl.py) 脚本生成，以下提供盘古 openPangu-VL-7B模型量化权重生成快速启动命令。


#### quant_pangu_vl.py 量化参数说明

| 参数名           | 含义           | 默认值  | 使用方法                          |
|---------------|--------------|------|-------------------------------|
| model_path    | 浮点权重路径       | 无默认值 | 必选参数；<br>输入Pangu权重目录路径。    |
| save_path     | 量化权重路径       | 无默认值 | 必选参数；<br>输出量化结果目录路径。          |
| calib_dataset | 量化权重路径       | 无默认值 | 可选参数；<br>量化校准集路径。             |
| anti_method  | 离群值抑制方法 | m2 | 可选参数；<br/> 可选项：m2,m4,m6 |

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


- 1、需安装更新transformers版本（>=4.53.2）


##### openPangu-VL-7B 动态量化

```shell
export QUANT_PATH=your_quant_save_dir
export MODEL_PATH=your_model_ckpt_dir
export CALI_DTATSET=your_cali_dataset_dir
python quant_pangu_vl.py \
--model_path $MODEL_PATH --calib_images $CALI_DTATSET \
--save_directory $QUANT_PATH --w_bit 8 --a_bit 8 --device_type npu \
--trust_remote_code True --anti_method m2 --act_method 3 --is_dynamic True
```


