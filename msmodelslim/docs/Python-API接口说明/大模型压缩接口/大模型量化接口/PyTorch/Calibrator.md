## Calibrator

### 功能说明
量化参数配置类，通过Calibrator类封装量化算法。

### 函数原型
```python
Calibrator(model, cfg: quantconfig, calib_data=None, disable_level='L0', all_tensors=None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 模型。| 必选。<br>数据类型：PyTorch模型。 |
| cfg | 输入 | 已配置的QuantConfig类。| 必选。<br>数据类型：QuantConfig。 |
| calib_data | 输入 | LLM大模型量化校准的数据，输入真实数据用于Label-Free量化。| 可选。<br>数据类型：object。<br>默认值为None，为Data-Free场景，Label-Free场景必须输入。<br>输入模板：\[[input1],[input2],[input3]]。|
| disable_level | 输入 | L自动回退等级，在模型精度损失大可以适当提升等级，但回退层数不可以大于模型总层数。| 可选。<br>数据类型：object。<br>配置示例如下：(1)'L0':默认值，不执行回退。(2)'L1'：回退1层。(3)'L2'：回退2层。(3)'L3'：回退3层。(4)'L4'：回退4层。(5)'L5'：回退5层。<br>以此类推。|
| all_tensors | 输入 | 用于逐层量化校准的{name:tensor}。| 可选。<br>数据类型：dict。<br>默认值为None，采用默认配置即可。|

### 调用示例
根据实际需求，在QuantConfig初始化中完成所有参数的配置。
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(dev_type='cpu', pr=0.5, mm_tensor=Flase)
model = AutoModel.from_pretrained('THUDM/chatglm2-6b', torch_dtype=torch.float32, trust_remote_code=True).cpu()   #根据模型实际路径配置
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
```
