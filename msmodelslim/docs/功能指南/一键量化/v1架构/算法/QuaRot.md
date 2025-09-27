# QuaRot：基于旋转的离群值抑制算法说明

## 硬件产品支持

| 产品系列 | 支持 |
|---------|------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | ✓ |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | ✓ |
| Atlas 推理系列产品 | ✗ |

## 背景和作用

- **来源**：学术研究/业界 proposed 方法。
- **概述**：QuaRot (Quantization with Rotation)
  是一种用于大语言模型量化的创新算法，通过数学变换有效抑制激活张量中的离群值。该算法通过对权重和激活值进行特定的旋转变换，将离群值"分散"到多个通道中，从而在量化前显著平滑数据的分布，有效降低量化误差。
- **核心思想**：QuaRot 的核心思想是应用一个精心构造的正交变换（旋转矩阵），使得变换后的激活张量在每个通道上的最大值尽可能均衡，从而避免单个通道因存在极端离群值而需要过大的缩放因子，进而提升整体量化精度。

## 使用方式

### 作为Processor使用

```yaml
- type: "quarot"                         # 固定为 `quarot`，用于指定 Processor 类型。
  online: False                          # 控制是否启用在线旋转，默认为 False。
  block_size: -1                         # 整数, 取值范围为-1或大于0的2的幂，表示启用块对角矩阵时每个块的大小，若为-1，表示不进行块对角矩阵处理。
  max_tp_size: 4                         # 整数，默认为4，该配置项仅在启用在线旋转时生效。最大张量并行大小，必须大于0且为2的幂或等于1，拉起模型时设置的tp值必须<=max_tp_size。
  down_proj_online_layers: [ ]            # 整数列表，默认为空。用于指定哪些层的down_proj使用在线旋转。
```

## YAML配置示例

```yaml
spec:
  process:
    - type: "quarot"   
      online: False                      # 控制是否启用在线旋转，默认为 False。
      block_size: -1                     # 旋转矩阵启用块对角矩阵时每个块的大小, 取值范围为-1或2的幂次方，如果大于0必须为2的幂，若为-1，表示不进行块对角矩阵处理
      max_tp_size: 4                     # 最大张量并行大小，默认为4，仅在启用启用在线旋转时生效，必须大于0且为2的幂，拉起模型时设置的tp值必须<=max_tp_size
      down_proj_online_layers: []        # 用于指定哪些层的down_proj使用在线旋转，默认为空
```

## YAML配置字段详解

| 字段名 | 作用 | 类型 | 说明 | 默认值 |
|--------|------|------|------|--------|
| type | 处理器类型标识 | `string` | 固定值，用于标识这是一个QuaRot量化处理器 | `"quarot"` |
| online | 在线旋转开关 | `bool` | 是否启用在线旋转，True表示使用在线旋转，False表示不使用 | `False` |
| block_size | 旋转矩阵的对角块大小 | `int` | 旋转矩阵的对角块大小，取值范围为-1或大于0的2的幂，若为-1表示不进行块对角矩阵处理 | `-1` |
| max_tp_size | 最大张量并行大小 | `int` | 该配置项仅在启用在线旋转时生效，最大张量并行大小，必须大于0且为2的幂或等于1 | `4` |
| down_proj_online_layers | 指定使用在线旋转的down层 | `array[int]` | 用于指定哪些层的down_proj使用在线旋转，类型为由层索引组成的列表 | `[]` |

## 原理和实现

### 原理

1. **核心概念**：
   QuaRot 采用正交旋转矩阵**（如 Hadamard 矩阵）对模型权重和激活值进行变换。正交矩阵满足 Q × Qᵀ = I 的关键性质，确保变换前后模型数学等价。
2. **旋转变换操作：**
   对权重矩阵 W 应用变换：`W' = Qᵀ × W`。
   对激活值 X 应用变换：`X' = X × Q`。
   变换保持计算等价：`X' × W' = (X × Q) × (Qᵀ × W) = X × W`。
3. **计算不变性**：
   旋转变换保持 Transformer 各层的输入-输出映射关系不变。即使层间包含 RMSNorm 操作，因 `RMSNorm(X) = RMSNorm(X × Qᵀ) × Q`，计算不变性依然成立。
4. **优化效果**：
   通过旋转重新分布参数，有效抑制激活值中的离群值，使数值分布更平滑，显著降低后续量化操作的误差，为低比特量化奠定基础。

### 实现

算法在 `msmodelslim/quant/processor/quarot/quarot.py` 中实现，处理流程如下：

#### 1) pre_run阶段

**计算旋转矩阵：**

- 根据从模型适配器中获取的模型维度信息以及配置信息，计算旋转矩阵，如果启用了在线旋转，还需要额外计算在线旋转相关矩阵，并且创建在线旋转矩阵保存器。

**对decoder外的指定层执行层融合与旋转应用：**

- 首先将需要插入旋转操作的Linear层与其相邻的LayerNorm层进行权重融合，将融合权重更新至Linear层，同时将LayerNorm层权重置为1。
- 对Linear层插入旋转操作。

#### 2) preprocess阶段

此阶段逐层循环调度，每次对一个decoder层进行操作。

**获取模型结构信息：**

- 利用模型适配器获取norm_linear、linear_linear、ov_pair。

**层融合：**

- 遍历 norm_linear_pairs 对象，将需要插入旋转操作的Linear层与其相邻的LayerNorm层进行权重融合，将融合权重更新至Linear层，同时将LayerNorm层权重置为1。

**插入离线旋转矩阵：**

- 利用获取到的 norm_linear、linear_linear、ov_pair，对需要的位置插入旋转操作。

**在线旋转操作（可选）：**

- 如果启用了在线旋转，根据配置对需要的down_proj（当前可配置执行在线旋转的层索引）和o_proj层（启用在线旋转时默认操作对象为所有的o_proj层，当前不支持通过配置选层）进行旋转。
- 针对down_proj和o_proj层，插入两种在线旋转操作，在前向传播时启动。

## 模型适配

### 接口与数据结构

该接口组用于使能模型自主适配QuaRot算法，但目前仅支持Qwen3稠密模型（如Qwen3-32B），尚未具备泛化至其他系列模型（如Qwen3 MOE模型或DeepSeek MOE模型）。

```python
# 模型适配QuaRot算法接口
class QuaRotAdapter:
    # 获取模型隐藏层维度信息
    def get_hidden_dim(self) -> int: ...

    # 获取模型头维度信息
    def get_head_dim(self) -> int: ...

    # 获取模型头数量信息
    def get_num_attention_heads(self) -> int: ...

    # 获取获取键值头数量
    def get_num_key_value_heads(self) -> int: ...

    # 获取模型lm_head层名称
    def get_lm_head(self) -> str: ...

    # 获取模型pre_head_layernorm层名称
    def get_pre_head_layernorm(self) -> str: ...

    # 获取模型embedding层名称
    def get_embedding(self) -> str: ...

    # 获取单个decoder层对应的norm_linear对
    def get_layer_wise_norm_liner_pair(self, decoder_module: nn.Module) -> Dict[nn.Module, List[nn.Module]]: ...

    # 获取单个decoder层对应的ov_pair对
    def get_layer_wise_ov_pair(self, decoder_module: nn.Module) -> Dict[nn.Module, List[nn.Module]]: ...

    # 获取单个decoder层对应的up_down对
    def get_layer_wise_up_down_pair(self, decoder_module: nn.Module) -> Dict[nn.Module, List[nn.Module]]: ...

```



### 适配步骤

- **前置要求**：
    - 确保所有返回的模块引用都是实际模型中的模块对象。
    - 模块路径必须与model.named_modules()返回的路径完全一致。
- **步骤**：
    1. 模型适配器继承 `QuaRotAdapter`接口，并实现所有抽象方法，可参考 `msmodelslim/model/qwen3.py`。
    2. 提供模型全局结构信息：`get_hidden_dim()`、`get_head_dim()`、`get_num_attention_heads()`、`get_num_key_value_heads()`。
    3. 提供decoder外部的模型结构信息：`get_lm_head()`、`get_pre_head_layernorm()`、`get_embedding()`。
    4. 提供decoder内部的模型结构信息：`get_layer_wise_norm_liner_pair()`、`get_layer_wise_ov_pair()`、
       `get_layer_wise_up_down_pair()`。

### 适用范围与局限性

- **模型结构限制**：当前的适配器仅支持Qwen3 Dense模型（如Qwen3-8B/14B/32B），尚未支持Qwen3 Moe模型或DeepSeek Moe模型等其他系列模型。
- **张量并行限制**：若在配置中启用了在线旋转，在使用推理引擎以TP并行的方式进行部署时，需要保证`tp_size`为2的幂，并且`tp_size`需要小于等于QuaRot的配置参数`max_tp_size`，否则必然导致精度异常。
- **在线旋转限制**：使用在线旋转通常可以获得更好的精度，但需要在部署时插入在线旋转的算子，这依赖于推理框架的支持，也会一定程度降低性能，在推理引擎都支持的背景下
  ，用户需要自行权衡精度与性能。

## 常见问题排查

### 1. 旋转矩阵创建失败

- **现象**：输入模型的维度暂未被支持，导致旋转矩阵创建失败。
- **解决方案**：请先确定指定维度的 Hadamard 矩阵存在，参考 `msmodelslim/quant/processor/quarot/hadamard_txt` 添加特定维度的矩阵，并且在 `msmodelslim/quant/processor/quarot/hadamard.py` 进行补充。

### 2. 张量并行配置错误

- **现象**：在使用推理引擎以TP并行的方式进行部署时出现精度异常。
- **原因**：`tp_size`不是2的幂，或者`tp_size`大于QuaRot配置的`max_tp_size`。
- **解决方案**：
  - 确保`tp_size`为2的幂（如1、2、4、8等）。
  - 确保`tp_size` ≤ `max_tp_size`。
  - 检查推理引擎是否支持在线旋转算子。

### 3. 在线旋转性能问题

- **现象**：启用在线旋转后推理性能下降明显。
- **原因**：在线旋转需要插入额外的算子，会增加计算开销。
- **解决方案**：
  - 根据精度要求权衡是否启用在线旋转。
  - 考虑仅使用离线旋转（`online: False`）来平衡精度和性能。
  - 确保推理框架对在线旋转算子有良好支持。

### 4. 模型适配失败

- **现象**：模型适配器无法正确识别模型结构，导致旋转操作插入失败。
- **原因**：模型结构不兼容或适配器实现不完整。
- **解决方案**：
  - 确保模型基于Transformer decoder架构。
  - 检查适配器是否正确实现了所有QuaRotAdapter接口方法。
  - 参考 `msmodelslim/model/qwen3.py` 的实现示例。
