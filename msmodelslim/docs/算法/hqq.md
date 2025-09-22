# HQQ：权重量化算法说明

## 背景和作用

- **来源**：参考自[mobiusml团队的实现](https://github.com/mobiusml/hqq)
- **问题**：传统量化方法（如MinMax）在权重分布不均匀时，量化误差较大，影响模型精度。
- **目标**：通过迭代搜索最优的偏移量（offset）来最小化量化误差，提高量化模型的精度。HQQ算法固定scale参数，专注于优化offset参数，特别适用于非对称量化场景。

## 使用方式

作为量化器使用，支持per tensor量化粒度的int8对称/非对称量化，int4对称/非对称量化量化通过配置一键量化yaml中的qconfig.weight.method部分启用。下面以w8a16量化为例。


```yaml
- type: "linear_quant" 
  qconfig:
   act:
     scope: "placeholder" # w8a16为仅权重量化，不进行激活值量化
     dtype: "float" 
     symmetric: False # 在仅权重量化场景下，任意配置True/False均可
     method: "dummy" # 配置为"dummy", 即仅权重量化
   weight:
     scope: "per_channel"
     dtype: "int8" 
     symmetric: False #如果使用对称量化，hqq算法不会有效果优化
     method: "hqq" # 配置为"hqq"
```
## 原理和实现

### 原理

HQQ算法基于以下核心思想：

1. **固定scale优化offset**：保持scale参数不变，专注于优化offset参数。
2. **广义软阈值操作**：使用shrink函数对量化误差进行稀疏化处理。
3. **迭代优化**：通过多次迭代来逐步优化量化参数。
4. **贪心更新策略**：只保留能改善量化误差的参数。
5. **收敛判断**：通过相对和绝对误差变化来判断收敛。

算法流程：
```
1. 使用 MinMax 观察器初始量化参数 scale 和 offset。
2. 通过量化参数对权重W进行伪量化，得到量化后的权重W_q和伪量化后的权重W_r。
3. 计算量化误差u = W - W_r。
4. 计算广义软阈值运算符W_e = shrink(u, beta, p) = sign(u) * relu(|u| - |u|^(p-1) / beta)。
5. 计算当前最优的 offset = E[W_q - (W - W_e) / scale]。
6. 比较新旧参数的量化误差，保留更好的参数。
7. 重复步骤2-6直到收敛或达到最大迭代次数，得到最终的量化参数。
```

### 实现

- 算法在 `msmodelslim/quant/quantizer/impl/hqq.py` 中实现，核心函数为 `hqq_calculate_qparam`：
    1. **初始化阶段**：
        - 使用MinMax观察器计算权重的统计信息（min/max值）。
        - 基于统计信息计算初始的量化参数（scale和offset）。
    2. **迭代优化阶段**：
        - 固定scale参数，只优化offset参数。
        - 使用广义软阈值操作对量化误差进行稀疏化处理。
        - 通过期望值计算最优offset：`offset = E[W_q - (W - W_e) / scale]`。
        - 贪心更新策略：只保留能改善量化误差的参数。
    3. **收敛判断**：
        - 相对误差变化：`(best_pnorm - current_pnorm) / best_pnorm < threshold`。
        - 绝对误差变化：`|best_pnorm - current_pnorm| < threshold`。
        - 所有通道都满足收敛条件时提前退出。

## 模型适配

### 接口与数据结构

```python
# HQQ量化器类
class WeightPerChannelHqq(AutoWeightQuantizer):
    def __init__(self, config: QConfig): ...
    
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor: ...
    
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None: ...
    
    def get_q_storage(self) -> QStorage: ...
    
    def get_q_param(self) -> QParam: ...

# 核心算法函数
def hqq_calculate_qparam(weight: QStorage, q_param: QParam) -> QParam: ...
```

### 适配步骤

- **前置要求**：
    - 权重必须是2D张量（如线性层的权重）。
    - 需要提供正确的量化配置（dtype、scope、method、symmetric）。
    - HQQ算法主要用于非对称量化（symmetric=False）。
- **步骤**：
    1. 创建HQQ量化配置：指定量化数据类型、范围、方法和对称性。
    2. 创建量化器实例：使用配置初始化WeightPerChannelHqq。
    3. 初始化权重：调用init_weight方法设置待量化的权重。
    4. 执行量化：调用forward方法进行量化计算。
    5. 获取结果：通过get_q_storage和get_q_param获取量化结果。

### 完整示例

```python
import torch
from msmodelslim.quant.quantizer.base import QConfig, AutoWeightQuantizer
from msmodelslim.core.QAL.qbase import QStorage, QDType

# 1. 创建配置
config = QConfig(
    dtype="int8",
    scope="per_channel", 
    method="hqq",
    symmetric=False  # HQQ主要用于非对称量化
)

# 2. 创建量化器
quantizer = AutoWeightQuantizer.from_config(config)

# 3. 准备权重数据
weight_tensor = torch.randn(256, 512)
weight_storage = QStorage(QDType.FLOAT, weight_tensor)

# 4. 初始化权重
quantizer.init_weight(weight_storage)

# 5. 执行量化
dequantized_weight = quantizer.forward()

# 6. 获取量化结果
q_storage = quantizer.get_q_storage()
q_param = quantizer.get_q_param()

print(f"原始权重形状: {weight_tensor.shape}")
print(f"量化后权重形状: {q_storage.value.shape}")
print(f"量化参数: {q_param}")
```

## 算法参数

HQQ算法内部使用以下参数（可通过修改源码调整）：

```python
SCALE_SEARCH_ITER_NUM = 20                    # 最大迭代次数
SCALE_SEARCH_CONVERGE_THRESHOLD = 1e-10       # 收敛阈值
SCALE_SEARCH_MIN_SCALE = 1e-5                 # 最小缩放因子
HQQ_SHRINK_P = 1                              # 稀疏性参数，p <= 1
BETA = 1000                                   # 软阈值参数
```

## 量化配置参数

```python
QConfig(
    dtype="int8",           # 量化数据类型：int8
    scope="per_channel",    # 量化范围：per_channel（每个通道独立量化）
    method="hqq",          # 量化方法：hqq
    symmetric=False         # 对称量化：False为非对称（HQQ主要用于非对称量化）
)
```

## 适用要求

- **高精度需求**：适用于对精度要求较高的模型量化场景。
- **权重分布不均匀**：特别适合权重分布不均匀的线性层。
- **非对称量化**：HQQ算法主要用于非对称量化场景，能更好地处理权重分布的不对称性。
- **计算成本**：HQQ算法需要多次迭代，某些场景下计算成本较大。
- **初始化依赖**：需要先使用MinMax观察器计算初始量化参数。
- **使用限制**：
    - 目前仅支持int8场景的per_channel量化。
    - 主要用于非对称量化（symmetric=False）。
    - 权重必须是2D张量。
    - 算法固定scale参数，专注于优化offset参数。

## 常见问题排查

### 1. 权重维度错误
**现象**：输入的权重维度错误，导致量化失败。
**解决方案**：检查权重维度是否正确，确保权重是2D张量。

### 2. 量化配置错误
**现象**：量化配置错误，导致量化失败。
**解决方案**：检查dtype、scope、method、symmetric参数设置是否正确。

### 3. 初始化顺序错误
**现象**：初始化顺序错误，导致量化失败。
**解决方案**：必须先调用init_weight，再调用forward。

### 4. 收敛问题
**现象**：如果算法不收敛，可以调整SCALE_SEARCH_CONVERGE_THRESHOLD参数。
**解决方案**：调整SCALE_SEARCH_CONVERGE_THRESHOLD参数。

### 5. 非对称量化限制
**现象**：HQQ算法主要用于非对称量化，对称量化时实际效果等同于minmax算法。
**解决方案**：确保symmetric=False以获得HQQ算法的优化效果。
