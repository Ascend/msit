# 浮点稀疏：基于 ADMM （Alternating Direction Method of Multipliers，交替方向乘子法）的模型稀疏化算法说明

## 背景和作用

- **来源**：华为自研。
- **问题**：现有W8A8S量化方法虽然支持权重稀疏量化，但稀疏率较低，在 Atlas 300I Duo 推理卡压缩单元上难以实现理想的压缩效果。此外，为满足精度要求通常需要回退部分网络层，这显著降低了模型的推理性能。因此，我们提出对浮点权重进行稀疏化处理，结合硬件压缩单元实现更高的压缩率，在保证模型精度的同时显著提升推理性能。
- **目标**：通过ADMM（交替方向乘子法）算法实现模型浮点稀疏化，结合L2量化保持重要位置的精度，在保证模型性能的同时实现高压缩率。

## 使用方式

作为稀疏处理器使用：

```python
from msmodelslim.quant.processor.sparse.float_sparse import FloatSparseProcessor, FloatSparseProcessorConfig

# 创建浮点稀疏处理器配置
config = FloatSparseProcessorConfig(
    sparse_ratio=0.3,           # 稀疏比例：30%
    include=["*"],              # 包含所有模块
    exclude=[]                  # 不排除任何模块
)

# 创建稀疏处理器
processor = FloatSparseProcessor(model, config)
```

## YAML配置示例

```yaml
spec:
  process:
    - type: "float_sparse"
      sparse_ratio: 0.3          # 稀疏比例，0-1之间，默认0.3。
      include: [ "*" ]           # 包含的层模式，支持通配符。
      exclude: ["*self_attn*"]   # 排除的层模式，支持通配符。
```

## YAML配置字段详解

| 字段名 | 作用 | 数据类型 | 默认值 | 说明 |
|--------|------|----------|--------|------|
| type | 处理器类型标识 | string | - | 固定值"float_sparse"，用于标识该对象为浮点稀疏量化处理器。 |
| sparse_ratio | 稀疏比例 | float | 0.3 | 稀疏比例，0-1之间，默认0.3。 |
| include | 包含的层模式 | array[string] | ["*"] | 支持通配符匹配，指定要执行浮点稀疏量化的层。 |
| exclude | 排除的层模式 | array[string] | [] | 支持通配符匹配，优先级高于include。 |

## 原理和实现

### 原理

浮点稀疏算法基于以下核心思想：

1. **ADMM优化**：使用交替方向乘子法求解带约束的优化问题，找到最优的权重稀疏模式。
2. **激活统计**：通过前向hook收集激活统计信息，构建Hessian矩阵。
3. **迭代稀疏**：通过多次迭代逐步优化稀疏模式，平衡稀疏率和模型精度。
4. **精度保护**：使用L2量化保持重要位置的精度，避免关键权重被过度压缩。

算法流程：

```text
1. 预处理阶段：安装前向hook，收集激活统计信息，构建Hessian矩阵。
2. ADMM稀疏化：使用ADMM算法求解最优稀疏模式。
3. 迭代优化：通过多次迭代优化稀疏结果。
4. 精度保护：识别重要权重位置，应用L2量化保持精度。
5. 模块部署：将稀疏化后的模块转换为量化模块。
```

### 实现

- 算法在 `msmodelslim/quant/processor/sparse/float_sparse.py` 和 `admm.py` 中实现：

#### ADMM稀疏器核心类

```python
class AdmmPruner:
    def __init__(self, layer: nn.Linear): ...
    
    def add_batch(self, inp: torch.Tensor): ...
    
    def fasterprune(self, sparse_ratio: float): ...
    
    def free(self): ...
```

#### 浮点稀疏处理器

```python
class FloatSparseProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter): ...
    
    def preprocess(self, request): ...
    
    def postprocess(self, request): ...
```

### 核心算法步骤

1. **统计信息收集**：
   - 安装前向hook收集输入激活数据。
   - 累积Hessian矩阵：`H += X^T * X`。
   - 计算行缩放因子：`scaler_row += ||X_i||_2^2 / n_samples`。

2. **ADMM稀疏化**：
   - 归一化Hessian矩阵和权重。
   - 设置初始惩罚参数：`rho0 = PERCDAMP * mean(diag(H))`。
   - 计算Hessian逆矩阵。
   - 执行ADMM主循环：
     - 投影到稀疏空间：`sparse_weights = (weights + lambda) * mask`。
     - 更新拉格朗日乘子：`lambda += (weights - sparse_weights)`。
     - 更新权重：`weights = H_inv * (H*weights + rho*(sparse_weights - lambda))`。

3. **精度保护**：
   - 使用量化误差和缩放因子的乘积作为重要性度量。
   - 选择top-k%的重要位置保持精度。
   - 应用L2量化：保持重要位置精度，其他位置进行量化。

## 模型适配

### 接口与数据结构

```python
# 浮点稀疏处理器配置
class FloatSparseProcessorConfig(AutoProcessorConfig):
    type: Literal["float_sparse"] = "float_sparse"
    sparse_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)

# 浮点稀疏处理器
class FloatSparseProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter): ...
    
    def preprocess(self, request): ...
    
    def postprocess(self, request): ...
```

### 适配步骤

- **前置要求**：
  - 模型必须包含nn.Linear模块。
  - 需要提供校准数据集用于收集激活统计信息。

- **步骤**：
  1. 创建浮点稀疏配置：指定稀疏比例、包含模块及排除模块。
  2. 创建处理器实例：使用配置初始化FloatSparseProcessor。
  3. 预处理阶段：安装hook收集统计信息。
  4. 后处理阶段：应用ADMM稀疏算法。
  5. 模块部署：转换为W16A16s模块，用于保存浮点稀疏后的模型。

### 完整示例
注：下面完整示例只能在 NPU 环境上运行
```python
import torch
import torch.nn as nn
from msmodelslim.quant.processor.sparse.float_sparse import FloatSparseProcessor, FloatSparseProcessorConfig
from msmodelslim.core.base.protocol import BatchProcessRequest

# 1. 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

device = "npu"
model = SimpleModel().to(device)
model = model.half()

# 2. 创建配置
config = FloatSparseProcessorConfig(
    sparse_ratio=0.3,           # 30%稀疏率
    include=["*"],              # 处理所有模块
    exclude=[]                  # 不排除任何模块
)

# 3. 创建处理器
processor = FloatSparseProcessor(model, config)

# 4. 预处理：收集统计信息
calibration_data = torch.randn(1, 784).to(device)  # 校准数据
calibration_data = calibration_data.half()
for batch in calibration_data:
    batch = batch.unsqueeze(0)
    # 预处理：安装hook并运行前向传播
    request = BatchProcessRequest(
        name="", 
        module=model, 
        datas=[((batch,), {})],  # 正确的数据格式：(args, kwargs)
        outputs=None
    )
    processor.preprocess(request)
    processor.process(request)
    processor.postprocess(request)

print("稀疏化完成！")
```

## 算法参数

浮点稀疏算法内部使用以下参数（可通过修改源码调整）：

```python
# ADMM参数
KEEP_BITS = 2                    # 保持精度的位数
KEEP_PROPORTION = 0.02          # 保持精度的比例：2%
PERCDAMP = 0.1                  # 阻尼系数
ITERATIVE_PRUNE = 15            # 迭代稀疏次数
ITERS = 20                      # ADMM最大迭代次数
```

## 稀疏配置参数

```python
FloatSparseProcessorConfig(
    sparse_ratio=0.3,           # 稀疏比例：0.0-1.0，越大越稀疏
    include=["*"],              # 包含的模块名称模式
    exclude=[]                  # 排除的模块名称模式
)
```

## 适用要求

- **高压缩需求**：适用于需要高压缩率的模型部署场景。
- **精度敏感**：通过精度保护机制，在压缩的同时保持关键权重精度。
- **计算成本**：ADMM算法需要多次迭代，计算成本较高，速度较慢。
- **内存需求**：需要存储Hessian矩阵和激活统计信息，显存占用较高。
- **使用限制**：
  - 当前算法只有结合 Atlas 300I Duo 推理卡才能够获取性能提升，Atlas 800I A2 无法获得性能提升。
  - 由于 Atlas 300I Duo 推理卡 不支持 bfloat 数据类型，因此对模型进行浮点稀疏时，需要手动将模型路径下的 config.json 中的 `torch_dtype` 字段修改成 float16。
  - 仅支持 `v1 框架` 中的 `逐层量化`。
  - 目前仅支持 `nn.Linear` 模块进行浮点稀疏。
  - 需要校准数据集收集激活统计信息，校准数据的 token id 个数 >= 2048。
  - 稀疏比例建议在 `0.3` 附近逐步调整。

## <span id="一键量化使用">一键量化使用</span>

### 步骤1：校准集生成

为了获得精度较好的浮点稀疏模型，校准集需要满足以下要求：

- **数据量**：包含多条代表性数据样本
- **数据长度**：每条数据经过tokenizer编码后的token数量 ≥ 1024
- **数据质量**：选择与目标任务相关的数据，确保覆盖模型的主要使用场景

#### 基于 aisbench 的校准集生成方法

以下提供基于 aisbench 工具的校准集生成流程，以 GPQA 数据集为例：

#### 1：模型配置修改
- 修改模型权重目录下的 config.json 文件，将 torch_dtype 字段修改为 "float16"

#### 2：启动服务化推理
拉起模型服务化，准备接收推理请求

#### 3：数据采集

```bash
# 使用 aisbench 采集指定数据集，添加 --dump-eval-details 参数获取详细评估结果
aisbench --models vllm_api_general_chat --datasets gpqa_gen --dump-eval-details
```

#### 4：校准集生成

使用以下 Python 脚本处理 aisbench 采集的数据，生成符合要求的校准集：

```python
import json
from transformers import AutoTokenizer

def process_input(tokenizer, data_name, json_path, select_num=5, final_count=0, max_seq_length=4096):
    """
    处理单个数据集，提取符合条件的数据样本
    
    Args:
        tokenizer: 模型对应的tokenizer
        data_name: 数据集名称
        json_path: aisbench生成的结果文件路径
        select_num: 选择的数据条数
        final_count: 全局计数
        max_seq_length: 最大序列长度
    
    Returns:
        list: 处理后的数据列表
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    count = 0

    for key, value in data['details'].items():
        if count >= select_num:
            break
        if value['correct']:  # 只选择正确答案的样本
            combined_text = f"{value['prompt']}\n{value['origin_prediction']}"

            inputs = tokenizer(
                combined_text,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )

            token_count = inputs['input_ids'].shape[1]

            if token_count > max_seq_length:
                truncated_ids = inputs['input_ids'][0, :max_seq_length]
                truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            else:
                truncated_text = combined_text

            results.append({
                "id": final_count,
                'inputs_pretokenized': truncated_text,
                'token_count': token_count
            })

            count += 1
            final_count += 1
            print(f"Processed {count} out of {select_num} data")

    return results

# 配置数据集信息
dataset = {
    "GPQA": {
        "path": "aisbench生成的results中的GPQA_diamond.json",
        "select_num": 10
    }
}

# 模型路径配置
model_path = "模型权重路径"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False,
    legacy=False,
    trust_remote_code=True
)

# 生成校准集
final_count = 0
final_results = []
for key, value in dataset.items():
    print(f"Processing {key} dataset")
    processed_data = process_input(tokenizer, key, value['path'], value['select_num'], final_count)
    final_results.extend(processed_data)

# 保存校准集
output_path = "输出校准集路径.jsonl"
with open(output_path, 'w') as f:
    for result in final_results:
        f.write(json.dumps(result) + '\n')
```

### 步骤2：创建 YAML 配置文件

创建浮点稀疏量化配置文件 `float_sparse_config.yaml`：

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "float_sparse"
      sparse_ratio: 0.25          # 稀疏比例，建议在0.3附近调整
      include: ["*"]              # 处理所有模块
      exclude: []                 # 不排除任何模块
  save:
    - type: "ascendv1_saver"
      part_file_size: 4           # 分片文件大小（GB）
  dataset: "校准集文件路径.jsonl"  # 替换为实际的校准集路径
```

### 步骤3：执行浮点稀疏量化

使用以下命令执行浮点稀疏量化：

```bash
msmodelslim quant \
  --model_path {浮点权重路径} \
  --save_path {W16A16S保存路径} \
  --device npu \
  --model_type {模型类型} \
  --trust_remote_code True \
  --config_path {YAML配置文件路径}
```

## 性能特点

### 优势

1. **高压缩率**：通过ADMM算法实现高稀疏率，压缩效果显著。
2. **精度保护**：智能识别重要权重位置，避免关键信息丢失。
3. **自适应优化**：基于激活统计信息自行调整稀疏策略。
4. **逐层量化**：支持逐层量化，降低内存占用。

### 局限性

1. **计算开销**：ADMM迭代和Hessian矩阵计算增加模型稀疏时间。
2. **显存占用**：需要存储额外的统计信息和中间结果。
3. **参数调优**：稀疏比例等参数需要根据具体模型调整。

## 常见问题排查

### 1. 执行模型压缩时报错

**现象**：执行模型压缩时报错：ValueError：quant_model_json_description must have model quant type

**解决方案**：将生成的浮点稀疏权重中的 quant_model_description.json 文件中添加 model_quant_type 字段，值为 W16A16S。

### 2. 浮点稀疏不支持叠加 w8a8 稀疏量化

**现象**：用户尝试对已经进行W8A8S（权重INT8稀疏量化）处理的模型，再应用浮点稀疏算法进行进一步稀疏化。

**解决方案**：浮点稀疏算法（W16A16S）和W8A8S稀疏量化是两种不同的技术路径，不支持叠加使用。多次稀疏化处理会累积精度损失，可能严重影响模型性能。

### 3. 稀疏比例设置过高

**现象**：稀疏比例过高导致模型精度严重下降。

**解决方案**：降低 sparse_ratio 配置参数，建议在 0.3 附近逐步调整。

### 4. 校准数据长度不够，导致求矩阵逆失败

**现象**：处理大模型时出现求矩阵逆失败错误。

**解决方案**：增加校准集中每条数据长度，保证经过 tokenizer 编码后的 token id 数量 >= 2048。

### 5. 校准集数量过多导致显存溢出

**现象**：处理大模型时出现显存溢出错误。

**解决方案**：减少校准集数量，或使用单卡显存更大的机器进行浮点稀疏。
