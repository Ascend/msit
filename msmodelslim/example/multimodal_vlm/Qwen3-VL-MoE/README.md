# Qwen3-VL-MoE 量化使用说明

## 模型介绍

Qwen3-VL-MoE 是阿里云 Qwen 团队推出的大规模多模态视觉语言 Mixture-of-Experts (MoE) 模型，具备以下特点：

- **稀疏 MoE 架构**: 采用稀疏激活的 MoE 结构，在保持高性能的同时显著降低计算成本
- **多模态理解能力**: 支持图像和文本的联合理解，可执行图像描述、视觉问答等多种任务
- **大规模参数**: 提供 30B-A3B 和 235B-A22B 两种规格，其中 "A" 代表激活参数量
- **3D 融合专家权重**: 专家层权重以 3D 张量形式融合存储，需要特殊的量化处理

## 环境配置

- 基础环境配置请参考[安装指南](../../../docs/安装指南.md)，注意：由于高版本transformers的特殊性，PyTorch及torch_npu需要配置安装为2.7.1版本
- 还需要安装 flax 依赖：
  ```bash
  pip install flax
  ```
- 针对 Qwen3-VL-MoE，transformers 版本需要 4.57.1：
  ```bash
  pip install transformers==4.57.1
  ```

## Qwen3-VL-MoE 模型当前已验证的量化方法

| 模型 | 原始浮点权重 | 量化方式 | 推理框架支持情况 | 量化命令 |
|------|-------------|---------|----------------|---------|
| Qwen3-VL-235B-A22B | [Qwen3-VL-235B-A22B](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/tree/main) | W8A8 混合量化 | MindIE 待支持<br>vLLM Ascend 支持中 | [W8A8 混合量化](#qwen3-vl-moe-w8a8-混合量化) |

注：[Qwen3-VL-30B-A3B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main) 尚未验证过量化精度，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。

**说明：**
- 点击量化命令列中的链接可跳转到对应的具体量化命令
- W8A8 混合量化：Attention 和常规 MLP 层使用静态量化，MoE experts 使用动态量化

## 量化特性

### MoE 专家层自动转换
- **3D 权重拆分**: 自动将融合的 3D 专家权重 `(num_experts, hidden_size, expert_dim)` 拆分为独立的 `nn.Linear` 层
- **逐层处理**: 结合 v1 框架的逐层加载机制，在加载每一层时自动完成 MoE 转换
- **内存友好**: 转换过程采用 in-place 策略，及时释放原始 3D 权重，大幅降低内存占用

### 异常值抑制 (Iterative Smooth)
- **iter_smooth 算法**: 使用迭代平滑算法抑制激活值异常点，提升量化精度
- **多种子图类型**: 支持 norm-linear、linear-linear、ov、up-down 等多种子图融合
- **自适应配置**: 自动识别 MoE 层结构，为不同层类型应用合适的平滑策略

### 混合量化策略
- **Attention 层**: W8A8 静态量化 (激活per_tensor)，适合激活分布稳定的层
- **MoE Experts**: W8A8 动态量化 (激活per_token)，适应不同 token 的激活差异，保持精度
- **语言部分 MLP Gate 层**: 不进行量化，保持浮点精度，确保专家路由准确性
- **视觉部分 linear_fc2 层**: 精度敏感，不进行量化，保持浮点精度
- **视觉部分 merger、deepstack 层**: 精度敏感，不进行量化，保持浮点精度



### 逐层量化
- **内存优化**: 支持逐层加载、量化、offload 的流程，显著降低显存占用
- **单卡支持**: 结合逐层量化特性，可在 Atlas 800I A2 (64G) 设备上完成大模型量化

## 生成量化权重

### 使用一键量化命令

Qwen3-VL-MoE 使用 msmodelslim 的一键量化接口，无需编写量化脚本。

#### <span id="qwen3-vl-moe-w8a8-混合量化">Qwen3-VL-MoE W8A8 混合量化</span>

**方式一：使用 quant_type 参数进行一键量化，以Qwen3-VL-235B-A22B为例**

```bash
msmodelslim quant \
    --model_path /path/to/qwen3_vl_moe_float_weights \
    --save_path /path/to/qwen3_vl_moe_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-235B-A22B \
    --quant_type w8a8 \
    --trust_remote_code True
```

**方式二：使用 config_path 参数指定配置文件进行一键量化，以Qwen3-VL-235B-A22B为例**

```bash
msmodelslim quant \
    --model_path /path/to/qwen3_vl_moe_float_weights \
    --save_path /path/to/qwen3_vl_moe_quantized_weights \
    --device npu \
    --model_type Qwen3-VL-235B-A22B \
    --config_path /path/to/qwen3_vl_moe_w8a8.yaml \
    --trust_remote_code True
```

### 一键量化命令参数说明

一键量化参数基本说明可参考：[一键量化参数说明](../../../docs/功能指南/一键量化/使用说明.md#接口说明)

针对 Qwen3-VL-MoE 模型，参数配置要求如下：

| 参数名称 | 解释 | 是否可选 | 范围 |
|---------|------|---------|------|
| model_path | Qwen3-VL-MoE 浮点权重目录 | 必选 | 类型：Str |
| save_path | Qwen3-VL-MoE 量化权重保存路径 | 必选 | 类型：Str |
| device | 量化设备 | 必选 | 1. 类型：Str <br>2. 仅支持 "npu" |
| model_type | 模型名称 | 必选 | 1. 类型：Str <br>2. 大小写敏感，可选值："Qwen3-VL-30B-A3B", "Qwen3-VL-235B-A22B"。[Qwen3-VL-30B-A3B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main) 尚未验证过量化精度，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。 |
| config_path | 指定配置路径 | 与 "quant_type" 二选一 | 1. 类型：Str <br>2. 配置文件格式为 yaml <br>3. 当前只支持最佳实践库中已验证的配置 [qwen3_vl_moe_w8a8.yaml](../../../lab_practice/qwen3_vl_moe/qwen3_vl_moe_w8a8.yaml)，若自定义配置，msmodelslim 不为量化结果负责 |
| quant_type | 量化类型 | 与 "config_path" 二选一 | 1. 类型：Str <br>2. 当前仅支持配置为 "w8a8" |
| trust_remote_code | 是否信任自定义代码 | 可选 | 1. 类型：Bool，默认值：False <br>2. 指定 `trust_remote_code=True` 让修改后的自定义代码文件能够正确地被加载（请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险） |

## 配置文件说明

### 基础配置结构

```yaml
apiversion: multimodal_vlm_modelslim_v1
metadata:
  config_id: qwen3_vl_moe_w8a8
  score: 90
  verified_model_types:
    - Qwen3-VL-235B-A22B
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: False
    kv_cache: False

default_w8a8_dynamic: &default_w8a8_dynamic
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: True
    method: "minmax"

default_w8a8: &default_w8a8
  act:
    scope: "per_tensor"
    dtype: "int8"
    symmetric: False
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: True
    method: "minmax"

spec:
  process:
    - type: "iter_smooth"                  
      alpha: 0.9  # 浮点数, > 0, 默认 0.9，平衡参数，控制激活和权重的相对重要性。
      scale_min: 1e-5  # 浮点数, > 0, 默认 1e-5，缩放因子的下界，防止数值过小导致数值不稳定。
      symmetric: True  # 使用is_shift=True时，应该将symmetric设置为False
      enable_subgraph_type:
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include:                             
        - "*"
    - type: "linear_quant"
      qconfig: *default_w8a8
      include:
        - "*"
      exclude:
        - "*experts*"  # Exclude MoE experts for dynamic quantization
        - "*linear_fc2"
        - "*merger*"
        - "*deepstack_merger_list*"
        - "*mlp.gate"
    - type: "linear_quant"
      qconfig: *default_w8a8_dynamic
      include:
        - "*experts*"  # MoE experts use dynamic quantization
      exclude:
        - "*linear_fc2"
        - "*merger*"
        - "*deepstack_merger_list*"
        - "*mlp.gate"
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
  dataset: "calibImages"  # Short name: auto-searches in lab_calib/
```

### 关键配置参数

此处只说明关键配置参数，更多参数说明可见：
- [Iterative Smooth 算法配置字段说明](../../../docs/算法说明/Iterative_Smooth.md#yaml配置字段详解)
- [LinearQuantProcess 线性层量化处理器配置字段说明](../../../docs/功能指南/一键量化/features/linear_quant.md#yaml配置字段详解)
- [保存处理器配置字段说明](../../../docs/功能指南/一键量化/配置协议说明.md#保存器配置字段-save)


#### 异常值抑制配置 (iter_smooth)

- **type**: 处理器类型，固定为 "iter_smooth"
- **alpha**: 平滑强度系数，范围 [0, 1]，默认 0.9
- **scale_min**: 最小缩放因子，防止数值不稳定，推荐 1e-5
- **symmetric**: 是否对称量化，True为对称，False为非对称，默认True
- **enable_subgraph_type**: 启用的子图类型
  - `norm-linear`: LayerNorm → Linear 结构
  - `linear-linear`: Linear → Linear 结构
  - `ov`: Attention 中的 V → O 投影
  - `up-down`: MLP 中的 Up → Down 结构
- **include**: 包含的层模式，支持通配符匹配
- **exclude**: 排除的层模式, 支持通配符匹配

#### 量化配置 (linear_quant)

**静态量化配置** (`default_w8a8`):
- **act.scope**: "per_tensor"，所有 token 共享一个量化 scale
- **act.symmetric**: False，允许非对称量化，适应偏移分布
- 适用于：Attention 层、常规 MLP 层等

**动态量化配置** (`default_w8a8_dynamic`):
- **act.scope**: "per_token"，每个 token 独立计算量化 scale
- **act.symmetric**: True，对称量化，适合动态范围变化大的场景
- 适用于：MoE experts 层

#### 保存配置 (save)

- **type**: 保存器类型，使用 "ascendv1_saver" 格式
- **part_file_size**: 分片文件大小（GB），推荐 4GB 避免单个文件过大

#### 校准数据集 (dataset)

- **短名称自动解析**: 配置为 "calibImages"，自动解析为 `lab_calib/calibImages/`
- **完整路径**: 也可配置为绝对路径或相对路径
- **数据来源**: 推荐从 [COCO](https://cocodataset.org/#download) 或 [textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset) 数据集中选取 20-30 张图片作为校准集

## 常见问题

### Q1: 为什么 MoE experts 使用动态量化？

**A**: MoE experts 的激活分布在不同 token 间差异较大：
- **静态量化** (per_tensor): 所有 token 共享一个 scale → 精度损失大
- **动态量化** (per_token): 每个 token 独立 scale → 精度更高

这是 MoE 模型的标准做法，参考 DeepSeek-V3 等模型的最佳实践。

### Q2: 如何自定义校准数据集？

**A**: 有两种方式：
1. 在配置文件中修改 `dataset` 字段为自定义路径
2. 将图片放到 `lab_calib/calibImages/` 目录，使用短名称 "calibImages"

建议使用与实际应用场景相似的图片作为校准集，数量 20-30 张即可。

## 相关资源

- [一键量化配置协议说明](../../../docs/功能指南/一键量化/配置协议说明.md)
- [逐层量化特性说明](../../../docs/功能指南/一键量化/features/layer_wise_quantization.md)
- [Iterative Smooth 算法说明](../../../docs/算法说明/Iterative_Smooth.md)
- [LinearQuantProcess 线性层量化处理器说明](../../../docs/功能指南/一键量化/features/linear_quant.md)

