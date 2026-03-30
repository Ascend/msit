# MindStudio 预检工具

## 简介

msprechecker (MindStudio Prechecker Tool)：MindStudio 预检工具。这是一个帮助用户在昇腾（Ascend）环境中快速部署 AI 推理服务、复现性能基线、定位部署与性能问题的工具。工具提供预检（precheck）、环境信息落盘（dump）和差异比对（compare）三大核心功能，**并于 2025.12.16 版本起**，新增支持基于可扩展规则引擎的 `run` 和 `inspect` 子命令，允许用户自定义和分享复杂的检查规则。

**基本概念**

- PD混部（Prefill-Decode混合部署）：一种部署模式，将模型的 Prefill 阶段​ 和 Decode 阶段​ 部署在同一个计算实例（Pod/容器）​ 中。
- PD分离（Prefill-Decode分离部署）：一种部署模式，将 Prefill 阶段​ 和 Decode 阶段​ 解耦，部署在不同的、可独立伸缩的计算实例或Pod上。
- Rank Table：一种描述昇腾 NPU 芯片在多机多卡环境中拓扑连接关系的配置文件，用于指导分布式训练或推理的任务启动和通信。
- cmate 文件：一种基于特定语法定义的规则文件，用于描述对系统、环境变量、配置文件等对象的检查和断言逻辑。工具内置了 MindIE 和 VLLM-Ascend 框架的常用规则集。

**工具使用流程**

用户通常首先使用 `precheck` 功能在部署前对目标环境进行全面检查；在服务运行（或出现问题）时，使用 `dump` 功能保存环境快照；当需要对比不同环境（如基线环境与问题环境）的差异时，使用 `compare` 功能对多个 dump 文件进行分析。对于有定制化检查需求的用户，可以使用 `inspect` 查看规则文件，并使用 `run` 执行自定义的规则检查。

## 使用前准备

**环境准备**

1. 准备一台安装有昇腾（Ascend）NPU 的服务器，并确保已正确安装 NPU 驱动、固件及 CANN 软件包。
2. 确保 Python 版本 >= 3.7。
3. 安装 msprechecker 工具及其依赖。

    您可以通过以下任一方式安装：
    * **PyPI 安装（推荐）**

        ```bash
        pip install msprechecker
        ```

    * **离线安装**
        1. 从可联网的机器访问 <https://pypi.org/project/msprechecker/#files> 下载 wheel 包。
        2. 将下载的 wheel 包上传至目标服务器。
        3. 执行安装命令（将 `whl_path` 替换为实际路径）：

            ```bash
            pip install whl_path
            ```

    * **源码安装**

        ```bash
        git clone https://gitcode.com/Ascend/msit.git
        pip install -e msit/msprechecker
        ```

4. 安装必要的第三方依赖：`psutil`, `pyyaml`, `importlib_metadata`，`ply`。通常这些会在安装 msprechecker 时自动解决。

**约束**

* 非 root 用户安装前建议执行 `umask 0027`，以避免后续使用中的权限问题。
* 当前工具主要支持 Atlas 800I A2, Atlas 800I A3 和 Atlas 9000 A2 (G8600) 等训练服务器。
* 支持 MindIE 和 VLLM-Ascend (v0.9.1-dev) 推理框架的校验。
* `dump` 功能暂不支持对多机 PD 分离、单机 PD 分离场景的配置文件进行落盘。

## 快速入门

本节以最常见的 **MindIE 框架 PD 混部场景** 为例，演示如何使用 msprechecker 进行部署前预检。

**前提条件**

1. 已完成环境准备，成功安装 msprechecker。
2. 已获取 MindIE 服务的配置文件 `config.json`（通常路径为 `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`）。
3. 确保待部署的模型权重目录可访问。

> [!Warning]
> 注意：MindIE 已于 2026 年 3 月全面开源。在此之后，其配置文件位置可能已发生变化。若在上述默认路径未找到，您可以通过环境变量 `MINDIE_LLM_HOME_PATH` 定位，配置文件的完整路径通常为 `$MINDIE_LLM_HOME_PATH/conf/config.json`。

**操作步骤**

1. 执行预检命令，指定 MindIE 服务配置文件和模型权重目录。

    ```bash
    msprechecker precheck --mies-config-path /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json --weight-dir /path/to/your/model_weights
    ```

2. 工具将运行一系列检查，包括系统配置、环境变量、NPU状态、网络连通性、模型配置合规性等。
3. 检查结果将在终端输出，并以不同等级标识：
    * `[NOK]`: 严重问题，可能导致部署失败或性能严重下降，必须修复。
    * `[WARNING]`: 潜在问题或非最优配置，建议修复。
    * `[RECOMMEND]`: 优化建议，修复后可获得更好性能。
4. 如果发现环境变量配置问题，工具会在当前目录生成 `msprechecker_env.sh` 文件，您可查看并决定是否采纳其中的建议。
5. 根据工具输出的建议，逐一修复标识出的问题，直至所有检查通过或仅有可接受的 `[RECOMMEND]` 项，即可开始部署。

## 命令行工具指导

msprechecker 包含五个子命令：`precheck`, `dump`, `compare`, `run`, `inspect`。

### 产品支持情况

>![](./pics/icon-note.gif) **说明：**
>昇腾产品的具体型号，请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》。

| 产品类型 | 是否支持 |
| ------------ | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品  |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品  |    √     |
| Atlas 200I/500 A2 推理产品  |    ×    |
| Atlas 推理系列产品  |    √    |
| Atlas 训练系列产品   |    ×    |

### 功能说明

`msprechecker` 工具用于对昇腾 AI 推理部署环境进行健康检查、信息收集和差异比对，旨在提升部署成功率和问题定位效率。新版本引入了基于 `cmate` 规则文件的可扩展检查引擎，为用户提供了更灵活、强大的自定义检查能力。

### 注意事项

- 多机 PD 混部场景需要在**每台**目标机器上分别执行预检命令，工具不支持从单点控制多机。
- 使用 `--rank-table-path` 进行网络检查时，工具默认按 MindIE 格式解析 rank table 文件。若用于 VLLM-Ascend 框架，请务必通过 `--scene vllm` 参数指定。
- 部分环境变量（如 `RANK_TABLE_FILE`）的正确性需要用户结合自身部署规划手动确认。

### 命令格式

```shell
msprechecker <subcommand> [options]
```

**子命令（任选其一）**

* `precheck`: 执行部署前环境预检。
* `dump`: 将当前环境信息（系统、环境变量、配置等）保存到文件。
* `compare`: 比较两个或多个由 `dump` 命令生成的文件，找出差异。
* `run`: 执行用户指定的 `cmate` 规则文件进行检查。
* `inspect`: 查看 `cmate` 规则文件的元信息和规则内容。

### 参数说明

#### `precheck` 子命令主要参数

| 参数 | 可选/必选 | 说明 |
| :--- | :--- | :--- |
| `--scene` | 可选 | 指定部署场景。格式：`<framework>,<deploy-mode>[,<npu_type>,<npu_count>,<arch>,<model_type>]`。例如：`mindie,pd_mix` 或 `vllm,ep,A2,8,arm`。用于辅助工具识别框架、部署模式、硬件等信息。默认值：`None` |
| `--mies-config-path` | 可选 (MindIE PD混部) | MindIE PD混部场景的配置文件路径，通常为 `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`。默认值：`None` |
| `--config-parent-dir` | 可选 (PD分离) | PD分离场景下，包含所有需修改的 `conf/*.json` 和 `deployments/*.yaml` 配置文件的父目录路径（通常名为 `kubernetes_deploy_scripts`）。需与 `--scene pd_disaggregation` 或 `--scene pd_disaggregation_single_container` 联用。默认值：`None` |
| `--user-config-path` | 可选 (大EP场景) | 大 EP 场景下的 `user_config.json` 文件路径。默认值：`None` |
| `--mindie-env-path` | 可选 (大EP场景) | 大 EP 场景下的 `mindie_env.json` 文件路径。默认值：`None` |
| `--rank-table-path` | 可选 | 指定 rank table 文件路径，用于触发 NPU 间网络连通性测试及相关配置检查。默认值：`None` |
| `--weight-dir` | 可选 | 指定模型权重目录路径，用于检查该目录下的 `config.json` 配置文件。默认值：`None` |
| `--hardware` | 可选 | 启用 CPU 和 NPU 硬件压测，检测是否存在算力异常的核或卡。默认值：`None` |
| `--threshold` | 可选 | 硬件压测的筛选阈值，取值范围 `[0-100]` 闭区间。单位为百分比。当某个核/卡的计算时间超过平均值的比例大于此阈值时，被标记为异常。默认值：20 |
| `--custom-config-path` | 可选 | 用户自定义检查规则文件的路径 (YAML 格式)。默认值：`None` |
| `-l`, `--severity-level` | 可选 | 控制输出信息的严重级别。可选值：`low`(显示所有)、`medium`(显示WARNING和NOK)、`high`(仅显示NOK)。默认值：`low` |

#### `dump` 子命令主要参数

| 参数 | 可选/必选 | 说明 |
| :--- | :--- | :--- |
| `--output-path` | 可选 | 文件路径，指定落盘数据文件的保存路径。默认值：`./msprechecker_dumped.json`。 |
| `--filter` | 可选 | 若启用，则只落盘与昇腾研发相关的环境变量，过滤无关变量。默认值：`False` |
| `--user-config-path` | 可选 | 文件路径，额外落盘指定的 `user_config.json` 文件内容。默认值：`None` |
| `--mindie-env-path` | 可选 | 文件路径，额外落盘指定的 `mindie_env.json` 文件内容。默认值：`None` |
| `--mies-config-path` | 可选 | 文件路径，额外落盘指定的 MindIE 服务 `config.json` 文件内容。默认值：`None` |
| `--rank-table-path` | 可选 | 文件路径，额外落盘指定的 rank table 文件内容。默认值：`None` |
| `--weight-dir` | 可选 | 目录路径，额外落盘指定权重目录下的 `config.json` 和所有权重文件的 SHA256 哈希值。默认值：`None` |
| `--chunk-size` | 可选 | 计算权重文件哈希时，每次读取的数据块大小 (MB)。可选值：32, 64, 128, 256。默认值：`32`。 |

#### `compare` 子命令参数

`compare` 子命令接受一个或多个文件路径作为位置参数，用于比较这些文件内容的差异。

#### `run` 子命令主要参数

| 参数                      | 可选/必选 | 说明                                                                                                                         |
|:------------------------| :--- |:---------------------------------------------------------------------------------------------------------------------------|
| `rule`                  | **必选** (位置参数) | 要执行的 `cmate` 规则文件的路径。                                                                                                      |
| `-C`, `--contexts`      | 可选 | 传入规则所需的上下文变量。语法：`-C 变量名:变量值`。例如：`-C npu_count:2` 传入整数 2，`-C model_name:"deepseek"` 传入字符串 "deepseek"。可多次使用传入多个变量。默认值：`None` |
| `-c`, `--configs`       | 可选 | 传入规则所需的配置文件路径。语法：`-c 规则中的配置变量名:实际文件路径`。可指定解析类型：`配置变量名:文件路径@解析类型` (如 `json`, `yaml`)。不指定则用规则定义或文件后缀。可多次使用。默认值：`None`        |
| `-co`, `--collect-only` | 可选 | 仅收集要执行的规则项，但不实际执行。类似于测试框架的 `--collect-only`。                                                                               |
| `--output-path`         | 可选 | 预检结果的保存路径，保存文件的格式为`msprechecker_{timestamp}_output.json`。默认值：`''`（表示直接打印预检结果）                                              |
| `-x`, `--fail-fast`     | 可选 | 遇到第一个规则检查失败后立即停止执行。默认值：`False`                                                                                             |
| `-v`, `--verbose`       | 可选 | 详细输出模式，显示每个规则在 cmate 文件中的行号和具体校验内容。默认值：`False`                                                                             |
| `-s`, `--severity`      | 可选 | 运行规则的最小严重级别。可选值：`info` (运行全部)、`warning` (不运行 info 级别)、`error` (仅运行 error 级别)。默认值：`info`                                    |

#### `inspect` 子命令参数

| 参数 | 可选/必选 | 说明 |
| :--- | :--- | :--- |
| `rule` | **必选** (位置参数) | 要查看的 `cmate` 规则文件的路径。 |
| `-f`, `--format` | 可选 | 输出格式。可选值：`text` (文本)，`json` (JSON格式)。默认值：`text` |

### 使用示例

**示例1：VLLM-Ascend 通用场景预检**
检查 VLLM-Ascend 框架部署的基本环境。

```bash
msprechecker precheck --scene vllm
```

**示例2：MindIE 大 EP 场景预检**
检查大 EP (Elastic Processing) 部署场景的配置。

```bash
msprechecker precheck --scene mindie,ep --user-config-path /path/to/user_config.json --mindie-env-path /path/to/mindie_env.json
```

**示例3：执行 cmate 规则文件**
运行内置的 MindIE 规则文件，并传入必要的配置文件和上下文变量。

```bash
msprechecker run /path/to/msprechecker/preset/mindie.cmate \
  -c mies_config:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
  -C deploy_mode:pd_mix model_type:deepseek npu_type:A2
```

**示例4：查看规则文件信息**
以文本格式查看规则文件的概览、上下文需求和配置定义。

```bash
msprechecker inspect /path/to/msprechecker/preset/mindie.cmate
```

**示例5：环境信息落盘**
将当前完整的系统、环境变量、昇腾相关配置等信息保存到指定文件。

```bash
msprechecker dump --output-path /tmp/env_baseline.json --weight-dir /path/to/model_weights
```

**示例6：环境差异比对**
比较两个不同时间点或不同机器上保存的环境信息文件。

```bash
msprechecker compare /tmp/baseline.json /tmp/problem_env.json
```

### 输出说明

`precheck` 命令执行后，会在终端输出详细的检查报告。报告按检查类别分组，每条结果前有严重性标记 (`[NOK]`, `[WARNING]`, `[RECOMMEND]`)，并附带问题描述和建议。

`run` 命令执行后，输出格式类似于测试框架，会显示规则收集进度、执行结果摘要以及详细的失败断言信息，包括期望值（`>` 标记）、实际值（`E` 标记）和错误等级、原因。

`dump` 命令执行后，会输出落盘文件的保存路径。

`compare` 命令执行后，会以清晰的 JSON 格式输出不同文件之间的差异。如果没有差异，则提示 `There is no difference found.`。

`inspect` 命令执行后，会根据 `--format` 参数输出规则文件的元信息、上下文变量、配置定义等。

## 扩展功能

### 自定义检查项配置 (YAML)

您可以通过 YAML 文件定义自定义的检查规则，并使用 `--custom-config-path` 参数传递给 `precheck` 命令。这允许您对特定环境变量或配置项设置期望值和检查逻辑。

**规则文件示例 (`custom_rules.yaml`):**

```yaml
MY_CUSTOM_ENV:
  expected:
    type: eq
    value: "expected_value"
  reason: "自定义环境变量应设置为特定值。"
  severity: high
```

目前支持 `eq`/`==`, `lt`/`<`, `le`/`<=`, `gt`/`>`, `ge`/`>=`, `ne`/`!=` 等比较类型。`value` 字段支持四则运算和字段引用（见下文）。

### 字段引用语法

在自定义检查规则中，可以使用 `${}` 语法引用配置文件中其他字段的值，用于定义复杂的关联性检查。

**示例:**
假设配置文件中有 `{"a": {"b": 10, "c": 20}}`，您可以创建如下规则来检查 `a.b + a.c == 30`：

```yaml
a:
  b:
    expected:
      type: eq
      value: 30 - ${.c}  # 相对引用 a.c
    reason: "a.b 应等于 30 减去 a.c 的值。"
  c:
    expected:
      type: eq
      value: 30 - ${a.b} # 绝对引用 a.b
    reason: "a.c 应等于 30 减去 a.b 的值。"
```

### 自定义规则引擎 (cmate 文件)

自 2025.12.16 版本起，msprechecker 引入了全新的、功能更强大的规则引擎，通过 `run` 和 `inspect` 子命令进行操作。规则通过 `cmate` 格式的文件定义，支持更复杂的断言逻辑、条件分支、模块化组织和丰富的上下文控制。

* **内置规则**: 工具已将 MindIE 和 VLLM-Ascend 框架的常用检查规则内置在 `msprechecker/preset/` 目录下的 `mindie.cmate` 和 `vllm.cmate` 文件中，用户可以直接使用或作为编写参考。
* **使用流程**:
    1. **查看规则**：使用 `msprechecker inspect <rule.cmate>` 查看规则文件所需的上下文变量、配置文件和规则描述。
    2. **执行规则**：使用 `msprechecker run <rule.cmate>` 并配合 `-C` (传入上下文) 和 `-c` (传入配置文件) 参数来执行检查。
* **优势**：
    * **可扩展**：用户可以基于 `cmate` 语法自行开发新的规则集，并在团队或社区内分享。
    * **灵活**：规则文件支持定义多种数据源（环境变量、配置文件、系统状态）的检查，并可通过上下文变量动态控制检查逻辑。
    * **结构化输出**：`run` 命令提供结构清晰、类似单元测试的输出，便于集成到 CI/CD 流程中。

关于 `cmate` 文件的详细语法和编写指南，请参见工具源码或相关开发文档。

## 附录

### FAQ

1. **Q: 如何为 VLLM-Ascend 框架指定 rank table 进行检查？**
  
    A: 在执行命令时，必须通过 `--scene vllm` 明确指定框架，工具才会使用正确的格式解析 rank table。例如：`msprechecker precheck --scene vllm --rank-table-path /path/to/vllm_hccl.json`

2. **Q: `dump` 时出现的 `[WARNING]` 是什么意思？**
  
    A: 这通常是因为在落盘某项信息（如某个配置文件）时，因路径未提供或文件不存在而跳过。这不影响其他信息的落盘，最终生成的 `.json` 文件仍是有效的，只是缺少对应部分的数据。

3. **Q: 多机场景下，需要在每台机器上都运行预检吗？**
   
    A: 是的。对于 PD 混部多机场景，需要在**每个参与计算的服务器节点**上分别运行预检命令，因为每台机器的环境配置可能不同。

4. **Q: 新版本中的 `run` 子命令和原来的 `precheck` 命令有什么区别？**
   
    A: `precheck` 是封装好的、针对特定场景的固定检查流程，开箱即用。`run` 命令则提供了一个通用的、可编程的规则执行引擎，允许用户通过 `cmate` 文件定义和执行任意复杂的检查逻辑，灵活性极高，适合有定制化检查需求或希望分享检查规则的场景。工具内置的 `precheck` 功能在底层也正在逐步迁移到新的规则引擎上。
