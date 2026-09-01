# Expert Load Balancing Tool

## Introduction

The Expert Load Balancing Tool performs load balancing affinity expert optimization in static and dynamic scenarios of MoE models. By loading the expert popularity information file (in csv or pt format) dumped during model inference, the tool computes and saves the configuration file for the prefill or decoder stage after algorithm calculation.

Currently, algorithms are provided for the following scenarios:

- Static scenario + Atlas 800I A2 inference server: Compute Communication Load Balance (C2LB) algorithm, speculative-moe level 1 algorithm, speculative-moe level 2 algorithm, speculative-moe level 1 mixed placement algorithm, and speculative-moe level 2 mixed placement algorithm.
- Static scenario + Atlas 800I A3 inference server: speculative-moe level 1 and speculative-moe level 2 algorithms.
- Dynamic scenario + Atlas 800I A2 inference server: C2LB algorithm to generate the initial configuration file.

**Preparation Before Use**

### Environment Preparation

1. Prepare an inference server based on the Ascend NPU.
2. Install Python 3.9 or later.

### Tool Installation

#### Installation Prerequisites

- Install the Ascend AI inference-related driver, firmware, and CANN package for the development and running environment. Refer to [CANN](https://www.hiascend.com/cann/download).
- After installation, you can specify the installed CANN version path by **setting the CANN_PATH environment variable**, for example: export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest.
- If not set, the tool attempts to obtain the CANN version from the environment variable ASCEND_TOOLKIT_HOME and the path /usr/local/Ascend/ascend-toolkit/latest respectively.

#### Installation Methods

The installation methods include: **source code installation** and **pip source installation**. You can choose as needed.

- [Source code installation](#source-code-installation): Install from source code to ensure the latest msit functionality.
- [pip source installation](#pip-source-installation): Install the msit package through pip. Packages are released once a quarter.

##### Source Code Installation

```shell
git clone https://gitcode.com/Ascend/msit.git
# 1. git pull origin to update the latest code
cd msit/msit

# 2. Install the msit package
pip install .

# 3. Install elb:
msit install elb

# 4. After installation, use the msit check command to verify the installation:
msit check elb
```

##### pip Source Installation

```shell
# 1. Install the msit package
pip install msit

# 2. Install elb:
msit install elb

# 3. After installation, use the msit check command to verify the installation:
msit check elb
```

### Constraints

When using the speculative-moe level 1 and speculative-moe level 2 algorithms, the corresponding files need to be input to the tool separately for the prefill and decode scenarios. The input path cannot contain both prefill and decode at the same time.

## Feature Introduction

**Command Format**

 ```Shell
msit elb -icp <info_csv_path> -dt <device_type> [options]
```

**Parameter Description**

| Parameter      | Required                   | Usage Description                                                     |
| ----------- | ---------------------- | ------------------------------------------------------------ |
| --info-csv-path, -icp  | Yes       | Path to read the expert popularity information file. Data type: str. No default value. The input path must exist.    |
| --output-dir, -o       | No       | Path to output the final configuration file. Data type: str. The default is the current path. If a path is provided, the path must exist.  |
| --num-redundant-expert, -nre  | No       | Number of redundant experts. Data type: int. The number of redundant experts is the total across all cards. The default value is 64.      |
| --num-share-expert-devices, -nsed  | No       | Number of cards for independent deployment of shared experts. Data type: int. The number of shared experts is the number of A3 external experts or A2 mixed placement experts. Applicable to speculative-moe algorithms in A2/A3 scenarios. The default value is 0.      |
| --num-nodes, -nd       | No       | Number of nodes, that is, the number of machines. Data type: int. The default value is 8.            |
| --num-npus, -nn        | No       | Number of NPU cards. Data type: int. The default value is 64.            |
| --algorithm, -al       | No       | Algorithm type selection. Data type: str. 0 represents the C2LB algorithm, 1 represents the speculative-moe level 1 algorithm, 2 represents the C2LB algorithm in dynamic scenarios (generating initial configuration file), 3 represents the speculative-moe level 2 algorithm, 4 represents the speculative-moe level 1 mixed placement algorithm, 5 represents the speculative-moe level 2 mixed placement algorithm. The default value is 3. |
| --device-type, -dt  | Yes      | Server type. a2 represents deployment on Atlas 800I A2 servers, a3 represents deployment on Atlas 800I A3 servers. Data type: str. No default value. You need to input it yourself.            |
| -h, --help             | No       | Tool usage help information. Prints the command help information of the tool.        |

### Precautions

The load balancing expert popularity information file required by this tool depends on the dump capability provided by MindIE. Two MindIE environment variables need to be enabled to obtain the input file. Ensure that business data or datasets can be collected in pure model or service scenarios while model inference can be completed normally. Set export MINDIE_ENABLE_EXPERT_HOTPOT_GATHER=1 and set export MINDIE_EXPERT_HOTPOT_DUMP_PATH="user-defined path". It also supports expert popularity input from the VLLM-Ascend service inference framework.

## Appendix

### Algorithm Introduction

#### C2LB Algorithm

##### Requirement: The Number of Redundant Experts Must Be Less Than or Equal to the Number of NPUs

For MoE models, when using expert parallelism (EP), different experts are assigned to different GPUs or NPUs. Since the load of different experts may vary based on the current workload, maintaining load balance across different GPUs or NPUs is very important.
C2LB (Compute Communication Load Balance) is a static expert placement strategy. Based on offline statistical expert load information, it globally considers compute communication balance and outputs the expert placement strategy on cards. This algorithm supports redundant expert placement and supports deployment with different numbers of experts on different cards. The current constraint is that the number of cards must be greater than or equal to the number of redundant experts, and each card can only deploy one redundant expert. Due to the deviation of offline statistical expert load information from datasets, the static expert deployment plan is difficult to handle scenarios with large load changes.

#### speculative moe Algorithm

In MoE inference scenarios, the activation levels of different experts are naturally uneven, causing severe imbalance in expert computation and all2all communication loads. This leads to fast and slow cards and resource idle, requiring load balancing measures to improve performance. The speculative-moe algorithm is a class of expert balanced deployment optimization algorithms, applied during the initialization phase of the inference system. Calling this algorithm generates a global redundant expert deployment table. After application, it can significantly improve load balance and achieve inference performance optimization. Its core advantages include:

1) Comprehensive support for A2/A3 generations, Prefill/Decode scenarios, and adaptation to various expert deployment forms (shared expert built-in, external, mixed placement (note));
2) Fine-grained tidal feature mining strategy based on expert popularity, collaborating with the inference framework side's redundant expert design, efficient popularity collection, and expert deployment form optimization, achieving load balance improvement in multiple scenarios, with end-to-end performance improvement of 15%+.
Among them, the Level 1 algorithm achieves fast matching of hot and cold experts through multi-stage tidal popularity, completing deployment calculation in sub-minute time. Enable method: al 1 (non-mixed placement); al 4 (mixed placement). The Level 2 algorithm further uses Stochastic Mixed Integer Linear Programming (SMILP) and black-box optimization on the basis of Level 1 for higher-precision optimization. The load balance can be further optimized by 5%+. Enable method: al 3 (non-mixed placement); al 5 (mixed placement).

**Note:**
Built-in: Shared and non-shared experts are deployed together on each card.
External: Shared and non-shared experts are deployed separately on different cards.
Mixed placement: Shared and non-shared experts are flexibly combined and deployed, with shared experts optimally selecting their placement positions.

**Recommended**: Among all the above algorithms, the speculative-moe Level 2 mixed placement algorithm (al 5) achieves the best performance on A2 inference servers, and the speculative-moe Level 2 algorithm (al 3) achieves the best performance on A3 supernodes.

### FAQ

- For issues during installation, first refer to the [FAQ](../install/FAQ.md).
