# Load balancing tool

## Brief Introduction

The Expert Load Balancing Tool is used to perform load balancing affinity expert optimization in static or dynamic MoE scenarios. The expert popularity information file (in CSV or PT format) dumped during model inference is loaded. After algorithm calculation, the file is flushed to the disk in the prefill or decoder phase.

Currently, the following algorithms are provided in two scenarios:

 * Static scenario + Atlas 800I A2 inference server: computing communication load balancing algorithm (C2LB), speculative-moe level 1 algorithm, speculative-moe level 2 algorithm, speculative-moe level 1 mixed algorithm, and speculative-moe Level 2 mixed algorithm;
 * Static scenario + Atlas 800I A3 inference server: speculative-moe level 1 and speculative-moe level 2 algorithms;
 * Dynamic scenario + Atlas 800I A2 inference server: Initial configuration file for the C2LB algorithm.

**Preparation Before Use**

### Environment preparation

1. Prepare a Ascend NPU-based inference server.
2. Install Python 3.9 or later.

### Installation Operation

1. Install the msIT tool and software.[Installing the Load Balancing Affinity Expert Optimization Tool](../install/README.md)    .
2. After the msIT is installed, run the msit install elb command to install the elb component in the msIT.

### Constraints

When the speculative-moe level 1 and speculative-moe level 2 algorithms are used, you need to import the corresponding files to the tool separately in the prefill and decode scenarios. The input path cannot contain both prefill and decode.

## Function Description

**Command Format**

```shell
msit elb -icp <info_csv_path> -dt <device_type> [options]
```

**Parameter Description**

| Parameters                        | Mandatory or not | Instructions for use                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| --info-csv-path, -icp             | Yes              | Path for reading the expert popularity information file. Data type: There is no default value for str. The input path must exist.                                                                                                                                                                                                                                                                                                                                  |
| --output-dir, -o                  | No.              | Output the path of the final configuration file. Data type: str. The default value is the current path. If a path is input, the path must exist.                                                                                                                                                                                                                                                                                                                   |
| --num-redundant-expert, -nre      | No.              | Number of redundant experts. Data type: int. The number of redundant experts is the total number on all cards. The default value is 64.                                                                                                                                                                                                                                                                                                                            |
| --num-share-expert-devices, -nsed | No.              | Number of cards for independently deployed Shared Expert. Data type: int. The number of shared experts equals the number of A3 external experts or the number of A2 hybrid experts. This parameter is applicable to the speculative-moe algorithm in A2/A3 scenarios. The default value is 0.                                                                                                                                                                      |
| --num-nodes, -nd                  | No.              | Number of nodes, that is, the number of machines. Data type: int. The default value is 8.                                                                                                                                                                                                                                                                                                                                                                          |
| --num-npus, -nn                   | No.              | Number of NPUs. Data type: int. The default value is 64.                                                                                                                                                                                                                                                                                                                                                                                                           |
| --algorithm, -al                  | No.              | Select the algorithm type. Data type: str. The value 0 indicates the C2LB algorithm, 1 indicates the speculative-moe level 1 algorithm, 2 indicates the C2LB algorithm in dynamic scenarios (including the initial configuration file), and 3 indicates the speculative-moe level 2 algorithm. The value 4 indicates the speculative-moe level 1 hybrid algorithm, and the value 5 indicates the speculative-moe level 2 hybrid algorithm. The default value is 3. |
| --device-type, -dt                | Yes              | Specifies the server type. a2 indicates the Atlas 800I A2 server deployment, and a3 indicates the Atlas 800I A3 server deployment. Data type: str. There is no default value. You need to enter a value.                                                                                                                                                                                                                                                           |
| -h, --help                        | No.              | Help information about the tool. Prints the help information about the tool.                                                                                                                                                                                                                                                                                                                                                                                       |

### Precautions

The load balancing expert popularity information file required by this tool depends on the dump capability provided by MindIE. The environment variables of two MindIEs need to be enabled to obtain the input file. This ensures that service data or datasets can be collected in pure model or service-oriented scenarios and model inference can be completed properly. Set export MINDIE_ENABLE_EXPERT_HOTPOT_GATHER=1 and export MINDIE_EXPERT_HOTPOT_DUMP_PATH= User-defined path. It also supports the input of expert popularity in the VLLM-Ascend service-based inference framework.

## Appendixes

### Algorithm Introduction

#### C2LB algorithm

##### The number of redundant experts must be less than or equal to the number of NPUs

For the MoE class model, when using Expert Parallelism (EP), different experts are allocated to different GPUs/NPUs. Since the load of different experts may vary according to the current workload, it is important to maintain load balancing among different GPUs/NPUs. The C2LB algorithm is a static expert placement policy. Based on the offline expert load information, the C2LB algorithm takes into account the global computing communication balance and outputs the expert placement policy on the card. This algorithm supports the placement of redundant experts and supports the deployment of different numbers of experts on cards. Currently, the number of restricted cards is greater than or equal to the number of redundant experts, and only one redundant expert can be deployed on each card. Due to the deviation of expert load information in offline statistics of datasets, the static expert deployment solution cannot cope with the scenario where the load changes greatly.

#### speculative moe algorithm

In the MoE inference scenario, the activation volume of different experts is uneven, resulting in severe imbalance between expert computing and all2all communication loads, resulting in fast and slow cards and resource bubbles. Therefore, load balancing needs to be used to improve performance. Speculative-moe algorithm is a kind of expert balanced deployment optimization algorithm, which is applied to the initialization stage of inference system. This algorithm is invoked to generate a global redundant expert deployment table. After the algorithm is used, the load balancing degree can be significantly improved and the inference performance can be optimized. Its core strengths include:

1) A2/A3 intergenerational, prefill, and decode scenarios are fully supported, adapting to multiple expert deployment modes. (Share the built-in, location, and hybrid deployment of experts (note));
2) Based on the fine-grained tidal feature mining policy for expert popularity, the redundant expert design, efficient popularity collection, and expert deployment mode optimization are coordinated with the inference framework, improving multi-scenario load balancing and E2E performance by more than 15%. The Level 1 algorithm quickly matches hot and cold experts through multi-phase tidal popularity and completes deployment calculation within subminutes. The enabling mode is al 1 (non-mixed deployment). al 4 (mixed); Based on Level 1, the Level 2 algorithm further uses Stochastic Mixed Integer Linear Programming (SMILP) and black box optimization to perform higher precision optimization. The load balancing degree can be further optimized by more than 5%. The enable mode is al 3 (non-mixed). al 5 (mixed).

**Note:**
Built-in: Shared/non-shared experts are deployed on each card.
External: Shared and non-shared experts are deployed separately on different cards.
Mixed deployment: Shared and non-shared experts can be flexibly deployed together, and sharing experts can select the optimal deployment position.

**Preferred recommendation: Among the preceding algorithms, the Speculative-moe Level 2 hybrid algorithm (al 5) achieves the optimal performance in the A2 inference server, and the Speculatvie-moe Level 2 algorithm (al 3) achieves the optimal performance in the A3 supernode.**
