# msit profile analyze Usage Guide

## Introduction

- A performance tuning and analysis tool for recommendation scenarios. Currently, it only supports analyzing GE auto-fused operators.

## Usage

### Prerequisites

- During operation, the tool calls the ATC tool to convert GE dump graphs. Before use, ensure that the CANN environment variables are sourced. For example: source /usr/local/Ascend/ascend-toolkit/set_env.sh

### Usage Entry

After installing the msit-profile tool, you can use it directly from the command line.

```bash
msit profile analyze --origin /tmp/op_summary_origin.csv --fused /tmp/op_summary.csv --ops-graph /tmp/ge_proto_00000001_graph_1_Build.txt
```

### Parameter Description

  | Parameter                    | Description                                       | Required |
  | ------------------------ | ---------------------------------------- | ---- |
  | --mode            | Configure the inference scenario mode, for example, single operator or graph mode. Currently, only graph mode is supported. | No  |
  | -f, --framework             | Configure the AI framework for model inference. Currently, only TensorFlow is supported.           | No  |
  | --origin       | The op_summary file collected when operator fusion is not enabled. | Yes  |
  | --fused       | The op_summary file collected after operator fusion is enabled. | Yes  |
  | -ops, --ops-graph       | The final generated graph of GE (Graph Engine), that is, the graph after GE optimization and compilation, for example, ge_proto_xx_Build.txt. If there are multiple graph files, you can enter the parent directory, for example, --ops-graph /tmp. Avoid placing graph files of different models in the same directory. | Yes |
  | -o, --output       | The save path for performance analysis results. The default is the current path. | No  |
  | -h, --help               | Tool usage help information.               | No  |

## Output Introduction
>
> Performance Analysis Results

After running, a profile_analysis.csv file is generated in the specified output path. It records the performance analysis results of fused operators. Currently, only two types of auto-fused operators are supported (operator types AscBackend and FusedAscBackend).

- HBM (High Bandwidth Memory) refers to the data transfer volume, in KB.
The analysis results have many columns. The meaning of each column is described below:

  | Column Name                    | Description                                       |
  | ------------------------ | ---------------------------------------- |
  | Fuse OpName   | The operator name of the fused operator |
  | Fuse OpType   | The operator type of the fused operator    |
  | Origin Ops    | The list of original operator tuples corresponding to the fused operator, recording the operator name and operator type of each original operator   |
  | Fused Durations(us)   | The total duration of the fused operator  |
  | Origin Durations(us)   | The total duration of all single operators before fusion  |
  | Time Ratio    | The duration of the fused operator **divided by** the total duration of operators before fusion  |
  | Time Difference   | The duration of the fused operator **minus** the total duration of operators before fusion       |
  | HBMs Difference  | The HBM of the fused operator **minus** the total HBM of operators before fusion  |
  | HBMs Ratio   | The HBM of the fused operator **divided by** the total HBM of operators before fusion  |
  | Fused HBMs(KB) | The input and output HBM of the fused operator (in KB)  |
  | Origin Duration(us) Each Op   | The list of durations for each original operator, recording the operator name and corresponding duration of each original operator     |
  | Origin HBMs Each Op(KB) | The HBM list for each original operator, recording the input HBM and output HBM of each original operator |
  | Origin HBMs Total(KB) | The total input HBM and total output HBM of all original operators |
  | Not Found Origin Op   | The names of operators for which profiling information was not collected before fusion |

> ATC Conversion File

  During operation, the ATC tool is called to convert the GE dump graph. The converted file is saved in the path specified by the --output parameter. The file name is the same as before conversion, and the suffix changes to 'json', that is:
  ge_proto_00000001_graph_1_Build.txt -> ge_proto_00000001_graph_1_Build.json.
