# msit graph Usage Instructions

msit graph is a graph performance analysis tool based on GE (Graph Engine). The following functions are currently provided:

* stats (statistics): View node statistics of the graph (the number of nodes of each type).
* extract: Extract specific subgraphs from the entire graph, including **diffusion extraction** and **range extraction** modes.
* strip: Some traditional model graphs have a very large number of nodes. Using visualization tools to view the model structure often causes lag. This function can remove constants and node attributes from the graph, compressing the graph size to approximately 1/10 of the original, making it easier to view the graph structure.
* fuse (identify repeated structures): Analyze the number of repeated structures in the graph and combine profiling data to further analyze the duration of a "repeated structure". This is typically used to analyze fusion opportunities in the graph.
* inspect (graph scan): To improve training and inference performance, dynamic shape operators in the GE graph need to be replaced to obtain a fully static graph. Therefore, the dynamic shape operators in the graph need to be filtered first.

## Installation Methods

```bash
# 1. Source code installation: first download the source code and enter the source code directory
```sh
git clone https://gitcode.com/Ascend/msit.git
cd msit/msit
pip install .

# 2.1 Install the graph tool
msit install graph

# 2.2 Install the whl package
cd ./components/graph

python setup.py bdist_wheel
cd ./dist
pip install msit*.whl
```

Choose either 2.1 or 2.2.

## Statistics Node Information

Count the number of nodes of each type in the graph. For example, if the dump graph file name is `ge_onnx_00449_graph_101_Build.pbtxt` [DUMP Graph Description Information](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/envvar/envref_07_0011.html), the command to view the graph statistics is:

```bash
msit graph stats --input ge_onnx_00449_graph_101_Build.pbtxt
load from ge_onnx_00449_graph_101_Build.pbtxt
graph stat:
        ge:Add = 11
        ge:AddN = 102
        ge:ApplyMomentumD = 269
        ge:ArgMaxD = 24
        ge:AssignAdd = 1
        ge:AtomicAddrClean = 1
        ge:BNTrainingReduceGrad = 97
        ge:BNTrainingUpdate = 97
        ge:BNTrainingUpdateGrad = 97
        ge:BiasAdd = 13
...

```

The displayed content shows the node type and the number of nodes of that type in the graph. Through this command, you can quickly obtain summary information of a graph.
You can use the summary information to quickly confirm whether two graphs are consistent and whether fusion rules are in effect.

## Extract Graph

When the number of nodes in a graph is very large, using visualization tools to analyze the graph structure usually causes lag. If you only need to view a portion of the graph, you can extract a small part of the graph as needed.

The command to extract a subgraph is `extract`. In general, there are two methods for extracting subgraphs: diffusion extraction based on node names, and range extraction based on start and end nodes.
Diffusion extraction and range extraction can be used independently or together.

### Diffusion Extraction

#### Default Mode

You can specify a center node name and diffuse forward or backward based on the node to extract a subgraph.
For example, if the graph file name is `ge_onnx_00449_graph_101_Build.pbtxt`, with the node `MatMul_1/v2` as the center, diffusing 3 layers forward and backward, execute the command:

```bash
msit graph extract --input=ge_onnx_00449_graph_101_Build.pbtxt --center-node "MatMul_1/v2" --layer-number 3
```

After entering this command, the tool performs the following tasks:

1. Load the graph file `ge_onnx_00449_graph_101_Build.pbtxt`.
2. Find the node named `MatMul_1/v2` in the graph.
3. With the node `MatMul_1/v2` as the center, search for all input and output nodes in the input direction and output direction cyclically, within a range of 3 layers, and deduplicate and save all found nodes.
4. Dump all found nodes into a graph.
5. Find the input and output nodes of all found nodes and dump them into the graph as well.

After execution, a file named `ge_onnx_00449_graph_101_Build_*.pbtxt` is generated in the same directory as `ge_onnx_00449_graph_101_Build.pbtxt`. This file is the extracted subgraph.

Regarding the extraction rules, a detailed explanation is provided through the example in the following figure. In the figure below (extraction diagram), the blue nodes are specified as the nodes to be extracted:

![Extraction Diagram](image/extract.PNG)

The green part represents all nodes found by diffusing 3 layers in the input and output directions in step 3.
The blue and green parts can be considered as the "backbone" of the graph we need to extract. The yellow part represents the leaf nodes of these backbone nodes, which are also extracted together.
Gray nodes are not backbone nodes and are not directly connected to backbone nodes, so they are not extracted.

Note the yellow node A. The distance from this node to the blue node is also 3, but this node is not a backbone node.
The reason is that node A is "an input node on the output direction branch", and during cyclic search, backbone nodes in the output direction only continue searching in the output direction, so A is not in the backbone nodes.
However, since A is directly connected to a backbone node (node 2), A still appears in the extracted graph.

#### Cancel Dumping Leaf Nodes

As mentioned in the [Default Mode](#default-mode), leaf nodes are dumped together with their directly connected backbone nodes into the extracted graph.
If you do not want this behavior, you can use the `--without-leaves` option to prevent leaf nodes from being dumped into the extracted graph.
From the [Extraction Diagram](image/extract.PNG), when this option is specified, only the blue and green nodes are extracted.

#### Stop Extraction

During graph data extraction, you may encounter specific types of nodes whose presence may significantly affect the scale and complexity of the extraction results. To ensure the effectiveness and controllability of the extraction process, when the following situations are detected, it is usually desirable to stop further extraction:
**High Connectivity Nodes:**

* These nodes are characterized by extremely high in-degree and out-degree. In some network structures, the in-degree or out-degree of a single node may exceed thousands. If such nodes are encountered during extraction, even if the number of extraction layers is limited, the resulting subgraph may be abnormally large.
**Specific Type Nodes:**

* In certain application scenarios, users may want to stop the graph extraction process after encountering a specific type of node or a clearly specified node. This requirement may arise from a focus on specific business logic or to address performance optimization issues. Regardless of the motivation, once a predefined node type or specific node is identified, the extraction operation should stop immediately to ensure that the generated subgraph meets the expected boundary conditions and analysis requirements.

In this case, you can use `--stop-name` to specify stopping the extraction, for example, `--stop-name "allreduce_1"` to specify the node name to stop at.

#### Unidirectional Extraction

If you want to start from a node and perform only forward or backward unidirectional extraction, you can use `--only-forward` or `--only-backward`. The directions of forward and backward are:

![forward_backward](image/fw_bw.PNG)

### Range Extraction Method

This method requires specifying a set of start nodes and end nodes, and dumps all nodes between the start and end nodes.

Command sample:

```bash
msit graph extract --input=ge_onnx_00449_graph_101_Build.pbtxt --start-node "MatMul_1/v2" --end-node "Mul_4" --without-leaves
```

In addition, it should be noted that diffusion extraction and range extraction each support generating only one subgraph at a time, that is, only one center node or one set of start and end nodes is supported. The two extraction methods cannot be used simultaneously.

## Strip Graph

When the pbtxt file is very large, using visualization tools to open it is time-consuming or even impossible. If you want to quickly view the graph structure, you can use the strip command. The command sample is as follows:

```bash
msit graph strip --input ge_onnx_00449_graph_101_Build.pbtxt --level 3
```

Different levels correspond to different compression levels. Level 1 has the minimum compression and retains the most graph information. Level 3 has the maximum compression and only retains basic graph structure information. The default value is 3. For detailed descriptions of each level, refer to Parameter Details.

## Identify Repeated Structures

Some traditional model graphs have a very large number of nodes. Manually identifying fusion opportunities in the graph is difficult and time-consuming. This function automatically calculates the number of occurrences of all possible structures in the graph and provides the total duration of each repeated structure based on profiling data. The command sample is as follows:

```bash
msit graph fuse --source ge_onnx_00449_graph_101_Build.pbtxt --profile op_summary_*.csv --max-nodes 8
```

Where --profile is the profiling file. For how to obtain it, refer to [Offline Inference Scenario Performance Analysis](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/profiling/atlasprofiling_16_0005.html).
--max-nodes is the maximum number of nodes allowed in a "repeated structure". Since the current calculation method considers all possibilities in the graph, this method has high time complexity. Therefore, only `--max-nodes`<=10 is currently supported.
After the command is executed, a csv file named fuse_duration_{timestamp} is generated. The header is as follows:

|Subgraph         | Count             | Root Nodes Index                                  |Task Sum Duration(us)| Total Duration(us)    |
|----------------|-------------------|-----------------------------------------------------|-------------------|------------------------|
|Repeated structures in the graph| Number of occurrences of the repeated structure| Root node Index value of the repeated structure, to facilitate finding the specific node position in the graph   |Duration of a single repeated structure     | Total duration of the repeated structure         |

It should be noted that "repeated structures" are composed of operator types rather than operator names. If you want to find the position of a repeated structure in the graph, concatenate the root node of the repeated structure with the "Root Nodes Index" value to form several node names. The first node of the repeated structure is the root node.

## Graph Inspect

To improve training and inference performance, dynamic shape operators in the GE graph need to be replaced to obtain a fully static graph. Therefore, dynamic shape operators in the graph need to be filtered first.

```sh
msit graph inspect <options>
```

Command sample:

```sh
msit graph inspect -i ./test_pbgraph.pbtxt -t dshape -o ./output
```

## Parameter Details

### General Parameters

|Parameter| Description        |  Required  |
|----|-------------|----------------|
|--input| The .pbtxt file to read |    Yes  |
|--log-level, -l| Log level: debug, info, warning, error, fatal, critical.|No|

### extract Parameters

|Parameter| Description                         |  Required  |
|----|------------------------------|----------------|
|--start-node| Start node for range extraction                    |  No   |
|--end-node| End node for range extraction                    |  No   |
|--center-node| Center node name for diffusion extraction                |  No   |
|--layer-number| Number of layers to extract forward/backward from the center node               |  No   |
|--only-forward| Extract only forward                        |  No   |
|--only-backward| Extract only backward                        |  No   |
|--without-leaves| Do not extract leaf nodes                      |  No   |
|--stop-name| Stop extraction when this node name is encountered          |  No   |
|--output| The generated .pbtxt subgraph |   No  |

### strip Parameters

| Parameter    | Description                  |Required  |
|--------|-----------------------|----------------|
| --level | Compression level. Available values:<br>1: Remove Const and Data nodes from the graph;<br>2: Remove Const and Data nodes and all node attributes except shape information;<br>3: Remove Const and Data nodes and all node attributes |   No   |
|--output| The generated .pbtxt subgraph |   No  |

### fuse Parameters

|Parameter| Description        | Required  |
|----|-------------|------------|
|--source| The .pbtxt file to read |   Yes  |
|--profile| The profiling .csv file to read|   Yes  |
|--max-nodes| Maximum number of nodes allowed in a repeated structure|  No   |
|--min-nodes| Minimum number of nodes allowed in a repeated structure|  No   |
|--min-times| Only present results where the number of occurrences of the repeated structure is not less than this value|  No   |
|--output| The generated .csv file |   No  |

### inspect Parameters

|Parameter|Description|Required|
|-----|-----|-----|
|--type, -t|Specify the specific scan type. Currently, only "dynamic shape" (dshape) is supported.|Yes|
|--output, -o|Output directory. Currently, a csv file with headers Graph_Name, Node_Name, Input, Output is generated. The default is "./".|No|

## Disclaimer

- This tool is for debugging and development purposes only and is not suitable for production environments. Users bear the risk of use and should understand the following:

  - [X] For debugging and development only: This tool is designed to assist developers in debugging and is not suitable for production environments or other commercial purposes. The tool and its developers are not responsible for data loss or damage caused by misuse of this tool.
  - [X] Data processing and deletion: Data generated by users during the use of this tool (including but not limited to dumped data) is the responsibility of the user. Users are advised to delete relevant data promptly after use to prevent leakage or unnecessary information disclosure.
  - [X] Data confidentiality and dissemination: Users understand and agree not to arbitrarily send or disseminate data generated through this tool. The tool and its developers are not responsible for any information leakage, data leakage, or other adverse consequences arising from this.
  - [X] User input security: Users are responsible for ensuring the security of input command lines and bear any security risks or losses caused by improper input. The tool and its developers are not responsible for problems caused by improper command line input.
- Disclaimer scope: This disclaimer applies to all individuals or entities using this tool. Using this tool indicates that you agree to and accept the contents of this statement and are willing to bear the risks and responsibilities arising from the use of this function. If you disagree, stop using this tool.
- Before using this tool, please **carefully read and understand the contents of the above disclaimer**. For any problems or questions arising from the use of this tool, please contact the developers in a timely manner.
