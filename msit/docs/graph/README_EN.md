# Usage Description of the msit graph

msit graph is a graph performance analysis tool based on the Graph Engine (GE). Currently, msit graph provides the following functions:

 * stats: Query the node statistics (number of nodes of each type) of the graph.
 * extract: Extracts specific submaps from the entire image, including diffusion extraction and interval extraction.
 * Strip: Some traditional model diagrams have many nodes. When you use the visualization tool to view the model structure, frame freezing often occurs. This function can remove the constants and node attributes on the diagram and compress the diagram size to about 1/10 of the original size, facilitating the view of the diagram structure.
 * fuse (identifying duplicate structures): Analyze the number of duplicate structures on a graph and further analyze the time consumption of a duplicate structure based on the profiling data. It is usually used to analyze the convergence opportunities on the graph.
 * inspect: To improve the training and inference performance, replace the dynamic shape operator in the GE graph to obtain a static graph. Therefore, you need to filter the dynamic shape operators in the graph first.

## Installation Method

```bash
#1. Source code installation: Download the source code and go to the source code directory.
```sh
git clone https://gitcode.com/Ascend/msit.git
cd msit/msit
pip install .

#2.1 Installing the Graph Tool
msit install graph

#2.2 Installing the WHL Package
cd ./components/graph

python setup.py bdist_wheel
cd ./dist
pip install msit*.whl
```

Select either 2.1 or 2.2.

## Statistics node information

Count the number of nodes of each type in a chart. For example, the file name of a dump chart is`ge_onnx_00449_graph_101_Build.pbtxt` [Description of the dump diagram](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/envvar/envref_07_0011.html)    , you can run the following command to view the graph statistics:

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

The displayed information includes the node type and the number of nodes of this type in the graph. This command helps you quickly obtain the brief information about a graph. You can quickly check whether the two charts are consistent and whether the convergence rule takes effect based on the summary information.

## Extract Graph

When a graph contains a large number of nodes, the graph structure may be interrupted when the visualization tool is used to analyze the graph structure. If you need to view only a part of the graph, you can extract a part of the graph as required.

To extract a submap, run the following command:`extract`Generally, subgraph extraction methods are classified into two types: diffusion extraction based on node names and interval extraction based on start and end nodes. Diffusion extraction and interval extraction can be used separately or together.

### Diffusion extraction

#### Default mode

You can specify the name of a central node and perform diffusion forward or backward based on the node to extract subgraphs. For example, the name of a graph is`ge_onnx_00449_graph_101_Build.pbtxt`, with the node`MatMul_1/v2`For the center, run the following command to spread three layers:

```bash
msit graph extract --input=ge_onnx_00449_graph_101_Build.pbtxt --center-node "MatMul_1/v2" --layer-number 3
```

After the command is executed, the tool performs the following tasks:

1. Load diagram file`ge_onnx_00449_graph_101_Build.pbtxt`
2. Find the name from the diagram.`MatMul_1/v2`Node of
3. With the node`MatMul_1/v2`Searches for all input and output nodes cyclically in the input and output directions, deduplicates and saves all found nodes in the range of three layers.
4. Dump all found nodes into a graph.
5. Search for the input and output nodes of all found nodes and dump them to the graph.

After the execution is complete,`ge_onnx_00449_graph_101_Build.pbtxt`A directory at the same level will be generated.`ge_onnx_00449_graph_101_Build_*.pbtxt`File of. This file is the extracted submap.

For details about the extraction rules, see the following figure. In the following figure, the nodes to be extracted are specified in blue.

![抽取示意图](image/extract.PNG)    

The green parts are all the nodes found in step 3 by spreading to the input and output layers. Blue and green can be considered as the "backbone" of the graph to be extracted, and yellow parts are the leaf nodes of these backbone nodes, which will also be extracted together. If the gray node does not belong to the backbone node or is not directly connected to the backbone node, it will not be extracted.

Note that the yellow node A is also 3 away from the blue node, but this node does not belong to the backbone node. The reason is that node A is an "input node on the branch in the output direction". During cyclic search, the backbone node in the output direction continues to search only in the output direction. Therefore, node A is not in the backbone node. However, because A is directly connected to the backbone node (node 2), A still appears in the extraction diagram.

#### Canceling the Dump Leaf Node

In the[Default mode](#default-mode)    As mentioned in, leaf nodes are dumped to the extracted graph along with their directly connected backbone nodes. If you don't want this to happen, you can`--without-leaves`Option to prevent leaf nodes from being dumped into the extracted graph.[Extraction Diagram](image/extract.PNG)    See, when this option is specified, only the blue and green nodes will be extracted.

#### Abort extraction

In the process of graph data extraction, there are sometimes specific types of nodes encountered, and the existence of these nodes may significantly affect the size and complexity of the extraction results. To ensure the effectiveness and controllability of the extraction process, it is usually desirable to stop further extraction operations when the following conditions are detected: High Connectivity Node:

 * This type of node is characterized by extremely high degree of entry and exit. In some network structures, the degree of in or out of a single node may exceed thousands. If such nodes are used during extraction, even if the number of layers to be extracted is limited, the number of extracted subgraphs may be extremely large. Nodes of specific types:
 * In some application scenarios, you may want to terminate the graph extraction process after encountering a specific type of node or a specified node. This requirement may be due to a concern for specific business logic, or to solve performance optimization problems. Regardless of the motivation, the extraction operation should stop as soon as the predefined node type or specific node is identified to ensure that the generated subgraph meets the expected boundary conditions and analysis requirements.

At this point, you can use the`--stop-name`Specifies that the extraction is aborted, for example,`--stop-name "allreduce_1"`, you can specify the name of the node to be stopped.

#### Unidirectional extraction

If you only want to perform unidirectional extraction starting from a certain node and only forward or backward, you can run the`--only-forward`or the`--only-backward`Specifies the direction of forward and backward.

![forward_backward](image/fw_bw.PNG)    

### Interval Extraction Method

In this method, you need to specify a group of start and end nodes and dump all nodes between the start and end nodes.

Example command:

```bash
msit graph extract --input=ge_onnx_00449_graph_101_Build.pbtxt --start-node "MatMul_1/v2" --end-node "Mul_4" --without-leaves
```

In addition, it should be noted that only one submap can be generated each time for diffusion extraction and interval extraction, that is, only one central node or a group of start and end nodes can be input, and the two extraction modes cannot be used at the same time.

## Compressed Graph

If the size of a pbtxt file is very large, it takes a long time to open it using the visualization tool or even cannot be opened. To quickly view the graph structure, run the strip command. The following is an example command:

```bash
msit graph strip --input ge_onnx_00449_graph_101_Build.pbtxt --level 3
```

Different levels correspond to different compression levels. 1 indicates the lowest compression level and retains the most information on the graph. 3 indicates the highest compression level and retains only the basic graph structure information. The default value is 3. For details about each level, see the parameter description.

## Identifying Duplicate Structures

Some traditional model diagrams have many nodes, and it is difficult and time-consuming to manually identify the fusion opportunity on the diagrams. This function automatically calculates the number of occurrences of all possible structures in the diagrams and provides the total time consumed by each duplicate structure based on the profiling data. The following is an example command:

```bash
msit graph fuse --source ge_onnx_00449_graph_101_Build.pbtxt --profile op_summary_*.csv --max-nodes 8
```

In the preceding command, --profile indicates the profile file. For details about how to obtain the file, see.[Performance Analysis in Offline Inference Scenarios](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/profiling/atlasprofiling_16_0005.html)    , --max-nodes indicates the maximum number of nodes that can be contained in the duplicate structure. The current calculation method takes into account the possibility in the figure. This method has high time complexity and therefore supports only configuration.`--max-nodes`<=10. After the command is executed, a CSV file named fuse_duration_\{timestamp\} is generated. The table header is as follows:

| Subgraph                                      | Count                                        | Root Nodes Index                                                                         | Task Sum Duration(us)                          | Total Duration(us)                            |
| --------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------- |
| Repetitive structures appearing in the figure | Number of occurrences of repeated structures | Repeat the index value of the root node to find the specific node position in the graph. | Time required for a single duplicate structure | Total time required by the repeated structure |

It should be noted that the "repeated structure" is composed of operator types instead of operator names. To find the position of a repeated structure in the graph, combine the root node of the repeated structure and the value of Root Nodes Index with several node names. The first node of the repeating structure is the root node.

## Figure Scan

To improve the training and inference performance, the dynamic shape operator in the GE graph needs to be replaced to obtain a full static graph. Therefore, you need to filter the dynamic shape operators in the graph first.

```sh
msit graph inspect <options>
```

Example command:

```sh
msit graph inspect -i ./test_pbgraph.pbtxt -t dshape -o ./output
```

## Parameter Description

### General Parameters

| Parameter name  | Function of the parameter                                                    | Mandatory or not |
| --------------- | ---------------------------------------------------------------------------- | ---------------- |
| --input         | Read .pbtxt file                                                             | Yes              |
| --log-level, -l | Log level. The options are debug, info, warning, error, fatal, and critical. | No.              |

### Extract Parameter

| Parameter name   | Function of the parameter                                           | Mandatory or not |
| ---------------- | ------------------------------------------------------------------- | ---------------- |
| --start-node     | Start node for interval extraction.                                 | No.              |
| --end-node       | End node for interval extraction.                                   | No.              |
| --center-node    | Perform diffusion extraction based on the central node name.        | No.              |
| --layer-number   | Number of layers extracted by the central node forward or backward. | No.              |
| --only-forward   | Extract only forward                                                | No.              |
| --only-backward  | Extract backwards only                                              | No.              |
| --without-leaves | Do not extract leaf nodes.                                          | No.              |
| --stop-name      | When the node name is found, the extraction is stopped.             | No.              |
| --output         | Generated .pbtxt submap                                             | No.              |

### Strip parameter

| Parameter name | Function of the parameter                                                                                                                                                                                                                                                | Mandatory or not |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| --level        | Compression level. The options are as follows:<br/>1: The Const Data node is deleted from the figure.<br/>2: Delete the attributes of the Const Data node and all nodes except the shape information.<br/>3: Delete the attributes of the Const Data node and all nodes. | No.              |
| --output       | Generated .pbtxt submap                                                                                                                                                                                                                                                  | No.              |

### Fuse parameter

| Parameter name | Function of the parameter                                                                                      | Mandatory or not |
| -------------- | -------------------------------------------------------------------------------------------------------------- | ---------------- |
| --source       | Read .pbtxt file                                                                                               | Yes              |
| --profile      | The profile.csv file is read.                                                                                  | Yes              |
| --max-nodes    | Maximum number of nodes that a duplicate structure can contain.                                                | No.              |
| --min-nodes    | Minimum number of nodes allowed in a duplicate structure.                                                      | No.              |
| --min-times    | Only the results with the number of occurrences of duplicate structures not less than this value are presented | No.              |
| --output       | Generated .csv file                                                                                            | No.              |

### inspect parameter

| Parameter name | Function of the parameter                                                                                               | Mandatory or not |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------- |
| --type, -t     | Specifies the scanning type. Currently, only the dynamic shape (dshape) is supported.                                   | Yes              |
| --output, -o   | Output directory. The current output table header is Graph_Name, Node_Name, Input, and Output. The default value is ./. | No.              |

## Disclaimer

 * This tool is for debugging and development only and is not applicable to the production environment. The user shall use it at his/her own risk and understand the following:
    
     * \[X\] For debugging and development only: This tool is designed to assist developers in debugging. It is not applicable to the production environment or other commercial use. This tool and its developers are not liable for any data loss or damage caused by misuse of this tool.
     * \[X\] Data processing and deletion: Data generated during the use of the tool (including but not limited to dump data) is within the user's responsibility. You are advised to delete related data in a timely manner after using the data to prevent leakage or unnecessary information disclosure.
     * \[X\] Data confidentiality and dissemination: Users understand and agree not to send or distribute the data generated by this tool. This tool and its developers are not responsible for any information leakage, data breach, or other adverse consequences arising therefrom.
     * \[X\] User input security: Users must ensure the security of the entered command lines and bear any security risks or losses caused by improper input. This tool and its developers are not responsible for any problems caused by improper command line input.
 * Scope of Disclaimer: This disclaimer applies to all individuals or entities using this tool. By using this tool, you agree to and accept the content of this statement and are willing to bear the risks and responsibilities arising from the use of this function. If you have any objection, stop using this tool.
 * Read and understand the disclaimer before using this tool. If you have any questions or questions about using this tool, contact the developer.
