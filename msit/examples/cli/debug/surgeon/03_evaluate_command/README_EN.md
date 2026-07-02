# Evaluate Command #

## Introduction ##

Searches for ONNX models that can be optimized by the specified knowledge base.

```
msit debug surgeon evaluate [OPTIONS] PATH
```

Evaluate can be abbreviated as eva.

Parameter description:

| Parameter | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Mandatory or not |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| OPTIONS   | Additional parameter. The options are as follows:-know/--knowledges: knowledge base list. You can specify the name or sequence number of the knowledge base. Use commas (,) to separate the names and sequence numbers. All knowledge bases except the remediation nature are enabled by default. -r/--recursive: indicates whether to search for a folder in recursive mode. This function is disabled by default. -v/--verbose: Displays more information. Currently, only the search progress is displayed. -p/--processes: Use multiprocess to search for parallel processes and specify the number of processes. -h, --help: tool usage help information. The default value is 1. | No.              |
| REQUIRED  | --path: specifies the target of the evaluation. The value can be an .onnx file or a folder that contains .onnx files.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Yes              |

## Running an Example ##

```
msit debug surgeon evaluate --path=aasist_bs1_ori.onnx
```

An example output is as follows:

```
2023-04-27 14:37:10,364 - 984068 - msit_debug_logger - INFO - aasist_bs1_ori.onnx    KnowledgeConv1d2Conv2d,KnowledgeMergeConsecutiveSlice,KnowledgeTransposeLargeInputConv,KnowledgeTypeCast,KnowledgeMergeCasts
```

