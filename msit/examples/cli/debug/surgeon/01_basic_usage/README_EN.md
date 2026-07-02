# Basic Usage #

## Introduction ##

The surgeon tool can be started by running the msit command line.

## Running an Example ##

```
msit debug surgeon COMMAND [OPTIONS] [REQUIRED]
```

COMMAND indicates the surgeon execution mode parameter, and the value can be list, evaluate, optimize, extract, or concatenate. \[OPTIONS\] and \[REQUIRED\] are optional and mandatory parameters. The optional and mandatory parameters under each subtask are different.

## Usage Process ##

You are advised to run the surgeon tool in the sequence of list, evaluate, and optimize. If you need to cut a subgraph, you can run the extract command to export the subgraph.

The operation process is as follows:

1.  Run the list command to list all knowledge bases that support automatic optimization.
2.  Run the evaluate command to search for ONNX models that can be optimized by the specified knowledge base.
3.  Run the optimize command to optimize the specified ONNX model by using the specified knowledge base.
4.  Run the extract command to split the model into submaps.
5.  Run the concatenate command to concatenate the model.

