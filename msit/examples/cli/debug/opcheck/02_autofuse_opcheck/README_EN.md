# Pre-checking the precision of the automatic fusion operator #

## Introduction ##

This tool can automatically parse the node type and input and output dtype and shape information in the graph of the ascgraph, construct a graph consisting of TensorFlow APIs, and run the graph to obtain a benchmark result. Then, the results can be compared with the results of the ascgraph fusion operator process to complete the fusion operator pre-check.

## Running an Example ##

### 1. Dump the tensor data on the ASCII graph and NPU. ###

Ensure that the automatic convergence feature is enabled and GE dump graph parameters are configured. The procedure is as follows:

```
#Auto Blend Switch
export EXPERIMENTAL_ENABLE_AUTOFUSE=1
export EXPERIMENTAL_LOWERING_REDUCE=1
export EXPERIMENTAL_LOWERING_CONCAT=1
#GE dump graph switch
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
export DUMP_GRAPH_PATH=/home/dump_graph  #Configure the specified dump path.
```

The data dump function of the NPU can be enabled by using the ConfigProto function of the TensorFlow Adaptor. The configuration is as follows:

```
import tensorflow.compat.v1 as tf  #In the tf1.x version, import tensorflow as tf is changed.
config = tf.ConfigProto()

custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["enable_dump"].b = True
custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/dump_data") 
custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0")
custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
with tf.Session(config=config) as sess:
  sess.run() #Model Inference Entry
```

### 2. Perform the precheck. ###

Run the following command to check the environment of the source CANN:

```
#The parameter settings in the example commands are based on the dump diagram and tensor data path in the previous section. You need to modify the parameters based on the site requirements.
msit debug opcheck -i {指定到NPU dump数据的上层目录，具体配置提示在‘注意事项’章节中介绍} -m autofuse -gp /home/dump_graph
```

After the execution is complete, the precheck result is saved in the current path where the command is executed. The file name is`autofuse_opcheck_result.csv`.

## Precautions ##

 *  Currently, this tool supports only the fusion operator whose operator type is AscBackend.
 *  During the running, the tool creates a directory in the directory specified by the -o parameter.`tmp`Subdirectory of, used to store the NPU dump data parsed files in the NPU in the Npy format. The files in this directory will be automatically deleted after the precheck is complete. If the program exits illegally, these files will be reserved for debugging.
 *  The path specified by the -i parameter must be the upper-level directory of the NPU dump data.

For example, if the dump function is enabled and the specified path is /home/dump_data, subdirectories are created in the specified directory for data flushing. The directory structure is as follows: /home/dump_data/\{timestamp\}/\{device_id\}/\{model name\}/\{model ID\}/\{step_id\}. Therefore, the path specified by the -i parameter must be /home/dump_data/\{timestamp\}/\{device_id\}/\{model name\}/\{model ID\}/\{step_id\} instead of /home/dump_data.

