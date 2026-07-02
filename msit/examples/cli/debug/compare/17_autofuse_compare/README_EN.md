# Network-wide precision comparison scenario for automatic convergence of GE ports #

## 1. Dependency ##

 *  CANN (8.1.RC1 or later)
 *  TensorFlow (1.15.0 or 2.65)

## 2. Tool installation ##

The msit tool and compare subtool must be installed.

 *  2.1 Install the msit tool. For details, see.[Installing the msit tool](../../../../../../msit/docs/install/README.md)	
 *  2.2 Install the compare sub-tool. Run the following command:`msit install compare`

## 3. Dump data ##

Before performing data dump comparison, disable all random configurations involved in the model, including but not limited to shuffle data sets and random parameter initialization.

### 3.1 Dumping Model Data on the NPU ###

Currently, the automatic convergence feature is controlled by three environment variables. The mapping between precision comparison operators depends on the GE dump diagram. The following provides a configuration example.

```
#Auto Blend Switch
export EXPERIMENTAL_ENABLE_AUTOFUSE=1
export EXPERIMENTAL_LOWERING_REDUCE=1
export EXPERIMENTAL_LOWERING_CONCAT=1
#GE dump graph switch
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
export DUMP_GRAPH_PATH=/home/dump_graph
```

If a model is saved in save model format, you can use the msit debug dump tool to capture inference data on the NPU. The following command is an example:`msit debug dump -m /home/mmoe_model -dp npu -i /home/input_float32.bin -is "input:1,128" -o /home/dump_data/npu`If the model is not saved in save model format, you can enable dump by configuring sessions. The following provides an example:

```
import tensorflow.compat.v1 as tf
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

### 3.2 Dumping Model Data on the CPU/GPU ###

You can use the fetches parameter in the tfdbg or session.run function to dump data. Note that to ensure the validity of the precision comparison result, the model input must be the same. The following provides an example of using fetches for dump:

```
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

input_file = "/home/input_float32.bin"
dump_path = "/home/dump_data/cpu/"
model_path = "/home/frozen_graph.pb"
input_dtype = np.float32
input_shape = [1, 128]

#load model input data
input_data = np.fromfile(input_file, dtype=input_dtype).reshape(input_shape)

#get all node name
node_names = []
data = open(model_path, "rb").read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(data)
for node in graph_def.node:
    node_names.append(node.name)

#session run model
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
    feeds = dict()
    feeds['input:0'] = input_data
    with tf.Session(graph=graph) as sess:
        to_sess_run = [graph.get_tensor_by_name(name + ':0') for name in node_names]
        results = sess.run(to_sess_run, feed_dict=feeds)

        #save dump data
        name_replace = 0
        for idx, node_name in enumerate(node_names):
            tensor = to_sess_run[idx]
            result = results[idx]
            output_node_name = node_name.replace("/", "_") + ".0." + str(int(time.time())) + ".npy"
            save_name = dump_path + output_node_name
            if len(output_node_name) > 210:
                #The flush file command must comply with the {op_name}.{output_index}.{timestamp}.npy format.
                replace_file_name = str(name_replace) + ".0." + str(int(time.time())) + ".npy"
                save_name = dump_path + replace_file_name
                print("dump file name: ", output_node_name, " replace to ", str(replace_file_name))
                name_replace += 1
            np.save(save_name, result)
```

In the preceding example, the loaded model is saved in FrozenGraph format. The difference between FrozenGraph and Save Model format is that the model weight variables in FrozenGraph are frozen and converted into constants. The following is a reference script for model conversion.

```
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

#Load the SavedModel. Currently, the API of tf2 is used.
saved_model_dir = '/home/mmoe_model'
loaded_model = tf.saved_model.load(saved_model_dir)
infer = loaded_model.signatures['serving_default']
#Convert a variable to a constant to generate a frozen ConcreteFunction.
frozen_func = convert_variables_to_constants_v2(infer)

frozen_graph_def = frozen_func.graph.as_graph_def()
with tf.io.gfile.GFile('frozen_graph.pb', 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())
```

## 4. Execution precision comparison ##

 *  Use the ATC tool to convert the dumped GE diagram to a JSON file,`atc --mode=5 --om={ge_proto_0001_graph_1_build.txt所在路径} --json={转换后文件保存路径}`
 *  Execution`msit debug compare -gp [3.2中数据所在文件] --mp [3.1中数据所在文件] --ops-json [atc转换后json文件路径]`

### Parameter Description ###

| Parameter name     | Description                                                                                                       | Mandatory. |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- | ---------- |
| --golden-path, -gp | Root path of CPU/GPU dump data                                                                                    | Yes        |
| --my-path, -mp     | NPU Dump Data Root Path                                                                                           | Yes        |
| --ops-json         | Enable the automatic convergence and optimized GE dump file.                                                      | Yes        |
| --output, -o       | Path for storing the CSV file of precision comparison results. By default, the file is saved in the current path. | No.        |

