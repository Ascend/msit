# msit benchmark interface python API User Guide

## Benchmark API Overview

The Python API provided by the benchmark can be used to enable offline model (.om model) inference based on Ascend hardware.

The API provided by the msit benchmark must be installed.`ais_bench`And with the`aclruntime`Package. Installation methods are:

 1. Reference[Integrated Installation Guide](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md) Installing the msit benchmark tool

 2. Install the ais_bench and aclruntime packages as required.[Path for Obtaining the Installation Package](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench#%E5%B7%A5%E5%85%B7%E5%AE%89%E8%A3%85%E6%96%B9%E5%BC%8F):
    
    ```cmd
    #Installing the aclruntime
    pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
    #Installing ais_bench
    pip3 install ./ais_bench-{version}-py3-none-any.whl
    #{version} indicates the software version, {python_version} indicates the Python version, and {arch} indicates the CPU architecture.
    ```

## Interface Python API Quickly Get Started

### Overall flow chart

![API使能模型推理流程](graphs/api_quick_start.png)    

### Importing the Dependency Package

```python
from ais_bench.infer.interface import InferSession
```

### Load Model

InferSession is the main class of the interface API in single-process mode. It is used to load the OM model and perform OM model inference. Before model inference, you need to initialize an InferSession instance.

```python
#InferSession initialization indicates that the model.om is loaded to the NPU chip whose device ID is 0.
session = InferSession(device_id=0, model_path="model.om")
```

### Invoke the interface to obtain the output of the inference model

After the InferSession instance session is created, all configurations required for model inference on the NPU chip are complete. Then, you can directly invoke the member function interface of the session to perform model inference. The return value of the interface is the inference result.

```python
#Feeds transfers a set of input data. mode: model type. static indicates a static model with a fixed input node shape.
#outputs are tensors in the ndarray format.
outputs = session.infer(feeds=inputs, mode="static")
```

### Performance of obtaining model data

After the inference is complete, the inference performance data is stored in the session. You can obtain the performance data through the session interface.

```python
#exec_time_list keeps the inference time of all sessions in sequence.
exec_time = session.summary().exec_time_list[-1]
```

### Release the memory occupied by the model

```python
session.free_resource()
```

## Interface Python API Description

### API Guided Tour

|No.<td rowspan='1'>**Main File**</td><td rowspan='1'>**Main Category**</td><td rowspan='1'>**Interface Category**</td><td rowspan='1'>**Interface name**</td>|
|----|
|1<td rowspan='18'>interface.py</td><td rowspan='9'>[InferSession](#infersession)</td><td rowspan='2'>Obtaining Model Information</td><td rowspan='1'>[get_inputs](#get_inputs-function)</td>|
|2<td rowspan='1'>[get_outputs](#get_outputs-function)</td>|
|3<td rowspan='3'>Perform model inference</td><td rowspan='1'>[infer](#infer-function)</td>|
|4<td rowspan='1'>[infer_pipeline](#infer_pipeline-function)</td>|
|5<td rowspan='1'>[infer_iteration](#infer_iteration-function)</td>|
|6<td rowspan='2'>Obtaining the Inference Performance</td><td rowspan='1'>[summary](#summary-function)</td>|
|7<td rowspan='1'>[reset_summaryinfo](#reset_summaryinfo-function)</td>|
|8<td rowspan='2'>Release model resources</td><td rowspan='1'>[free_resource](#free_resource-function)</td>|
|9<td rowspan='1'>[finalize](#finalize-function)</td>|
|10<td rowspan='4'>[MultiDeviceSession](#multidevicesession)</td><td rowspan='3'>Perform model inference</td><td rowspan='1'>[infer](#infer-function-1)</td>|
|11<td rowspan='1'>[infer_pipeline](#infer_pipeline-function-1)</td>|
|12<td rowspan='1'>[infer_iteration](#infer_iteration-function-1)</td>|
|13<td rowspan='1'>Obtaining the Inference Performance</td><td rowspan='1'>[summary](#summary-function-1)</td>|
|14<td rowspan='3'>[MemorySummary](#memorysummary)</td><td rowspan='3'>Resource Copy Time</td><td rowspan='1'>[get_h2d_time_list](#get_h2d_time_list-function)</td>|
|15<td rowspan='1'>[get_d2h_time_list](#get_d2h_time_list-function)</td>|
|16<td rowspan='1'>[reset](#reset-function)</td>|

### InferSession

#### class prototype

```python
class InferSession(device_id: int, model_path: str, acl_json_path: str = None, debug: bool = False, loop: int = 1)
```

#### Class Description

InferSession is a class used for OM model inference in a single process.

#### Initialization parameter

| Parameter name    | Description                                                                                                                                                                                    | Mandatory or not |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **device_id**     | uint8: ID of the NPU chip, which is used on the server where the CANN driver is installed.`npu-smi info`Query the IDs of the available NPU chips.                                              | Yes              |
| **model_path**    | str: path of the OM model, which can be an absolute path or a relative path.                                                                                                                   | Yes              |
| **acl_json_path** | str: ACL JSON file, used to configure the profiling (collecting detailed performance data during inference) and dump (collecting input and output data of operators at each layer of a model). | No.              |
| **debug**         | bool: indicates whether to display more detailed debug log information. True indicates that the function is enabled.                                                                           | No.              |
| **loop**          | int: number of times that a group of input data repeats the inference. The value must be at least 1.                                                                                           | No.              |

#### **get_inputs function**

**Function Description**

Obtains the input node information of the model loaded by InferSession.

**Function prototype**

```python
get_inputs()
```

**Return Value**

The return type islist \[[aclruntime.tensor_desc](#aclruntimetensor_desc)    \]Input node attribute information.

#### **get_outputs function**

**Function Description**

Obtains the output node information of the model loaded by InferSession.

**Function prototype**

```python
get_outputs()
```

**Return Value**

The return type islist \[[aclruntime.tensor_desc](#aclruntimetensor_desc)    \]Output node attribute information of the.

#### **infer function**

**Function Description**

This API is used to infer a group of input data at a time. It can infer models in the static shape, dynamic batch, dynamic resolution, dynamic dims, and dynamic shape scenarios.

**Function prototype**

```python
infer(feeds, mode='static', custom_sizes=100000, out_array=True)
```

**Parameter Description**

| Parameter name   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Mandatory or not |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **feeds**        | A set of input data required for inference. The following data types are supported:<br/><br/>1. numpy.ndarray; 2. (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.float16, np.float32, np.float64); 3 for a single Numpy data, Tensor(torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor, torch.ByteTensor, torch.CharTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor, torch.IntTensor) 4 for torch data,[aclruntime.Tensor](#aclruntimetensor)     | Yes              |
| **mode**         | str: specifies the type of the model to be loaded. The options are static (static model), dynbatch (dynamic batch model), dynhw (dynamic resolution model), dyndims (dynamic dims model), and dynshape. '(dynamic shape model)                                                                                                                                                                                                                                                                                 | No.              |
| **custom_sizes** | int or \[int\]: size of the memory occupied by the inference output data, in bytes, required by the dynamic shape model.<br/><br/>1. When the input is int, each output of the model is pre-allocated with the memory of custom_sizes. 2. When the input is list:\[int\], each output of the model is pre-allocated with the memory of the corresponding element size in custom_sizes.                                                                                                                         | No.              |
| **out_array**    | bool: indicates whether to transfer the model inference result from the device to the host.                                                                                                                                                                                                                                                                                                                                                                                                                    | No.              |

**Return Value**

 * out_array == True: Returns the inference output of the numpy.ndarray type. The data is stored on the host.
 * out_array == False, Return[aclruntime.Tensor](#aclruntimetensor)    Type inference output. The data is stored on the device side.

#### **infer_pipeline function**

**Function Description**

Multi-thread inference interface (computing and data transfer are in different threads). It is recommended that this interface be used to infer multiple groups of data at a time.`infer`Interface inference of multiple groups of data can effectively shorten the end-to-end time.

**Function prototype**

```python
infer_pipeline(feeds_list, mode = 'static', custom_sizes = 100000)
```

**Parameter Description**

| Parameter name   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Mandatory or not |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **feeds_list**   | list: indicates the groups of input data required for inference. The list supports the following data types:\:<br/><br/>1. numpy.ndarray; 2. (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.float16, np.float32, np.float64); 3 for a single Numpy data, Tensor(torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor, torch.ByteTensor, torch.CharTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor, torch.IntTensor) 4 for torch data,[aclruntime.Tensor](#aclruntimetensor)    <br/><br/>**Note:**<br/><br/>1. In the static, dynbatch, and dynhw scenarios, the shape of each feed in the feeds_list must be the same. 2. In the dyndims and dynshape scenarios, the shape of each feed in the feeds_list must be the same. The shapes can be different. | Yes              |
| **mode**         | str: specifies the type of the model to be loaded. The options are static (static model), dynbatch (dynamic batch model), dynhw (dynamic resolution model), dyndims (dynamic dims model), and dynshape. '(dynamic shape model)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | No.              |
| **custom_sizes** | int or \[int\]: size of the memory occupied by the inference output data, in bytes, required by the dynamic shape model.<br/><br/>1. When the input is int, the memory of custom_sizes is allocated to each output of the model in advance. 2. When the input is list:\[int\], each output of the model is pre-allocated with the memory of the corresponding element size in custom_sizes.                                                                                                                                                                                                                                                                                                                                                                                                                         | No.              |

 * **Return Value**

The inference output of the list:\[numpy.ndarray\] type is returned. The data is stored on the host.

#### **infer_iteration function**

**Function Description**

Iterative inference interface. Iterative inference (circular inference) indicates that part of the input data of the next inference comes from the output data of the previous inference. Relative to a circular call`infer`This interface implements iterative inference, which shortens the end-to-end time.

**Function prototype**

```python
infer_iteration(feeds, in_out_list = None, iteration_times = 1, mode = 'static', custom_sizes = 100000, mem_copy = True)
```

**Parameter Description**

| Parameter name      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Mandatory or not |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **feeds**           | A set of input data required for inference. The following data types are supported:<br/><br/>1. numpy.ndarray; 2. (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.float16, np.float32, np.float64); 3 for a single numpy data and Tensor(torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor, torch.ByteTensor, torch.CharTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor, torch.IntTensor) for torch data                                                                 | Yes              |
| **in_out_list**     | \[int\] indicates the sequence number of the output from which the input of the model comes in each iteration. The sequence of the input and output is the same as that of the output.`get_inputs()`And to the`get_outputs()`The sequence of the elements in the obtained list is the same. For example, \[- 1, 1, 0\] indicates that the first input data multiplexes the original input data (represented by -- 1), the second input data is from the second output data, and the third input data is from the first output data.           | Yes              |
| **iteration_times** | int: number of iterations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | No.              |
| **mode**            | str: specifies the type of the model to be loaded. The options are static (static model), dynbatch (dynamic batch model), dynhw (dynamic resolution model), dyndims (dynamic dims model), and dynshape. '(dynamic shape model)                                                                                                                                                                                                                                                                                                                | No.              |
| **custom_sizes**    | int or \[int\]: size of the memory occupied by the inference output data (unit: byte) used by the dynamic shape model.<br/><br/>1. When the input is int, the memory of custom_sizes is allocated to each output of the model in advance. 2. When the input is list:\[int\], each output of the model is pre-allocated with the memory of the corresponding element size in custom_sizes.                                                                                                                                                     | No.              |
| **mem_copy**        | bool: indicates whether the input data in iterative inference uses the output data in the last inference.<br/><br/>1. If mem_copy is True, copy is used. After the inference is complete, the underlying ACL interface does not report an error and the inference result is correct. 2. If mem_copy is false, memory sharing is used. After the inference is complete, the underlying ACL interface may report an error (when the plog function is enabled), the inference result is correct, and the end-to-end inference time is shortened. | No.              |

 * **Return Value**

Returns the inference output of the numpy.ndarray type. The data is stored on the host.

#### **Summary function**

**Function Description**

Obtains the performance data of the inference process.

**Function prototype**

```python
summary()
```

**Return Value**

Returns data of the \[float\] type. The returned list stores the inference time of each group of data in the inference execution sequence.

#### **reset_summaryinfo function**

**Function Description**

Used to clear`summary()`Obtained performance data.

**Function prototype**

```python
reset_summaryinfo()
```

**Return Value**

None

#### **free_resource function**

**Function Description**

Releases device-side resources related to InferSession, but does not release other AscendCL-related resources in the process where InferSession is located in the device corresponding to InferSession.

**Function prototype**

```python
free_resource()
```

**Return Value**

None

#### **finalize function**

**Function Description**

Releases all AscendCL-related resources of the process where InferSession is located in the device corresponding to InferSession.

**Function prototype**

```python
finalize()
```

**Return Value**

None

### MultiDeviceSession

#### class prototype

```python
class MultiDeviceSession(model_path: str, acl_json_path: str = None, debug: bool = False, loop: int = 1)
```

#### Class Description

MultiDeviceSession is a class used for OM model inference in multi-process. During initialization, models are not loaded on the NPU chip (device). An InferSession is created in each process of the specified devices only when the inference API is used.

#### Initialization parameter

| Parameter name    | Description                                                                                                                                                                                | Mandatory or not |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| **model_path**    | str: path of the OM model, which can be an absolute path or a relative path.                                                                                                               | Yes              |
| **acl_json_path** | str: acl json file, used to configure profiling (collecting detailed performance data during inference) and dump (collecting input and output data of operators at each layer of a model). | No.              |
| **debug**         | bool: indicates whether to display more detailed debug log information. True indicates that the function is enabled.                                                                       | No.              |
| **loop**          | int: number of times that a group of input data repeats the inference. The value must be at least 1.                                                                                       | No.              |

#### **infer function**

**Function Description**

InferSession[Infer interface](#interface-python-api-description)    Reasoning

**Function prototype**

```python
infer(devices_feeds, mode='static', custom_sizes=100000)
```

**Parameter Description**

| Parameter name    | Description                                                                                                                                                                                                                                                                                                                                                                                            | Mandatory or not |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| **devices_feeds** | Each feed in the device corresponding to dict, \{device_id: \[feeds1, feeds2, ...\]\}, and device_id has an independent inference process. For details about the feed definition, see.[Definition of feeds in the infer interface of InferSession](#interface-python-api-description)                                                                                                                                                | Yes              |
| **mode**          | str: specifies the type of the model to be loaded. The options are static (static model), dynbatch (dynamic batch model), dynhw (dynamic resolution model), dyndims (dynamic dims model), and dynshape. '(dynamic shape model)                                                                                                                                                                         | No.              |
| **custom_sizes**  | int or \[int\]: size of the memory occupied by the inference output data, which is required by the dynamic shape model. Unit: byte.<br/><br/>1. When the input is int, the memory of custom_sizes is allocated to each output of the model in advance. 2. When the input is list:\[int\], each output of the model is pre-allocated with the memory of the corresponding element size in custom_sizes. | No.              |

**The return value is \{device_id:\[output1, output2, ...\]\}. output\* indicates the inference output of the numpy.ndarray type. The data is stored on the host.**

#### **infer_pipeline function**

**Function Description**

InferSession[infer_pipeline interface](#infer_pipeline-function)    To reason.

**Function prototype**

```python
infer_pipeline(devices_feeds_list, mode = 'static', custom_sizes = 100000)
```

**Parameter Description**

| Parameter name         | Description                                                                                                                                                                                                                                                                                                                                                                               | Mandatory or not |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **devices_feeds_list** | Each feeds_list of the device corresponding to dict, \{device_id: \[feeds_list1, feeds_list2, ...\]\}, and device_id has an independent inference process. For details about the definition of feeds_list, see.[Definition of feeds_list in the infer_pipeline interface of InferSession](#interface-python-api-description)    .                                                                                       | Yes              |
| **mode**               | str: specifies the type of the model to be loaded. The options are static (static model), dynbatch (dynamic batch model), dynhw (dynamic resolution model), dyndims (dynamic dims model), and dynshape. '(dynamic shape model)                                                                                                                                                            | No.              |
| **custom_sizes**       | int or \[int\]: size of the memory occupied by the inference output data (unit: byte) used by the dynamic shape model.<br/><br/>1. When the input is int, the memory of custom_sizes is allocated to each output of the model in advance. 2. When the input is list:\[int\], each output of the model is pre-allocated with the memory of the corresponding element size in custom_sizes. | No.              |

**The return value is \{device_id:\[output1, output2, ...\]\}, where output\* is the inference output result of the \[numpy.ndarray\] type. The data is stored on the host.**

#### **infer_iteration function**

**Function Description**

InferSession[infer_iteration interface](#infer_iteration-function)    To reason.

**Function prototype**

```python
infer_iteration(devices_feeds, in_out_list = None, iteration_times = 1, mode = 'static', custom_sizes = None, mem_copy = True)
```

**Parameter Description**

| Parameter name      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Optional or not |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| **devices_feeds**   | Each feed of the device corresponding to dict, \{device_id: \[feeds1, feeds2, ...\]\}, and device_id has an independent inference process. For details about the feed definition, see.[Definition of feeds in the infer_iteration interface of InferSession](#interface-python-api-description)    .                                                                                                                                                                                                                                                                             | Yes             |
| **in_out_list**     | \[int\] indicates the sequence number of the output from which the model input comes in each iteration. The sequence of the input and output is the same as that of the output.`get_inputs()`And to the`get_outputs()`The sequence of the elements in the obtained list is the same. For example, \[-1, 1, 0\] indicates that the first input data is multiplexed with the original input data (represented by --1), the second input data is derived from the second output data, and the third input data is derived from the first output data. | Yes             |
| **iteration_times** | int: number of iterations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | No.             |
| **mode**            | str: specifies the type of the model to be loaded. The options are static (static model), dynbatch (dynamic batch model), dynhw (dynamic resolution model), dyndims (dynamic dims model), and dynshape. '(dynamic shape model)                                                                                                                                                                                                                                                                                                                     | No.             |
| **custom_sizes**    | int or \[int\]: size of the memory occupied by the inference output data (unit: byte) used by the dynamic shape model.<br/><br/>1. When the input is int, each output of the model is pre-allocated with the memory of custom_sizes. 2. When the input is list:\[int\], each output of the model is pre-allocated with the memory of the corresponding element size in custom_sizes.                                                                                                                                                               | No.             |
| **mem_copy**        | bool, which determines whether to copy the input data in iterative inference to the output data of the last inference.<br/><br/>1. If mem_copy is True, copy is used. After the inference is complete, the underlying ACL interface does not report an error and the inference result is correct. 2. If mem_copy is false, memory sharing is used. After the inference is complete, the underlying ACL interface may report an error (when plog is enabled), the inference result is correct, and the end-to-end inference time is shortened.      | No.             |

**Return Value**

\{device_id:\[output1, output2, ...\]\} is returned. output\* indicates the inference output of the numpy.ndarray type. The data is stored on the host.

#### **Summary function**

**Function Description**

Obtains the latest end-to-end inference time (including the model loading time) obtained by using the multi-process inference interface.

**Function prototype**

```python
summary()
```

**Return Value**

\{device_id:\[e2etime1, e2etime2, ...\]\} is returned. e2etime\* indicates the end-to-end inference time (including the model loading time) of each process.

### MemorySummary

#### class prototype

```python
MemorySummary()
```

#### Class Description

MemorySummary is used to collect the copy time of the host2device and device2host processes in an inference process.

#### **get_h2d_time_list function**

**Function Description**

Obtain the copy time of all host2device processes in the entire process.

**Function prototype**

```python
get_h2d_time_list()
```

**Return Value**

Returns data of the \[float\] type. Time in the returned list, which is sorted by inference execution sequence.

#### **get_d2h_time_list function**

**Function Description**

Obtain the copy time of all device2host processes in the entire process.

**Function prototype**

```python
get_d2h_time_list()
```

**Return Value**

Returns data of the \[float\] type. Time in the returned list, which is sorted by inference execution sequence.

#### **Reset function**

**Function Description**

Used to clear`get_h2d_time_list`And to the`get_d2h_time_list`Obtained data.

**Function prototype**

```python
reset()
```

**Return Value**

None

### Internal Data Type Description

#### **aclruntime.tensor_desc**

Structure that describes the input and output node information of a model.

 * propertyname\:str
    
     * Node name.
 * propertydatatype\:[aclruntime.dtype](#aclruntimedtypeenum)    
    
     * Tensor data type received by the node.
 * propertyformat\:int
    
     * The node accepts the tensor format. The value 0 indicates the NCHW format, and the value 1 indicates the NHWC format.
 * propertyshape\:list \[int\]
    
     * Tensor shape received by the node.
 * propertysize\:int
    
     * Tensor size received by a node.
 * propertyrealsize\:int
    
     * Actual size of the tensor received by the node, which is required in the dynamic shape and dynamic ranking scenario.

#### **aclruntime.dtype**(enum)

Enumerated type of data type name:

 * Contains'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'float16', 'float32', 'float64', 'bool'

#### **aclruntime.Tensor**

 * Tensor storage mode on the device side cannot be directly accessed on the host side.
