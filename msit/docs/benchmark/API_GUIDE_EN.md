# msit benchmark interface python API Usage Guide

## benchmark API Introduction

  The Python API provided by benchmark enables inference of offline models (.om models) based on Ascend hardware.<br>

To use the API provided by msit benchmark, you need to install the `ais_bench` and `aclruntime` packages. The installation method is as follows:

- Install the ais_bench package and aclruntime package separately as needed ([Installation Package Download Address](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench#%E5%B7%A5%E5%85%B7%E5%AE%89%E8%A3%85%E6%96%B9%E5%BC%8F)):

  ``` cmd
  # Install aclruntime
  pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
  # Install ais_bench
  pip3 install ./ais_bench-{version}-py3-none-any.whl
  # {version} indicates the software version number, {python_version} indicates the Python version number, and {arch} indicates the CPU architecture.
  ```

## interface python API Quick Start

### Overall Flowchart

![API Enabled Model Inference Flow](graphs/api_quick_start.png)

### Import Dependency Packages

```python
from ais_bench.infer.interface import InferSession
```

### Load Model

InferSession is the main class of the interface API in a single process. It is used to load the om model and execute om model inference. An instance of InferSession needs to be initialized before model inference.

```python
# The initialization of InferSession indicates loading the model model.om on the npu chip with device id 0
session = InferSession(device_id=0, model_path="model.om")
```

### Call Interface to Infer Model and Get Output

After creating the InferSession instance session, the configuration required for model inference on the npu chip is complete. You can then directly call the member function interface of session for model inference. The return value of the interface is the inference result.

```python
# feeds passes in a set of input data; mode selects the model type. static indicates a static model with fixed input node shape
# outputs is the tensor in ndarray format
outputs = session.infer(feeds=inputs, mode="static")
```

### Get Model Performance Data

After inference, the inference performance data is also saved in the session. You can obtain the performance data through the session interface.

```python
# exec_time_list preserves the time of all sessions executing inference in chronological order.
exec_time = session.summary().exec_time_list[-1]
```

### Release Model Memory

```python
session.free_resource()
```

## interface python API Detailed Introduction

### API Overview

|Number<td rowspan='1'>**Main File**</td><td rowspan='1'>**Main Class**</td><td rowspan='1'>**Interface Category**</td><td rowspan='1'>**Interface Name**</td>|
|----|
|1<td rowspan='18'>interface.py</td><td rowspan='9'>[InferSession](#InferSession1)</td><td rowspan='2'>Get Model Information</td><td rowspan='1'>[get_inputs](#get_inputs1)</td>|
|2<td rowspan='1'>[get_outputs](#get_outputs1)</td>|
|3<td rowspan='3'>Model Inference</td><td rowspan='1'>[infer](#infer1)</td>|
|4<td rowspan='1'>[infer_pipeline](#infer_pipeline1)</td>|
|5<td rowspan='1'>[infer_iteration](#infer_iteration1)</td>|
|6<td rowspan='2'>Get Inference Performance</td><td rowspan='1'>[summary](#summary1)</td>|
|7<td rowspan='1'>[reset_summaryinfo](#reset_summaryinfo1)</td>|
|8<td rowspan='2'>Release Model Resources</td><td rowspan='1'>[free_resource](#free_resource1)</td>|
|9<td rowspan='1'>[finalize](#finalize1)</td>|
|10<td rowspan='4'>[MultiDeviceSession](#MultiDeviceSession1)</td><td rowspan='3'>Model Inference</td><td rowspan='1'>[infer](#infer2)</td>|
|11<td rowspan='1'>[infer_pipeline](#infer_pipeline2)</td>|
|12<td rowspan='1'>[infer_iteration](#infer_iteration2)</td>|
|13<td rowspan='1'>Get Inference Performance</td><td rowspan='1'>[summary](#summary2)</td>|
|14<td rowspan='3'>[MemorySummary](#MemorySummary1)</td><td rowspan='3'>Resource Copy Time</td><td rowspan='1'>[get_h2d_time_list](#get_h2d_time_list1)</td>|
|15<td rowspan='1'>[get_d2h_time_list](#get_d2h_time_list1)</td>|
|16<td rowspan='1'>[reset](#reset1)</td>|

<a name="InferSession1"></a>

### InferSession

#### Class Prototype

```python
class InferSession(device_id: int, model_path: str, acl_json_path: str = None, debug: bool = False, loop: int = 1)
```

#### Class Description

InferSession is the class for om model inference in a **single process**.

#### Initialization Parameters

|Parameter|Description|Required|
|----|----|----|
|**device_id**|uint8, the id of the npu chip. Use `npu-smi info` on the server with CANN driver installed to view the available npu chip ids.|Yes|
|**model_path**|str, the path of the om model. Supports absolute paths and relative paths.|Yes|
|**acl_json_path**|str, the acl json file, used to configure profiling (collect detailed performance data during inference) and dump (collect input and output data of each operator in the model).|No|
|**debug**|bool, switch to display more detailed debug-level log information. True enables the switch.|No|
|**loop**|int, the number of times a set of input data is repeatedly inferred. Must be at least 1.|No|

<a name="get_inputs1"></a>

#### <font color="#DD4466">**get_inputs Function**</font>

**Function Description**

Used to get the input node information of the model loaded by InferSession.

**Function Prototype**

```python
get_inputs()
```

**Return Value**

Returns the input node attribute information of type <font color="#44AA00">list [[aclruntime.tensor_desc](#acl_tensor_desc)]</font>.

<a name="get_outputs1"></a>

#### <font color="#DD4466">**get_outputs Function**</font>

**Function Description**

Used to get the output node information of the model loaded by InferSession.

**Function Prototype**

```python
get_outputs()
```

**Return Value**

Returns the output node attribute information of type <font color="#44AA00">list [[aclruntime.tensor_desc](#acl_tensor_desc)]</font>. <br>
<a name="jump1"></a>

<a name="infer1"></a>

#### <font color="#DD4466">**infer Function**</font>

**Function Description**

Model inference interface. Infers a set of input data at a time. Supports models with static shape, dynamic batch, dynamic resolution, dynamic dims, and dynamic shape scenarios.

**Function Prototype**

```python
infer(feeds, mode='static', custom_sizes=100000, out_array=True)
```

**Parameter Description**

|Parameter|Description|Required|
|----|----|----|
|**feeds**|A set of input data required for inference. Supported data types:<a name="jump0"></a> <br> <ul>1. numpy.ndarray; <br> 2. single numpy type data (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.float16, np.float32, np.float64); <br> 3. torch type Tensor (torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor, torch.ByteTensor, torch.CharTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor, torch.IntTensor) <br> 4. [aclruntime.Tensor](#acl_Tensor) </ul>|Yes|
|**mode**|str, specifies the model type to load. Options: 'static' (static model), 'dynbatch' (dynamic batch model), 'dynhw' (dynamic resolution model), 'dyndims' (dynamic dims model), 'dynshape' (dynamic shape model).|No|
|**custom_sizes**|int or [int], required for dynamic shape models. The memory size occupied by the inference output data (in bytes).<br> <ul>1. When the input is int, each output of the model is pre-allocated with custom_sizes memory.<br> 2. When the input is list:[int], each output of the model is pre-allocated with the corresponding element size in custom_sizes.</ul>|No|
|**out_array**|bool, whether to transfer the model inference result from the device side to the host side.|No|

**Return Value**

+ When out_array == True, returns the inference output result of type numpy.ndarray. The data is on the host side.
+ When out_array == False, returns the inference output result of type <font color="#44AA00">[aclruntime.Tensor](#acl_Tensor)</font>. The data is on the device side.

<a name="jump3"></a> <a name="infer_pipeline1"></a>

#### <font color="#DD4466">**infer_pipeline Function**</font>

**Function Description**

Multi-thread inference interface (computation and data transfer in different threads). It is recommended to use this interface when inferring multiple sets of data at once. Compared with calling the `infer` interface multiple times to infer multiple sets of data, it can effectively shorten the end-to-end time.

**Function Prototype**

```python
infer_pipeline(feeds_list, mode = 'static', custom_sizes = 100000)
```

**Parameter Description**

|Parameter|Description|Required|
|----|----|----|
|**feeds_list**|list, several sets of input data for inference. Supported data types in the list:<a name="jump2"></a>: <br> <ul>1. numpy.ndarray; <br> 2. single numpy type data (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.float16, np.float32, np.float64); <br> 3. torch type Tensor (torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor, torch.ByteTensor, torch.CharTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor, torch.IntTensor) <br> 4. [aclruntime.Tensor](#acl_Tensor) </ul><b>Note:</b><br> <ul>1. In 'static', 'dynbatch', and 'dynhw' scenarios, each feeds in feeds_list must have the same shape. <br> 2. In 'dyndims' and 'dynshape' scenarios, each feeds in feeds_list can have different shapes.</ul>|Yes|
|**mode**|str, specifies the model type to load. Options: 'static' (static model), 'dynbatch' (dynamic batch model), 'dynhw' (dynamic resolution model), 'dyndims' (dynamic dims model), 'dynshape' (dynamic shape model).|No|
|**custom_sizes**|int or [int], required for dynamic shape models. The memory size occupied by the inference output data (in bytes).<br><ul>1. When the input is int, each output of the model is pre-allocated with custom_sizes memory.<br>2. When the input is list:[int], each output of the model is pre-allocated with the corresponding element size in custom_sizes.</ul>|No|

- **Return Value**

Returns the inference output result of type list:[numpy.ndarray]. The data is on the host side.

<a name="jump5"></a> <a name="infer_iteration1"></a>

#### <font color="#DD4466">**infer_iteration Function**</font>

**Function Description**

Iterative inference interface. Iterative inference (loop inference) means that the input data for the next inference partially comes from the output data of the previous inference. Compared with calling the `infer` interface in a loop for iterative inference, this interface can shorten the end-to-end time.

**Function Prototype**

```python
infer_iteration(feeds, in_out_list = None, iteration_times = 1, mode = 'static', custom_sizes = 100000, mem_copy = True)
```

**Parameter Description**

|Parameter|Description|Required|
|----|----|----|
|**feeds**|A set of input data required for inference. Supported data types: <a name="jump4"></a> <br> <ul>1. numpy.ndarray; <br> 2. single numpy type data (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.float16, np.float32, np.float64); <br> 3. torch type Tensor (torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor, torch.BFloat16Tensor, torch.ByteTensor, torch.CharTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor, torch.IntTensor) <br> </ul> |Yes|
|**in_out_list**|[int], indicates which output each input of the model comes from in each iteration. The order of inputs and outputs is consistent with the element order in the list obtained by `get_inputs()` and `get_outputs()`. For example, [-1, 1, 0] means the first input data reuses the original input data (represented by -1), the second input data comes from the second output data, and the third input comes from the first output data.|Yes|
|**iteration_times**|int, the number of iterations.|No|
|**mode**|str, specifies the model type to load. Options: 'static' (static model), 'dynbatch' (dynamic batch model), 'dynhw' (dynamic resolution model), 'dyndims' (dynamic dims model), 'dynshape' (dynamic shape model).|No|
|**custom_sizes**|int or [int], required for dynamic shape models. The memory size occupied by the inference output data (in bytes).<br><ul> 1. When the input is int, each output of the model is pre-allocated with custom_sizes memory.<br> 2. When the input is list:[int], each output of the model is pre-allocated with the corresponding element size in custom_sizes.</ul>|No|
|**mem_copy**|bool, determines whether the input data in iterative inference uses the output data of the previous inference by copying.<br><ul> 1. mem_copy == True: uses copying. The underlying acl interface will not report errors after inference. The inference result is correct.<br> 2. mem_copy == False: uses memory sharing. The underlying acl interface may report errors after inference (when plog is enabled). The inference result is correct. The inference end-to-end time is shorter.</ul>|No|

- **Return Value**

Returns the inference output result of type numpy.ndarray. The data is on the host side.

<a name="summary1"></a>

#### <font color="#DD4466">**summary Function**</font>

**Function Description**

Used to get the performance data of the inference process.

**Function Prototype**

```python
summary()
```

**Return Value**

Returns data of type [float]. The returned list saves the inference time of each set of data in the order of inference execution.

<a name="reset_summaryinfo1"></a>

#### <font color="#DD4466">**reset_summaryinfo Function**</font>

**Function Description**

Used to clear the performance data obtained by `summary()`.

**Function Prototype**

```python
reset_summaryinfo()
```

**Return Value**

None

<a name="free_resource1"></a>

#### <font color="#DD4466">**free_resource Function**</font>

**Function Description**

Used to release the device-side resources related to InferSession, but does not release other AscendCL-related resources in the process where InferSession is located on the corresponding device of InferSession.

**Function Prototype**

```python
free_resource()
```

**Return Value**

None

<a name="finalize1"></a>

#### <font color="#DD4466">**finalize Function**</font>

**Function Description**

Used to release all AscendCL-related resources in the process where InferSession is located on the corresponding device of InferSession.

**Function Prototype**

```python
finalize()
```

**Return Value**

None

<a name="MultiDeviceSession1"></a>

### MultiDeviceSession

#### Class Prototype

```python
class MultiDeviceSession(model_path: str, acl_json_path: str = None, debug: bool = False, loop: int = 1)
```

#### Class Description

MultiDeviceSession is the class for om model inference in **multi-process**. During initialization, the model is not loaded on the npu chip (device). An InferSession is created in each process of the specified devices when the inference interface is used.

#### Initialization Parameters

|Parameter|Description|Required|
|----|----|----|
|**model_path**|str, the path of the om model. Supports absolute paths and relative paths.|Yes|
|**acl_json_path**|str, the acl json file, used to configure profiling (collect detailed performance data during inference) and dump (collect input and output data of each operator in the model).|No|
|**debug**|bool, switch to display more detailed debug-level log information. True enables the switch.|No|
|**loop**|int, the number of times a set of input data is repeatedly inferred. Must be at least 1.|No|

<a name="infer2"></a>

#### <font color="#DD4466">**infer Function**</font>

**Function Description**

Calls the [infer interface](#jump1) of InferSession in multi-process for inference.

**Function Prototype**

```python
infer(devices_feeds, mode='static', custom_sizes=100000)
```

**Parameter Description**

|Parameter|Description|Required|
|----|----|----|
|**devices_feeds**|dict, {device_id: [feeds1, feeds2, ...]}. Each feeds in the device corresponding to device_id will start a separate process for inference. For the definition of feeds, refer to [the definition of feeds in the infer interface of InferSession](#jump0).|Yes|
|**mode**|str, specifies the model type to load. Options: 'static' (static model), 'dynbatch' (dynamic batch model), 'dynhw' (dynamic resolution model), 'dyndims' (dynamic dims model), 'dynshape' (dynamic shape model).|No|
|**custom_sizes**|int or [int], required for dynamic shape models. The memory size occupied by the inference output data (in bytes).<br><ul> 1. When the input is int, each output of the model is pre-allocated with custom_sizes memory.<br> 2. When the input is list:[int], each output of the model is pre-allocated with the corresponding element size in custom_sizes.</ul>|No|

**Return Value**
Returns {device_id:[output1, output2, ...]}. output* is the inference output result of type numpy.ndarray. The data is on the host side.

<a name="infer_pipeline2"></a>

#### <font color="#DD4466">**infer_pipeline Function**</font>

**Function Description**

Calls the [infer_pipeline interface](#jump3) of InferSession in multi-process for inference.

**Function Prototype**

```python
infer_pipeline(devices_feeds_list, mode = 'static', custom_sizes = 100000)
```

**Parameter Description**

|Parameter|Description|Required|
|----|----|----|
|**devices_feeds_list**|dict, {device_id: [feeds_list1, feeds_list2, ...]}. Each feeds_list in the device corresponding to device_id will start a separate process for inference. For the definition of feeds_list, refer to [the definition of feeds_list in the infer_pipeline interface of InferSession](#jump2).|Yes|
|**mode**|str, specifies the model type to load. Options: 'static' (static model), 'dynbatch' (dynamic batch model), 'dynhw' (dynamic resolution model), 'dyndims' (dynamic dims model), 'dynshape' (dynamic shape model).|No|
|**custom_sizes**|int or [int], required for dynamic shape models. The memory size occupied by the inference output data (in bytes).<ul><br> 1. When the input is int, each output of the model is pre-allocated with custom_sizes memory.<br> 2. When the input is list:[int], each output of the model is pre-allocated with the corresponding element size in custom_sizes.</ul>|No|

**Return Value**
Returns {device_id:[output1, output2, ...]}. output* is the inference output result of type [numpy.ndarray]. The data is on the host side.

<a name="infer_iteration2"></a>

#### <font color="#DD4466">**infer_iteration Function**</font>

**Function Description**

Calls the [infer_iteration interface](#jump5) of InferSession in multi-process for inference.

**Function Prototype**

```python
infer_iteration(devices_feeds, in_out_list = None, iteration_times = 1, mode = 'static', custom_sizes = None, mem_copy = True)
```

**Parameter Description**

|Parameter|Description|Required|
|----|----|----|
|**devices_feeds**|dict, {device_id: [feeds1, feeds2, ...]}. Each feeds in the device corresponding to device_id will start a separate process for inference. For the definition of feeds, refer to [the definition of feeds in the infer_iteration interface of InferSession](#jump4).|Yes|
|**in_out_list**|[int], indicates which output each input of the model comes from in each iteration. The order of inputs and outputs is consistent with the element order in the list obtained by `get_inputs()` and `get_outputs()`. For example, [-1, 1, 0] means the first input data reuses the original input data (represented by -1), the second input data comes from the second output data, and the third input comes from the first output data.|Yes|
|**iteration_times**|int, the number of iterations.|No|
|**mode**|str, specifies the model type to load. Options: 'static' (static model), 'dynbatch' (dynamic batch model), 'dynhw' (dynamic resolution model), 'dyndims' (dynamic dims model), 'dynshape' (dynamic shape model).|No|
|**custom_sizes**|int or [int], required for dynamic shape models. The memory size occupied by the inference output data (in bytes).<ul><br> 1. When the input is int, each output of the model is pre-allocated with custom_sizes memory.<br> 2. When the input is list:[int], each output of the model is pre-allocated with the corresponding element size in custom_sizes.</ul>|No|
|**mem_copy**|bool, determines whether the input data in iterative inference uses the output data of the previous inference by copying.<ul><br> 1. mem_copy == True: uses copying. The underlying acl interface will not report errors after inference. The inference result is correct.<br>2. mem_copy == False: uses memory sharing. The underlying acl interface may report errors after inference (when plog is enabled). The inference result is correct. The inference end-to-end time is shorter.</ul>|No|

**Return Value**

Returns {device_id:[output1, output2, ...]}. output* is the inference output result of type numpy.ndarray. The data is on the host side.

<a name="summary2"></a>

#### <font color="#DD4466">**summary Function**</font>

**Function Description**

Gets the end-to-end inference time (including model loading time) of the most recent multi-process inference interface call.

**Function Prototype**

```python
summary()
```

**Return Value**

Returns {device_id:[e2etime1, e2etime2, ...]}. e2etime* is the end-to-end inference time of each process (including model loading time).

<a name="MemorySummary1"></a>

### MemorySummary

#### Class Prototype

```python
MemorySummary()
```

#### Class Description

MemorySummary is used to collect the copy time of the host2device and device2host processes in an inference process.

<a name="get_h2d_time_list1"></a>

#### <font color="#DD4466">**get_h2d_time_list Function**</font>

**Function Description**

Gets all host2device copy times in the entire process.

**Function Prototype**

```python
get_h2d_time_list()
```

**Return Value**

Returns data of type [float]. The times in the returned list are sorted by the order of inference execution.

<a name="get_d2h_time_list1"></a>

#### <font color="#DD4466">**get_d2h_time_list Function**</font>

**Function Description**

Gets all device2host copy times in the entire process.

**Function Prototype**

```python
get_d2h_time_list()
```

**Return Value**

Returns data of type [float]. The times in the returned list are sorted by the order of inference execution.

<a name="reset1"></a>

#### <font color="#DD4466">**reset Function**</font>

**Function Description**

Used to clear the data obtained by `get_h2d_time_list` and `get_d2h_time_list`.

**Function Prototype**

```python
reset()
```

**Return Value**

None

### Internal Data Type Explanation

<a name="acl_tensor_desc"></a>

#### <font color="#DD4466">**aclruntime.tensor_desc**</font>

Structure describing model input and output node information:<br>

- property <font color="#DD4466">**name**</font>:str
    + Node name.
- property <font color="#DD4466">**datatype**</font>:[aclruntime.dtype](#acl_dtype)
    + The data type of the tensor accepted by the node.
- property <font color="#DD4466">**format**</font>:int
    + The tensor format accepted by the node. 0 indicates NCHW format, and 1 indicates NHWC format.
- property <font color="#DD4466">**shape**</font>:list [int]
    + The shape of the tensor accepted by the node.
- property <font color="#DD4466">**size**</font>:int
    + The size of the tensor accepted by the node.
- property <font color="#DD4466">**realsize**</font>:int
    + The actual size of the tensor accepted by the node. The actual required size for dynamic shape and dynamic grading scenarios.

<a name="acl_dtype"></a>

#### <font color="#DD4466">**aclruntime.dtype**</font>(enum)

An enumeration type of data type names:<br>

- Includes 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'float16', 'float32', 'float64', 'bool'

<a name="acl_Tensor"></a>

#### <font color="#DD4466">**aclruntime.Tensor**</font>

- The way tensors are stored on the device side. Cannot be accessed directly on the host side.
