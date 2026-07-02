# Introduction to the Acceleration Library Online Inference Precision Comparison Tool #

## 1. Precision Problem Location Process ##

The general procedure for precision location is as follows:![精度定位流程图](./LocationProgress.png)	

The main steps are as follows: data set assessment > analysis of answer differences > analysis of tensor differences.

 *  Dataset evaluation is based on the performance dimension to evaluate the model precision. In the future, the benchmark will support the automatic dataset evaluation function.
 *  Answer difference analysis is based on the token dimension to evaluate the model precision. After finding the different sentences, analyze the number of words in the different sentences. In addition, compare the token output indicators matrix of the model to determine which token round has the problem in the inference result.
 *  This tool is used to analyze the tensor differences. After locating the problem caused by the inference result of the Kth token, you can perform the comparison by tensor by automatic mapping or semi-automatically setting the kth round of comparison to narrow down the scope.

## 2. Scenarios and Usage ##

### 2.1 Comparison Scenario ###

Currently, the acceleration library precision comparison tool is used in the following scenarios:

| Scenario Name | Scenario Description                                                                                                                                                                                                                                                                                                                                            |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scenario 1    | The pytorch-npu inference and acceleration library inference are in the same inference script. Developers usually use torch.allclose to compare the inference output of the acceleration library with that of the pytorch-npu inference output. However, the difference between the data in the acceleration library and torch-npu cannot be directly compared. |
| Scenario 2    | pytorch-npu and acceleration library inferenceIn the two inference scripts, you execute torch-npu inference and acceleration library inference respectively to obtain two copies of data. Use torch-npu as the benchmark data and compare the direct differences between the two benchmark data.                                                                |
| Scenario 3    | pytorch-gpu and acceleration library inferenceIn the two inference scripts, you execute torch-gpu inference and acceleration library inference respectively to obtain two copies of data. Use torch-gpu as the benchmark data and compare the direct differences between the two benchmark data.                                                                |
| Scenario 4    | Use Operation (Layer) replacement as the benchmark data and compare the difference between Layer (Model) replacement and the benchmark.                                                                                                                                                                                                                         |

### 2.2 Usage ###

The processing process of the precision comparison tool is as follows:

![1696728244401](./工具比对流程.png)	

The tool provides the dump_data interface, inserts code dump data into the model script, and then uses the msit debug compare aclcmp command to compare data. For details, see:[Instructions for inserting dump_data codes](../13_dump_and_compare/README.md)	

### 2.3 Special instructions ###

#### 2.3.1 Weight-based Automatic Operator Mapping and Comparison ####

For scenarios 2 and 3, all operators with weights are provided by the tool.[Automatic mapping and comparison feature](../12_pta_acl_cmp_weight_map/README.md)	. You do not need to manually set the operator mapping. For some operators that do not have weights, you can use the[Insert and compare dump_data codes.](../13_dump_and_compare/README.md)	Method: Define the mapping relationship and compare the precision. The following figure shows the tool usage process.

![场景2和场景3流程图](./场景2和场景3流程图.png)	

#### 2.3.2 Set_label Code Insertion Mode ####

The tool provides the set_label code insertion mode, which is applicable to scenario 1. In an inference script, you can use the tool to start the inference process to perform inference to complete precision comparison.

For details, see:[Set_label Code Insertion Instructions](../11_pta_acl_cmp/basic_usage.md)	

### 2.4. AllReduce Communication Data Comparison ###

In the reasoning process of large models, weight segmentation enables large models to be inferred on different cards at the same time. The allreduce operator sums up the weights of different cards during communication. For example, when two cards are running, the precision comparison function is used to compare the weights of the allreduce communication operator. Use the sum of the intensors of card 0 and card 1 as the benchmark data to compare the precision with the outtensors of card 0 and card 1.

This tool is used to compare AllReduce communication data and compare the input and output errors of AllReduceHcclRunner at each layer of the model to narrow down the range.

#### 2.4.1 AllReduce Communication Data Comparison in the llama_parallel Scenario ####

According to the`pytorch/examples/llama_parallel/readme.md`Configure the environment and dump the data before and after the AllReduce operator.

```
export ATB_SAVE_TENSOR=1
export ATB_SAVE_TENSOR_START=0
export ATB_SAVE_TENSOR_END=10
export ATB_SAVE_TENSOR_RUNNER="AllReduceHcclRunner"
bash cut_model_and_run_llama.sh
```

After obtaining the data before and after AllReduce,`/llama_parallel/atb_temp/tensors/`Obtain the names of the two processes on the dual-SIM card in the directory and run the following command to invoke the processes:

```
export MSQUICKCMP_PATH=`python -c 'import os, msquickcmp; print(os.path.dirname(msquickcmp.__file__))'`
python $MSQUICKCMP_PATH/pta_acl_cmp/allreduce.py --process_0_path '/xxx/进程1/' --process_1_path '/xxx/进程2/' --output_path '生成csv路径'
```

#### 2.4.2 AllReduce Communication Data Comparison in the Chatglm2_6b Scenario ####

According to the`pytorch/examples/chatglm2_6b/ChatGLM2-6B量化推理指导.md`Configure the environment and dump the data before and after the AllReduce operator.

```
export ATB_SAVE_TENSOR=1
export ATB_SAVE_TENSOR_START=0
export ATB_SAVE_TENSOR_END=10
export ATB_SAVE_TENSOR_RUNNER="AllReduceHcclRunner"
bash run_quant_parallel.sh patches/models/modeling_chatglm2_6b_quant_mix_parallel_fa.py --evaluate_single
```

After obtaining the data before and after AllReduce,`/chatglm2_6b/atb_temp/tensors/`Obtain the names of the two processes on the dual cards in the directory and follow the instructions provided in.`2.4.1`Invoke the command.

#### 2.4.3 Comparison Result ####

 *  The comparison result is stored in the file.`allreduce_compare_result.csv`Medium
 *  The following is a brief description of the results:

| `allreduce`              | `cosine_similarity`                    | `max_relative_error`                        | `mean_relative_error` | `relative_euclidean_distance`                    |
| ------------------------ | -------------------------------------- | ------------------------------------------- | --------------------- | ------------------------------------------------ |
| Comparison Operator Name | Cosine similarity algorithm comparison | Maximum relative error algorithm comparison | mean relative error   | Euclidean relative distance algorithm comparison |

Output Result Reference[Procedure for Comparing Result Analysis](../result_analyse/README.md)	

