# Save Output Data #

## Introduction ##

By default, the benchmark inference tool does not save the output result data file after execution. After related parameters are set, the following results can be generated:

| File/Directory                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| \{filename\}.bin, \{filename\}.npy, or \{filename\}.txt | Model inference output file. File naming format: name_output sequence number. Suffix. If input is not specified (pure inference), the name is fixed to pure_infer_data. If input is specified, the first input name is used as the first input name. The output sequence number starts from 0 and is arranged in the output sequence. The file name extension is specified by the --outfmt parameter. By default, a directory named Date+Time is created in the directory specified by the --output parameter and the result file is saved in the directory. If --output-dirname is specified, the result file is stored in the directory specified by --output-dirname. If the --output-dirname parameter is specified, multiple executions of tool inference will cause the result file to be overwritten by the same name. |
| xx_summary.json                                         | The tool outputs the model performance result data. By default, xx is named in the format of date+time. When --output-dirname is specified, "xx" is named after the directory name specified by --output-dirname. If the --output-dirname parameter is specified, multiple executions of tool inference will cause the result file to be overwritten by the same name.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| dump                                                    | Directory for storing dump data files. When the --dump command is used to enable dump, create a dump directory in the directory specified by the --output parameter to save dump data files.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| profiler                                                | Directory for storing performance data files collected by the Profiler. When the --profiler command is used to enable performance data collection, create the profiler directory in the directory specified by the --output parameter to save performance data files.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

## Running an Example ##

1.  Set only the --output parameter. The following is an example of the command output:
    
    ```
    msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result
    ```
    
    ```
    result
    |-- 2022_12_17-07_37_18
    │   |-- pure_infer_data_0.bin
    |-- 2022_12_17-07_37_18_summary.json
    ```
2.  Set the --input and --output parameters. The following is an example of the command output:
    
    ```
    #The contents of the input folder are as follows:
    ls ./data
    196608-0.bin  196608-1.bin  196608-2.bin  196608-3.bin  196608-4.bin  196608-5.bin  196608-6.bin  196608-7.bin  196608-8.bin  196608-9.bin
    ```
    
     *  Note: The .bin file stores the tensor data entered by the user and can be generated in the following way. In the example, size and astype can be obtained by using the debug tool. The --input parameter is designed for users to specify input data.
        
        ```
        import numpy as np
        np.random.uniform(size=[32,32]).astype('float32').tofile('foo.bin')
        ```
    
    ```
    msit benchmark --om-model ./pth_resnet50_bs1.om --input ./data --output ./result
    ```
    
    ```
    result/
    |-- 2023_01_03-06_35_53
    |   |-- 196608-0_0.bin
    |   |-- 196608-1_0.bin
    |   |-- 196608-2_0.bin
    |   |-- 196608-3_0.bin
    |   |-- 196608-4_0.bin
    |   |-- 196608-5_0.bin
    |   |-- 196608-6_0.bin
    |   |-- 196608-7_0.bin
    |   |-- 196608-8_0.bin
    |   |-- 196608-9_0.bin
    |-- 2023_01_03-06_35_53_summary.json
    ```
3.  Set the --output-dirname parameter. The following is an example of the command output:
    
    ```
    msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result --output-dirname subdir
    ```
    
    ```
    result
    |-- subdir
    │   |-- pure_infer_data_0.bin
    |-- subdir_summary.json
    ```
4.  Set the --dump parameter. The following is an example of the command output:
    
    ```
    msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result --dump 1
    ```
    
    ```
    result
    |-- 2022_12_17-07_37_18
    │   |-- pure_infer_data_0.bin
    |-- dump
    |-- 2022_12_17-07_37_18_summary.json
    ```
5.  Set the --profiler parameter. The following is an example of the command output:
    
    ```
    msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result --profiler 1
    ```
    
    ```
    result
    |-- 2022_12_17-07_56_10
    │   |-- pure_infer_data_0.bin
    |-- profiler
    │   |-- PROF_000001_20221217075609326_GLKQJOGROQGOLIIB
    |-- 2022_12_17-07_56_10_summary.json
    ```
6.  Output result explanation.

After the benchmark inference tool is executed, the output is as follows:

 *  When --display-all-summary is set to False, the following information is displayed:
    
    ```
    [INFO] -----------------Performance Summary------------------
    [INFO] NPU_compute_time (ms): min = 0.6610000133514404, max = 0.6610000133514404, mean = 0.6610000133514404, median = 0.6610000133514404, percentile(99%) = 0.6610000133514404
    [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(0.6610000133514404): 1512.8592735267011
    [INFO] ------------------------------------------------------
    ```
 *  When --display-all-summary is set to True, the following information is displayed:
    
    ```
    [INFO] -----------------Performance Summary------------------
    [INFO] H2D_latency (ms): min = 0.05700000002980232, max = 0.05700000002980232, mean = 0.05700000002980232, median = 0.05700000002980232, percentile(99%) = 0.05700000002980232
    [INFO] NPU_compute_time (ms): min = 0.6650000214576721, max = 0.6650000214576721, mean = 0.6650000214576721, median = 0.6650000214576721, percentile(99%) = 0.6650000214576721
    [INFO] D2H_latency (ms): min = 0.014999999664723873, max = 0.014999999664723873, mean = 0.014999999664723873, median = 0.014999999664723873, percentile(99%) = 0.014999999664723873
    [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(0.6650000214576721): 1503.759349974173
    ```

You can view the model execution duration and throughput based on the output results. A smaller duration and a higher throughput indicate higher performance of the model.

**Field Description**

| Field                 | Description                                                                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| H2D_latency (ms)      | Memory copy from host to device takes a long time. The unit is ms.                                                                                                 |
| min                   | Minimum inference execution time.                                                                                                                                  |
| max                   | Maximum inference execution time.                                                                                                                                  |
| mean                  | Indicates the average inference execution time.                                                                                                                    |
| median                | The median inference execution time is used.                                                                                                                       |
| percentile(99%)       | The percentile in the inference execution time.                                                                                                                    |
| NPU_compute_time (ms) | NPU inference calculation time. The unit is ms.                                                                                                                    |
| D2H_latency (ms)      | Time required for memory copy from Device to Host. The unit is ms.                                                                                                 |
| throughput            | Throughput. Throughput calculation formula: 1000 \*batchsize/npu_compute_time.mean                                                                                 |
| batchsize             | Batch size. This tool may not accurately identify the batch size of the current sample. You are advised to set the batch size by using the --batch-size parameter. |

## FAQ ##

If a problem occurs, refer to.[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)	

