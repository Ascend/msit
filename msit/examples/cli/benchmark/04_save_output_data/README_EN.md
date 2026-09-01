# Save Output Data

## Introduction

By default, the benchmark inference tool does not save output result data files after execution. After configuring the relevant parameters, the following result data can be generated:

| File/Directory                                | Description                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| {filename}.bin, {filename}.npy, or {filename}.txt | Model inference output result files.<br/>File naming format: name_output_index.suffix. When input is not specified (pure inference), the name is fixed as "pure_infer_data". When input is specified, the name is based on the first name of the first input. The output index starts from 0 and is arranged in output order. The file name suffix is controlled by the --outfmt parameter.<br/>By default, a "date+time" directory is created under the directory specified by the --output parameter, and the result files are saved in this directory. When --output-dirname is specified, the result files are saved directly in the directory specified by the --output-dirname parameter.<br/>When the --output-dirname parameter is specified, running the inference tool multiple times causes result files to be overwritten due to duplicate names. |
| xx_summary.json                          | The tool outputs model performance result data. By default, "xx" is named with "date+time". When --output-dirname is specified, "xx" is named with the directory name specified by --output-dirname.<br/>When the --output-dirname parameter is specified, running the inference tool multiple times causes result files to be overwritten due to duplicate names.                                                                                                                                                                   |
| dump                                     | Dump data file directory. When --dump is used to enable dump, a dump directory is created under the directory specified by the --output parameter to save dump data files.                                                                                                                                                                                                                                           |
| profiler                                 | Profiler performance data file directory. When --profiler is used to enable performance data collection, a profiler directory is created under the directory specified by the --output parameter to save performance data files.                                                                                                                                                                                                                           |

## Running Samples

1. Set only the --output parameter. The sample command and results are as follows:

    ```bash
    msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result
    ```

    ```ColdFusion
    result
    |-- 2022_12_17-07_37_18
    │   |-- pure_infer_data_0.bin
    |-- 2022_12_17-07_37_18_summary.json
    ```

2. Set the --input and --output parameters. The sample command and results are as follows:

    ```bash
    # The input directory contents are as follows
    ls ./data
    196608-0.bin  196608-1.bin  196608-2.bin  196608-3.bin  196608-4.bin  196608-5.bin  196608-6.bin  196608-7.bin  196608-8.bin  196608-9.bin
    ```

    - Description:
      The .bin file stores the tensor data entered by the user. You can generate it using the following method. The size and astype in the sample can be obtained through the debug mode tool. The --input parameter is designed for users to specify input data.

      ```python
      import numpy as np
      np.random.uniform(size=[32,32]).astype('float32').tofile('foo.bin')
      ```

    ```bash
    msit benchmark --om-model ./pth_resnet50_bs1.om --input ./data --output ./result
    ```

    ```bash
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

3. Set the --output-dirname parameter. The sample command and results are as follows:

    ```bash
    msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result --output-dirname subdir
    ```

    ```bash
    result
    |-- subdir
    │   |-- pure_infer_data_0.bin
    |-- subdir_summary.json
    ```

4. Set the --dump parameter. The sample command and results are as follows:

    ```bash
    msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result --dump 1
    ```

    ```bash
    result
    |-- 2022_12_17-07_37_18
    │   |-- pure_infer_data_0.bin
    |-- dump
    |-- 2022_12_17-07_37_18_summary.json
    ```

5. Set the --profiler parameter. The sample command and results are as follows:

    ```bash
    msit benchmark --om-model ./pth_resnet50_bs1.om --output ./result --profiler 1
    ```

    ```bash
    result
    |-- 2022_12_17-07_56_10
    │   |-- pure_infer_data_0.bin
    |-- profiler
    │   |-- PROF_000001_20221217075609326_GLKQJOGROQGOLIIB
    |-- 2022_12_17-07_56_10_summary.json
    ```

6. Output Result Explanation.

After the benchmark inference tool is executed, the screen output result sample is as follows:

- When the --display-all-summary parameter is set to False, the output is as follows:

  ```bash
  [INFO] -----------------Performance Summary------------------
  [INFO] NPU_compute_time (ms): min = 0.6610000133514404, max = 0.6610000133514404, mean = 0.6610000133514404, median = 0.6610000133514404, percentile(99%) = 0.6610000133514404
  [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(0.6610000133514404): 1512.8592735267011
  [INFO] ------------------------------------------------------
  ```

- When the --display-all-summary parameter is set to True, the output is as follows:

  ```bash
  [INFO] -----------------Performance Summary------------------
  [INFO] H2D_latency (ms): min = 0.05700000002980232, max = 0.05700000002980232, mean = 0.05700000002980232, median = 0.05700000002980232, percentile(99%) = 0.05700000002980232
  [INFO] NPU_compute_time (ms): min = 0.6650000214576721, max = 0.6650000214576721, mean = 0.6650000214576721, median = 0.6650000214576721, percentile(99%) = 0.6650000214576721
  [INFO] D2H_latency (ms): min = 0.014999999664723873, max = 0.014999999664723873, mean = 0.014999999664723873, median = 0.014999999664723873, percentile(99%) = 0.014999999664723873
  [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(0.6650000214576721): 1503.759349974173
  ```

Through the output results, you can view the model execution time and throughput. A smaller execution time and a higher throughput indicate better model performance.

**Field Description**

| Field                  | Description                                                    |
| --------------------- |-------------------------------------------------------|
| H2D_latency (ms)      | Host to Device memory copy time. The unit is ms.                          |
| min                   | Minimum inference execution time.                                            |
| max                   | Maximum inference execution time.                                            |
| mean                  | Average inference execution time.                                            |
| median                | Median inference execution time.                                           |
| percentile(99%)       | Percentile of inference execution time.                                         |
| NPU_compute_time (ms) | NPU inference computation time. The unit is ms.                                     |
| D2H_latency (ms)      | Device to Host memory copy time. The unit is ms.                          |
| throughput            | Throughput. Throughput formula: 1000 *batchsize/npu_compute_time.mean     |
| batchsize             | Batch size. This tool may not accurately identify the batchsize of the current sample. It is recommended to set it through the --batch-size parameter. |

## FAQ

For issues during use, refer to the [FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
