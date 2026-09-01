# Basic Usage

## Introduction

The benchmark inference tool can start model testing through the msit command line.

## Running Samples

### 1. Pure Inference Scenario

**By default, data filled with zeros is constructed and sent to the model for inference, and the output information is only displayed on the screen.**

- Enter the command:

    ```bash
    msit benchmark --om-model *.om
    ```

  where * is the OM offline model file name.
- Take the static model `resnet50_bs1.om` with a single batch as an example for inference. The execution process is as follows:

    ```ColdFusion
    [INFO] acl init success
    [INFO] open device 0 success
    [INFO] load model pth_resnet50_bs1.om success
    [INFO] create model description success
    [INFO] try get model batchsize:1
    [INFO] warm up 1 done
    Inference array Processing: 100%|████████████████████████████████| 1/1 [00:00<00:00, 10.73it/s]
    [INFO] -----------------Performance Summary------------------
    [INFO] NPU_compute_time (ms): min = 2.4560000896453857, max = 2.4560000896453857, mean = 2.4560000896453857, median = 2.4560000896453857, percentile(99%) = 2.4560000896453857
    [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(2.4560000896453857): 407.16610891670894
    [INFO] ------------------------------------------------------
    [INFO] unload model success, model Id is 1
    [INFO] end to destroy context
    [INFO] end to reset device is 0
    [INFO] end to finalize acl
    ```

  - Screen output explanation:

  ```ColdFusion
  NPU_compute_time (ms): # Inference time, excluding H2D (host to device) and D2H (device to host) time
      min = 2.4560000896453857 # Minimum inference time
      max = 2.4560000896453857 # Maximum inference time
      mean = 2.4560000896453857 # Average inference time
      median = 2.4560000896453857 # Median inference time
  ```

  ```ColdFusion
  throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(2.4560000896453857): 407.16610891670894 # Inference throughput, calculated as 1000*batchsize.mean(1)/NPU_compute_time.mean(2.4560000896453857)
  ```

### 2. Debug Mode

  **Enable the debug mode.**

  ```bash
  msit benchmark --om-model /home/model/resnet50_v1.om --output ./ --debug 1
  ```

  After the debug mode is enabled, more log information is displayed, including:

   - Model input and output parameter information

     ```bash
      [INFO] try get model batchsize:1
      [DEBUG] Input nums: 1
      [DEBUG] Model id: 1
      [DEBUG] aipp_input_exist: 0
      [DEBUG] session info:<Model>
      device: 0
      input:
        #0    actual_input_1  (1, 3, 224, 224)  float32  602112  602112
      output:
        #0    PartitionedCall_/fc/Gemm_add_4:0:output1  (1, 1000)  float32  4000  4000
     ```

   - Detailed inference time information

     ```bash
     [DEBUG] model aclExec cost : 2.336000
     ```

   - Detailed operation information such as model input and output

## FAQ

For issues during use, refer to the [FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
