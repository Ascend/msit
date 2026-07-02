# Basic Usage #

## Introduction ##

The benchmark inference tool can start the model test in the msit command line mode.

## Running an Example ##

### 1. Pure inference scenario ###

**By default, data whose values are all 0s is constructed and sent to the model inference. The output information is displayed only on the screen.**

 *  Enter the command.
    
    ```
    msit benchmark --om-model *.om
    ```
    
    In the preceding command, \* indicates the name of the OM offline model file.
 *  Static model in a single batch`resnet50_bs1.om`For example, the inference process is as follows:
    
    ```
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
    
     *  Description of the screenshot:
    
    ```
    NPU_compute_time (ms): #Inference time, excluding the host to device (H2D) and device to host (D2H) time.
        min = 2.4560000896453857 #Minimum time for reasoning
        max = 2.4560000896453857 #Maximum inference time
        mean = 2.4560000896453857 #Average time for inference
        median = 2.4560000896453857 #Median inference time
    ```
    
    ```
    throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(2.4560000896453857): 407.16610891670894 #Inferred throughput, which is calculated as 1000*batchsize.mean(1)/NPU_compute_time.mean(2.4560000896453857).
    ```

### 2. Debugging mode ###

**Enable the debug mode.**

```
msit benchmark --om-model /home/model/resnet50_v1.om --output ./ --debug 1
```

After the debugging mode is enabled, more information is displayed, including:

 *  Input and output parameters of the model
    
    ```
    [INFO] try get model batchsize:1
     [DEBUG] Input nums: 1
     [DEBUG] Model id: 1
     [DEBUG] aipp_input_exist: 0
     [DEBUG] session info:<Model>
     device: 0
     input:
       #0 actual_input_1 (1, 3, 224, 224) float32 602112 602112
     output:
       #0 PartitionedCall_/fc/Gemm_add_4:0:output1 (1, 1000) float32 4000 4000
    ```
 *  Detailed inference time information
    
    ```
    [DEBUG] model aclExec cost : 2.336000
    ```
 *  Detailed operation information such as model input and output

## FAQ ##

If a problem occurs, refer to.[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)	

