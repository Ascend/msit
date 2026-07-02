# trtexec scenario #

The benchmark supports ONNX model inference (integrated with trtexec), which is a built-in tool of NVIDIA TensorRT. Use the benchmark tool to start the trtexec tool to test the inference performance. During the test, trtexec logs are generated in real time and printed on the console. After the inference performance test is complete, performance data is output on the console.

## Prerequisite ##

The inference performance test environment must be configured with GPUs, and CANN, CUDA, and TensorRT must be installed. In addition, trtexec can be invoked through the CLI. For details about the installation method, see.[TensorRT](https://github.com/NVIDIA/TensorRT)	.

The example command is as follows:

```
msit benchmark -om pth_resnet50.onnx --backend trtexec --perf 1
```

Output the inference test result. The following is an example:

```
[INFO] [05/27/2023-12:05:31] [I] === Performance summary ===
[INFO] [05/27/2023-12:05:31] [I] Throughput: 120.699 qps
[INFO] [05/27/2023-12:05:31] [I] Latency: min = 9.11414 ms, max = 11.7442 ms, mean = 9.81005 ms, median = 9.76404 ms, percentile(90%) = 10.1075 ms, percentile(95%) = 10.1624 ms, percentile(99%) = 11.4742 ms
[INFO] [05/27/2023-12:05:31] [I] Enqueue Time: min = 0.516296 ms, max = 0.598633 ms, mean = 0.531443 ms, median = 0.5271 ms, percentile(90%) = 0.546875 ms, percentile(95%) = 0.564575 ms, percentile(99%) = 0.580566 ms
[INFO] [05/27/2023-12:05:31] [I] H2D Latency: min = 1.55066 ms, max = 1.57336 ms, mean = 1.55492 ms, median = 1.55444 ms, percentile(90%) = 1.55664 ms, percentile(95%) = 1.55835 ms, percentile(99%) = 1.56458 ms
[INFO] [05/27/2023-12:05:31] [I] GPU Compute Time: min = 7.54407 ms, max = 10.1723 ms, mean = 8.23978 ms, median = 8.19409 ms, percentile(90%) = 8.5354 ms, percentile(95%) = 8.59131 ms, percentile(99%) = 9.90002 ms
[INFO] [05/27/2023-12:05:31] [I] D2H Latency: min = 0.0130615 ms, max = 0.0170898 ms, mean = 0.015342 ms, median = 0.0153809 ms, percentile(90%) = 0.0162354 ms, percentile(95%) = 0.0163574 ms, percentile(99%) = 0.0168457 ms
[INFO] [05/27/2023-12:05:31] [I] Total Host Walltime: 3.02405 s
[INFO] [05/27/2023-12:05:31] [I] Total GPU Compute Time: 3.00752 s
```

**Field Description**

| Field                  | Description                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------ |
| Throughput             | Throughput.                                                                                            |
| Latency                | Sum of H2D latency, GPU compute time, and D2H latency. This is the delay inferring a single execution. |
| min                    | Minimum inference execution time.                                                                      |
| max                    | Maximum inference execution time.                                                                      |
| mean                   | Indicates the average inference execution time.                                                        |
| median                 | The median inference execution time is used.                                                           |
| percentile(99%)        | Percentile in the inference execution time.                                                            |
| H2D Latency            | Delay in host-to-device data transfer for a single executed input tensor.                              |
| GPU Compute Time       | GPU latency for executing CUDA cores.                                                                  |
| D2H Latency            | Delay in the device-to-host data transfer for a single executed output tensor.                         |
| Total Host Walltime    | Host time from the first execution (after warm-up) enqueued to the last execution completed.           |
| Total GPU Compute Time | The sum of all GPU compute time executed.                                                              |

## FAQ ##

If a problem occurs, refer to.[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)	

