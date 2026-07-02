# Energy Consumption #

## Introduction ##

Specify the NPU ID of the device to perform the inference test and obtain the inference energy consumption data.

## Running an Example ##

To query the mapping information about all chips, run the following command:

```
npu-smi info -m
```

The following is an example of the output:

```
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
4                              0                              0                              <soc_version>
4                              1                              -                              Mcu
```

Output description:

| Field         | Description    |
| ------------- | -------------- |
| NPU ID        | Device ID.     |
| Chip ID       | Chip ID.       |
| Chip Logic ID | Chip logic ID. |
| Chip Name     | Chip name      |

The NPU ID corresponding to device 0 is 4.

```
msit benchmark --om-model ./pth_resnet50_bs1.om --device 0 --energy-consumption 1 --npu-id 4
```

The following is an example of the output:

```
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 2.4769999980926514, max = 3.937000036239624, mean = 3.5538000106811523, median = 3.7230000495910645, percentile(99%) = 3.936680030822754
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(3.5538000106811523): 281.38893494131406
[INFO] ------------------------------------------------------
[INFO] NPU ID:4 energy consumption(J):59.88656545251415
[INFO] unload model success, model Id is 1
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

At the end of the command output, the energy consumption (J) consumed by the NPU ID corresponding to the specified device for model inference is displayed. The unit is J.

## FAQ ##

If a problem occurs, refer to.[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)	

