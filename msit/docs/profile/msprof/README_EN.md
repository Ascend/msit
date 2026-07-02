# User Guide to the msit profile msprof Function

## Brief Introduction

 * For om files (Offline model converted from files such as onnx) Analyze the model inference performance on the Ascend.
 * The one-click full-process inference tool msit integrates the Profiling performance analysis tool to analyze the key performance bottlenecks in each running phase of the app project running on the Ascend AI processor and provide performance optimization suggestions to achieve ultimate product performance.
 * Profiling data is collected by using the binary executable file msprof. To collect profiling data, ensure that the Toolkit component package has been installed in the running environment of the application project or operator project.
 * For details about the usage restrictions of this tool, visit the following website:[CANN Commercial Version/Restrictions (Only in the Inference Scenario)](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0003.html)    

## Tool installation

 * For details about how to install the tool, see.[MSIT Integrated Tool User Guide](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md)    

## How to Use

### Function Description

#### Using the entrance

The profile can directly start performance analysis of model inference through the msit command line. Use the msit benchmark. (The msit benchmark project is built in the msit inference project. You only need to modify the OM path to analyze the performance and collect data of model inference.) The commands for inference performance analysis are as follows:

```bash
msit profile msprof --application "msit benchmark -om *.om --device 0" --output <some path>
```

\* indicates the name of the OM offline model file. some path indicates the path name. The main output is as follows:

```ColdFusion
**some path**
└── profiler
    └── PROF_000001_20231023172400639_NJDOONIBJCPMJGGB
        ├── device_0 #Indicates the performance data of the chip with the device ID of 0. device_0 indicates the performance data of the chip with the device ID of 0.
        │   ├── data #Raw performance data
        │   ├── log #Log of the profiling process
        │   ├── summary #Performance Data Summary Table
        │   └── timeline #Performance data is displayed in the time axis.
        └── host
            ├── data #Raw data on the host side
            ├── log #Log of the profiling process
            ├── summary #Performance Data Summary Table
            └── timeline #Performance data is displayed in the time axis.
```

The files in summary and timeline vary depending on the command line parameters. For details, see.[Using the msprof Tool](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/opdev/optool/atlasopdev_16_00851.html)    

#### Parameter Description

| Parameter name      | Description                                                                                                                                                                                                                                                                                                                                                                                                   | Mandatory. |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| --application       | Set this parameter to the executable file of the app in the running environment. You can configure the benchmark inference program of the msit. If the application parameter is specified, you need to use double quotation marks to enclose the parameter value of application, for example, --application "msit benchmark -om /home/HwHiAiUser/resnet50.om". You only need to modify the specified OM path. | Yes        |
| -o, --output        | Path for storing the collected profiling data. The default path is the output directory in the current path.                                                                                                                                                                                                                                                                                                  | No.        |
| --model-execution   | Indicates whether to enable the performance data collection function of the ge model execution module. The value can be on or off. The default value is on. The prerequisite for setting this parameter is that the application parameter has been set.                                                                                                                                                       | No.        |
| --sys-hardware-mem  | Data collection switch for DDR and LLC read/write bandwidth. The value can be on or off. The default value is on.                                                                                                                                                                                                                                                                                             | No.        |
| --sys-cpu-profiling | CPU (AI CPU, Ctrl CPU, or TS CPU) collection switch. The value can be on or off. The default value is off.                                                                                                                                                                                                                                                                                                    | No.        |
| --sys-profiling     | Enables or disables the collection of system CPU usage and system memory. The value can be on or off. The default value is off.                                                                                                                                                                                                                                                                               | No.        |
| --sys-pid-profiling | Indicates the CPU usage and memory collection switch of a process. The value can be on or off. The default value is off.                                                                                                                                                                                                                                                                                      | No.        |
| --dvpp-profiling    | DVPP collection switch. The value can be on or off. The default value is on.                                                                                                                                                                                                                                                                                                                                  | No.        |
| --runtime-api       | Specifies whether to enable the performance data collection function of the runtime API. The value can be on or off. The default value is on. The prerequisite for setting this parameter is that the application parameter has been set.                                                                                                                                                                     | No.        |
| --task-time         | Indicates whether to enable the TS timeline data collection function. The value can be on or off. The default value is on. The prerequisite for setting this parameter is that the application parameter has been set.                                                                                                                                                                                        | No.        |
| --aicpu             | Specifies whether the AICPU is enabled. The value can be on or off. The default value is on.                                                                                                                                                                                                                                                                                                                  | No.        |
| -h, --help          | Help information about the tool.                                                                                                                                                                                                                                                                                                                                                                              | No.        |
