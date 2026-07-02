# Save Profiler or Dump Data #

## 0 Basic Introduction ##

 *  When the optional command is enabled`--profiler`, the benchmark invokes msprof to collect performance data in inference.
 *  When the optional command is enabled`--dump`. The output of all operators is retained during benchmark inference.
 *  When turned on`--acl-json-path`. You can customize the configuration parameters of profiler or dump in the JSON file.

## 1 Basic Operation Example ##

### 1.1 --Profiler collects performance data during inference. ###

 *  Example command:

```
msit benchmark --om-model /home/model/resnet50_v1.om --output ./output --profiler 1
```

 *  An example of the output file directory is as follows:

```
|--- output/
|    |--- 2023_06_08_19_27_summary.json #Summarize inference results (overall inference performance data)
|    |--- 2023_06_08_19_27/ #Input File
|    |    |--- pure_infer_data_0.bin
|    |--- profiler/  #Collected performance data
|    |    |--- PROF_000001_20230608201922856_LPKNFOADMAQRMDGC/ #Data stored in the msprof.
|    |    |    |--- host/ #Data on the host side
|    |    |    |--- device_0/ #Data on the device side. When multiple devices are invoked, multiple and device folders are generated.
```

### 1.2 --Dump collects output data of operators at each layer in inference. ###

 *  Example command:

```
msit benchmark --om-model /home/model/resnet50_v1.om --output ./output --dump 1
```

 *  An example of the output file directory is as follows:

```
|--- output/
|    |--- acl.json #Same as the JSON file configured using the --acl-json-path command.
|    |--- 2023_06_08_19_27_summary.json  #Summarize inference results (overall inference performance data)
|    |--- 2023_06_08_19_27/
|    |    |--- pure_infer_data_0.bin #Input File
|    |--- dump/  #Output data of collected operators at each layer
|    |    |--- 20230608192722/ #Dump data
|    |    |    |--- 0/
|    |    |    |    |--- resnet50_v1/
|    |    |    |    |    |--- 1/
|    |    |    |    |    |--- 0/
```

### 1.3 --acl-json-path Customizing Data Collection and Inference ###

 *  The --acl-json-path parameter specifies the acl.json file. You can set the profiler or dump parameter in the file. An example JSON file is as follows:
    
     *  Use the profiler to collect performance data during inference.
        
        ```
        #acl.json
        {
        "profiler": {
                      "switch": "on",
                      "output": "./result/profiler"
                    }
        }
        ```
        
        For details about how to configure more performance parameters based on the CANN package type (commercial edition or community edition), see "Other Performance Data Collection Methods" in the CANN Performance Optimization Tool User Guide.[Using the acl.json Configuration File to Collect Performance Data](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0054.html)	"Other Performance Data Collection Methods" in the CANN Performance Optimization Tool User Guide for Community Edition[Using the acl.json Configuration File to Collect Performance Data](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/devaids/Profiling/atlasprofiling_16_0054.html)	describes the parameter settings in.
     *  Collecting the Output of the Operator Through Dump
        
        ```
        #acl.json
        {
            "dump": {
                "dump_list": [
                    {
                        "model_name": "{model_name}"
                    }
                ],
                "dump_mode": "output",
                "dump_path": "./result/dump"
            }
        }
        ```
        
        For details about dump configurations, see "NPU vs NPU (Offline Inference)" in the CANN Precision Debugger User Guide.[Obtaining the Dump Data File of an Offline Model](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/modelaccuracy/atlasaccuracy_16_0028.html)	"chapter.

 *  If the environment variables are configured during profiler collection in this mode,`export MSIT_NO_MSPROF_MODE=1`. For details, see section "Using the msprof Command to Parse and Export Performance Data >" in the CANN Performance Optimization Tool User Guide.[Parse and export performance data.](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0021.html)	Parse the generated performance data file and export it as a file in the mindstudio_profiler_output directory.
 *  If the environment variables are not configured when the profiler is collected in this mode,`MSIT_NO_MSPROF_MODE=1`, the benchmark parses the profiler-related parameters in the acl.json file into the msprof command and invokes the msprof command to collect performance data. For details about the output files, see "Performance Data File Reference" in the CANN Performance Optimization Tool User Guide.[General Description](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0057.html)	"chapter.
 *  If both the profiler and dump parameters are configured in the acl.json file, you need to configure environment variables.`export MSIT_NO_MSPROF_MODE=1`Ensure simultaneous collection.

Example command:

```
msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --acl-json-path ./acl.json
```

For details about the output file, see the examples in sections 1.1 and 1.2.

## 2 Extended Usage Instructions ##

### 2.1 --Customized usage of the profiler ###

 *  The profiler is a group of performance data collection configurations fixed into the program. The generated performance data is stored in the profiler folder in the directory specified by the --output parameter.
    
    This parameter is used to invoke the msprof_run_profiling function in msit/components/profile/msprof/ait_prof/msprof_process.py to start the msprof command to collect performance data. To modify performance data collection parameters, modify the msprof_cmd parameter in the msprof_run_profiling function based on the site requirements. The following is an example:
    
    ```
    msprof_cmd="{} --output={}/profiler --application=\"{}\" --model-execution=on --sys-hardware-mem=on --sys-cpu-profiling=off --sys-profiling=off --sys-pid-profiling=off --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on".format(
            msprof_bin, args.output, cmd)
    ```
    
    Before collecting performance data in this mode, check whether the msprof command exists.
    
     *  If the command exists, run the command to collect performance data, parse the data, and export the data to the mindstudio_profiler_output directory.
     *  If the command does not exist, the MSProf layer reports an error and the Benchmark layer does not check the validity of the command content.
     *  If MSIT_NO_MSPROF_MODE is set to 1, the default acl.json file constructed by the benchmark is invoked when the --profiler parameter is used to collect performance data.

 *  When the msprof command does not exist or MSIT_NO_MSPROF_MODE is set to 1, the collected performance data files are not automatically parsed. For details, see section "Using the msprof Command to Parse and Export Performance Data" in the CANN Performance Optimization Tool User Guide.[Parse and export performance data.](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0021.html)	Parse and export performance data as a file in the mindstudio_profiler_output directory.
 *  For details about the parameters for collecting performance data, see "Using the msprof Command to Collect Performance Data >" in the CANN Performance Optimization Tool User Guide.[Common Commands for Collecting MSProf Data](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0010.html)	"chapter.

### 2.2`--profiler` `--dump`And to the`--acl-json-path`Mixed Use Instructions ###

 *  The priority of --acl-json-path is higher than that of --profiler and --dump. If both --acl-json-path and --dump are set, --acl-json-path takes effect.
 *  \--profiler and --dump parameters. The --output parameter must be added to indicate the output path.
 *  \--profiler and --dump can be used separately, but not simultaneously.

## FAQ ##

If a problem occurs, refer to.[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)	

