# Save Profiler or Dump Data

## 0 Basic Introduction

- When the optional command `--profiler` is enabled, benchmark calls msprof to collect performance data during inference.
- When the optional command `--dump` is enabled, benchmark retains the output of all operators during inference.
- When `--acl-json-path` is enabled, you can customize the profiler or dump configuration parameters in the json file (only one can be selected).

## 1 Basic Running Samples

### 1.1 --profiler Collects Performance Data During Inference

- Sample command:

```bash
msit benchmark --om-model /home/model/resnet50_v1.om --output ./output --profiler 1

```

- Sample output file directory:

```bash
|--- output/
|    |--- 2023_06_08_19_27_summary.json # Summary of inference results (overall inference performance data)
|    |--- 2023_06_08_19_27/ # Input files
|    |    |--- pure_infer_data_0.bin
|    |--- profiler/  # Collected performance data
|    |    |--- PROF_000001_20230608201922856_LPKNFOADMAQRMDGC/ # Data saved by msprof
|    |    |    |--- host/ # Host-side data
|    |    |    |--- device_0/ # Device-side data. Multiple device folders appear when multiple devices are used for inference

```

### 1.2 --dump Collects Output Data of Each Operator During Inference

- Sample command:

```bash
msit benchmark --om-model /home/model/resnet50_v1.om --output ./output --dump 1
```

- Sample output file directory:

```bash
|--- output/
|    |--- acl.json # Same as the json file configured by the --acl-json-path command
|    |--- 2023_06_08_19_27_summary.json  # Summary of inference results (overall inference performance data)
|    |--- 2023_06_08_19_27/
|    |    |--- pure_infer_data_0.bin # Input file
|    |--- dump/  # Collected output data of each operator
|    |    |--- 20230608192722/ # Dump data
|    |    |    |--- 0/
|    |    |    |    |--- resnet50_v1/
|    |    |    |    |    |--- 1/
|    |    |    |    |    |--- 0/
```

### 1.3 --acl-json-path Customizes Data Collection During Inference

+ The --acl-json-path parameter specifies the acl.json file. You can configure profiler or dump parameters in this file. The sample json file is as follows:

  + Collect performance data during inference through profiler

    ```bash
    # acl.json
    {
    "profiler": {
                  "switch": "on",
                  "output": "./result/profiler"
                }
    }
    ```

    For more performance parameter configurations, refer to the "Other Performance Data Collection Methods > [Collecting Performance Data Using the acl.json Configuration File](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0054.html)" chapter in the CANN Commercial Edition Performance Tuning Tool User Guide, or the "Other Performance Data Collection Methods > [Collecting Performance Data Using the acl.json Configuration File](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/devaids/Profiling/atlasprofiling_16_0054.html)" chapter in the CANN Community Edition Performance Tuning Tool User Guide, depending on your CANN package type (commercial or community edition).

  + Collect operator output through dump

    ```bash
    # acl.json
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

    For more dump configurations, refer to the "NPU vs NPU (Offline Inference) > [Preparing Offline Model Dump Data Files](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/modelaccuracy/atlasaccuracy_16_0028.html)" chapter in the CANN Precision Debugging Tool User Guide.

- When collecting profiler data through this method, if the environment variable `export MSIT_NO_MSPROF_MODE=1` is configured, refer to the "Using the msprof Command to Parse and Export Performance Data > [Parse and Export Performance Data](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0021.html)" chapter in the CANN Performance Tuning Tool User Guide to parse and export the output performance data files to the mindstudio_profiler_output directory.
- When collecting profiler data through this method, if the environment variable `MSIT_NO_MSPROF_MODE=1` is **not** configured, benchmark parses the profiler-related parameters in acl.json into msprof commands and calls msprof to collect performance data. For the meaning of msprof output files, refer to the "Performance Data File Reference > [Overall Description](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0057.html)" chapter in the CANN Performance Tuning Tool User Guide.
- If both profiler and dump parameters are configured in the acl.json file, you must configure the environment variable `export MSIT_NO_MSPROF_MODE=1` to ensure simultaneous collection.

Sample command:

  ```bash
  msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --acl-json-path ./acl.json
  ```

For the output files, refer to the samples in sections 1.1 and 1.2.

## 2 Extended Usage

### 2.1 Custom Usage of --profiler

+ The profiler is a set of performance data collection configurations built into the program. The generated performance data is saved in the profiler folder under the directory specified by the --output parameter.

    This parameter starts the msprof command for performance data collection by calling the msprof_run_profiling function in msit/components/profile/msprof/ait_prof/msprof_process.py. To modify the performance data collection parameters, modify the msprof_cmd parameter in the msprof_run_profiling function as needed. The sample is as follows:

    ```bash
    msprof_cmd="{} --output={}/profiler --application=\"{}\" --model-execution=on --sys-hardware-mem=on --sys-cpu-profiling=off --sys-profiling=off --sys-pid-profiling=off --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on".format(
            msprof_bin, args.output, cmd)
    ```

    When collecting performance data through this method, the system first checks whether the msprof command exists:

    - If the command exists, the command is used for performance data collection. The parsed and exported files are stored in the mindstudio_profiler_output directory.
    - If the command does not exist, an error is reported at the msprof level. The benchmark does not check the validity of the command content.
    - If the environment has MSIT_NO_MSPROF_MODE=1 configured, the default acl.json file constructed by benchmark is used when the --profiler parameter collects performance data.

- If the msprof command does not exist or the environment has MSIT_NO_MSPROF_MODE=1 configured, the collected performance data files are not automatically parsed. Refer to the "Using the msprof Command to Parse and Export Performance Data > [Parse and Export Performance Data](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0021.html)" chapter in the CANN Performance Tuning Tool User Guide to parse and export the performance data to the mindstudio_profiler_output directory.
- For more performance data collection parameter descriptions, refer to the "Using the msprof Command to Collect Performance Data > [Common msprof Collection Commands](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/profiling/atlasprofiling_16_0010.html)" chapter in the CANN Performance Tuning Tool User Guide.

### 2.2 Mixed Usage of `--profiler`, `--dump`, and `--acl-json-path`

  + --acl-json-path takes precedence over --profiler and --dump. When both are set, --acl-json-path prevails.
  + For the --profiler and --dump parameters, you must add the --output parameter to specify the output path.
  + --profiler and --dump can be used separately, but cannot be enabled at the same time.

## FAQ

For issues during use, refer to the [FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
