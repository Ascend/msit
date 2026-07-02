# MindStudio Prechecker Tool

## Overview

MindStudio's prechecker tool (msprechecker): A tool that helps users quickly deploy AI inference services, reproduce performance baselines, and locate deployment and performance issues in the Ascend environment. The tool provides three core functions: precheck, environment information dump, and difference comparison. **Starting from version 2025.12.16**, it has added support for the `run` and `inspect` subcommands based on an extensible rule engine, allowing users to customize and share complex check rules.

### Basic Concepts

- PD colocation (Prefill-Decode colocation deployment): A deployment mode where the Prefill phase and Decode phase of a model are deployed in the same compute instance (Pod/container).
- PD separation (Prefill-Decode separation deployment): A deployment mode where the Prefill phase and Decode phase are decoupled and deployed on different, independently scalable compute instances or Pods.
- Rank Table: A configuration file that describes the topological connection relationships of Ascend NPU chips in a multi-node multi-card environment, used to guide task startup and communication for distributed training or inference.
- cmate file: A rule file defined based on a specific syntax, used to describe check and assertion logic for objects such as systems, environment variables, and configuration files. The tool has built-in common rule sets for the MindIE and VLLM-Ascend frameworks.

### Tool Usage Flow

First use the `precheck` function to perform a comprehensive check of the target environment before deployment; when a service is running (or encountering issues), use the `dump` function to save an environment snapshot; when you need to compare differences between different environments (such as a baseline environment and a problematic environment), use the `compare` function to analyze multiple dump files. For users with customized check requirements, you can use `inspect` to view rule files and use `run` to execute custom rule checks.

## Preparation

### Environment Setup

1. Prepare a server installed with Ascend NPUs, and ensure that the NPU driver, firmware, and CANN software package are correctly installed.
2. Ensure that the Python version is 3.7 or later.
3. Install msprechecker and its dependencies.

    You can install it in either of the following ways:
    - PyPI installation (recommended)

        ```bash
        pip install msprechecker
        ```

    - Offline installation
        1. From a machine with internet access, visit <https://pypi.org/project/msprechecker/#files> to download the wheel package.
        2. Upload the downloaded wheel package to the target server.
        3. Run the installation command (replace `whl_path` with the actual path):

            ```bash
            pip install whl_path
            ```

    - Source Installation

        ```bash
        git clone https://gitcode.com/Ascend/msit.git
        pip install -e msit/msprechecker
        ```

4. Install the necessary third-party dependencies: `psutil`, `pyyaml`, `importlib_metadata`, `ply`. These are usually resolved automatically when installing msprechecker.

### Constraints

- Before installation as a non-root user, it is recommended to run `umask 0027` to avoid permission issues during subsequent use.
- The current tool primarily supports training servers such as Atlas 800I A2, Atlas 800I A3, and Atlas 9000 A2 (G8600).
- Supports MindIE and vLLM-Ascend (v0.9.1-dev) inference verification.
- The `dump` function does not currently support disk flushing of configuration files for multi-node PD separation and single-node PD separation scenarios.

## Quick Start

This section uses the most common **PD colocation in MindIE** as an example to demonstrate how to use msprechecker for pre-deployment checks.

### Prerequisites

1. The environment preparation is complete and msprechecker has been successfully installed.
2. The configuration file `config.json` for the MindIE service has been obtained (the typical path is `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`).
3. Ensure that the model weight directory to be deployed is accessible.

> [!Warning]
> Note: MindIE has been fully open-sourced since March 2026. After this date, the location of its configuration file may have changed. If it is not found at the default path mentioned above, you can locate it using the environment variable `MINDIE_LLM_HOME_PATH`. The full path to the configuration file is typically `$MINDIE_LLM_HOME_PATH/conf/config.json`.

### Procedure

1. Execute the precheck command, specifying the MindIE service configuration file and the model weight directory.

    ```bash
    msprechecker precheck --mies-config-path /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json --weight-dir /path/to/your/model_weights
    ```

2. The tool will run a series of checks, including system configuration, environment variables, NPU status, network connectivity, and model configuration compliance.
3. The check results will be displayed in the terminal and marked with different levels:
    - `[NOK]`: Critical issue that may cause deployment failure or severe performance degradation. Must be fixed.
    - `[WARNING]`: Potential issue or non-optimal configuration. Recommended to fix.
    - `[RECOMMEND]`: Optimization suggestion. Fixing it can yield better performance.
4. If environment variable configuration issues are found, the tool generates an `msprechecker_env.sh` file in the current directory. You can review it and decide whether to adopt the suggestions.
5. Based on the suggestions output by the tool, fix the identified issues one by one until all checks pass or only acceptable `[RECOMMEND]` items remain. Then you can start deployment.

## Command-line Tool Guide

msprechecker includes five subcommands: `precheck`, `dump`, `compare`, `run`, and `inspect`.

### Supported Products

>[!NOTE]
>For specific models of Ascend products, see [Ascend Product Overview](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html).

| Product Type | Supported |
| ------------ | :-------: |
| Atlas A3 training products/Atlas A3 inference products | √ |
| Atlas A2 training products/Atlas A2 inference products | √ |
| Atlas 200I/500 A2 inference products | × |
| Atlas inference products | √ |
| Atlas training products | × |

### Description

`msprechecker` is used to perform health checks, information collection, and difference comparison on the Ascend AI inference deployment environment, aiming to improve deployment success rates and problem locating efficiency. The new version introduces an extensible check engine based on `cmate` rule files, providing users with more flexible and powerful custom check capabilities.

### Precautions

- In a multi-node PD colocation scenario, the precheck command must be executed separately on **each** target node. The tool does not support controlling multiple machines from a single point.
- When using `--rank-table-path` for network checks, the tool parses the rank table file in MindIE format by default. If used for the VLLM-Ascend framework, be sure to specify this with the `--scene vllm` parameter.
- The correctness of some environment variables (such as `RANK_TABLE_FILE`) requires users to manually confirm based on their own deployment planning.

### Command Format

```shell
msprechecker <subcommand> [options]
```

#### Subcommands (choose one)

- `precheck`: Perform pre-deployment environment precheck.
- `dump`: Save current environment information (system, environment variables, configurations, etc.) to a file.
- `compare`: Compare two or more files generated by the `dump` command to identify differences.
- `run`: Execute the user-specified `cmate` rule file for checking.
- `inspect`: View the metadata and rule content of the `cmate` rule file.

### Parameter Description

#### `precheck` subcommand parameters

| Parameter | Optional/Required | Description |
| :--- | :--- | :--- |
| `--scene` | Optional | Specify the deployment scenario. Format: `<framework>,<deploy-mode>[,<npu_type>,<npu_count>,<arch>,<model_type>]`. For example: `mindie,pd_mix` or `vllm,ep,A2,8,arm`. Used to help the tool identify information such as the framework, deployment mode, and hardware. Default value: `None` |
| `--mies-config-path` | Optional (MindIE PD colocation) | The configuration file path for the MindIE PD colocation scenario, typically `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`. Default value: `None` |
| `--config-parent-dir` | Optional (PD separation) | In the PD separation scenario, the parent directory path (usually named `kubernetes_deploy_scripts`) containing all `conf/*.json` and `deployments/*.yaml` configuration files to be modified. Must be used with `--scene pd_disaggregation` or `--scene pd_disaggregation_single_container`. Default value: `None` |
| `--user-config-path` | Optional (Large EP scenario) | The `user_config.json` file path in the Large EP scenario. Default value: `None` |
| `--mindie-env-path` | Optional (Large EP scenario) | The `mindie_env.json` file path in the Large EP scenario. Default value: `None` |
| `--rank-table-path` | Optional | Specify the rank table file path, used to trigger network connectivity tests between NPUs and related configuration checks. Default value: `None` |
| `--weight-dir` | Optional | Specify the model weight directory path, used to check the `config.json` configuration file in that directory. Default value: `None` |
| `--hardware` | Optional | Enable CPU and NPU hardware stress testing to detect cores or cards with abnormal computing power. Default value: `None` |
| `--threshold` | Optional | The filtering threshold for hardware stress testing, with a value range of `[0-100]` (closed interval). The unit is percentage. When the ratio of a core/card's computation time exceeding the average value is greater than this threshold, it is marked as abnormal. Default value: 20 |
| `--custom-config-path` | Optional | The path to the user-defined check rule file (YAML format). Default value: `None` |
| `-l`, `--severity-level` | Optional | Control the severity level of output information. Optional values: `low` (show all), `medium` (show WARNING and NOK), `high` (show only NOK). Default value: `low` |

#### `dump` subcommand parameters

| Parameter | Optional/Required | Description |
| :--- | :--- | :--- |
| `--output-path` | Optional | File path, specifies the save path for the dumped data file. Default value: `./msprechecker_dumped.json`. |
| `--filter` | Optional | If enabled, only dumps environment variables related to Ascend R&D, filtering out irrelevant variables. Default value: `False` |
| `--user-config-path` | Optional | File path, performs extra disk flushing of the specified `user_config.json` file content. Default value: `None` |
| `--mindie-env-path` | Optional | File path, performs extra disk flushing of the specified `mindie_env.json` file content. Default value: `None` |
| `--mies-config-path` | Optional | File path, performs extra disk flushing of the specified MindIE service `config.json` file content. Default value: `None` |
| `--rank-table-path` | Optional | File path, performs extra disk flushing of the specified rank table file content. Default value: `None` |
| `--weight-dir` | Optional | Directory path, performs extra disk flushing of the `config.json` and SHA256 hash values of all weight files in the specified weight directory. Default value: `None` |
| `--chunk-size` | Optional | The data block size (MB) read each time when calculating the hash of weight files. Optional values: 32, 64, 128, 256. Default value: `32`. |

#### `compare` subcommand parameters

The `compare` subcommand accepts one or more file paths as positional parameters for comparing the differences in the file content of these files.

#### `run` subcommand parameters

| Parameter | Optional/Required | Description |
| :-------- | :---------------- | :---------- |
| `rule` | **Required** (positional parameter) | The path of the `cmate` rule file to be executed. |
| `-C`, `--contexts` | Optional | Passes context variables required by the rule. Syntax: `-C variable_name:variable_value`. For example: `-C npu_count:2` passes the integer 2, `-C model_name:"deepseek"` passes the string "deepseek". Can be used multiple times to pass multiple variables. Default value: `None` |
| `-c`, `--configs` | Optional | Passes the configuration file path required by the rule. Syntax: `-c config_variable_name_in_rule:actual_file_path`. The parsing type can be specified: `config_variable_name:file_path@parsing_type` (e.g., `json`, `yaml`). If not specified, the rule definition or file extension is used. Can be used multiple times. Default value: `None` |
| `-co`, `--collect-only` | Optional | Only collects the rule items to be executed, but does not actually execute them. Similar to the `--collect-only` option in testing frameworks. |
| `--output-path` | Optional | The save path for the precheck results. The saved file format is `msprechecker_{timestamp}_output.json`. Default value: `''` (meaning the precheck results are printed directly) |
| `-x`, `--fail-fast` | Optional | Stops execution immediately upon encountering the first rule check failure. Default value: `False` |
| `-v`, `--verbose` | Optional | Verbose output mode, displays the line number and specific check content of each rule in the cmate file. Default value: `False` |
| `-s`, `--severity` | Optional | The minimum severity level for running rules. Optional values: `info` (run all), `warning` (do not run info level), `error` (only run error level). Default value: `info` |

#### `inspect` subcommand parameters

| Name | Optional/Required | Description |
| :--- | :--- | :--- |
| `rule` | **Required** (positional parameter) | Path to the `cmate` rule file to view. |
| `-f`, `--format` | Optional | Output format. Optional values: `text` (text), `json` (JSON format). Default value: `text` |

### Usage Examples

**Example 1: Precheck for vLLM-Ascend general scenarios**
Check the basic environment for vLLM-Ascend framework deployment.

```bash
msprechecker precheck --scene vllm
```

**Example 2: Precheck for MindIE large EP scenarios**
Check the configuration for the large EP (Elastic Processing) deployment scenario.

```bash
msprechecker precheck --scene mindie,ep --user-config-path /path/to/user_config.json --mindie-env-path /path/to/mindie_env.json
```

**Example 3: Execute a cmate rule file**
Run the built-in MindIE rule file, and pass in the necessary configuration file and context variables.

```bash
msprechecker run /path/to/msprechecker/preset/mindie.cmate \
  -c mies_config:/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json \
  -C deploy_mode:pd_mix model_type:deepseek npu_type:A2
```

**Example 4: View rule file information**
View the overview, context requirements, and configuration definitions of a rule file in text format.

```bash
msprechecker inspect /path/to/msprechecker/preset/mindie.cmate
```

**Example 5: Environment information dump**
Save the current complete system, environment variables, Ascend-related configurations, and other information to a specified file.

```bash
msprechecker dump --output-path /tmp/env_baseline.json --weight-dir /path/to/model_weights
```

**Example 6: Environment difference comparison**
Compare environment information files saved at two different time points or on different machines.

```bash
msprechecker compare /tmp/baseline.json /tmp/problem_env.json
```

### Output Description

After the `precheck` command is executed, a detailed check report is output in the terminal. The report is grouped by check category, with a severity marker (`[NOK]`, `[WARNING]`, `[RECOMMEND]`) before each result, accompanied by a problem description and suggestion.

After the `run` command is executed, the output format is similar to a testing framework, displaying the rule collection progress, execution result summary, and detailed failure assertion information, including the expected value (marked with `>`), actual value (marked with `E`), error level, and reason.

After the `dump` command is executed, the save path of the flushed files is output.

After the `compare` command is executed, the differences between different files are output in a clear JSON format. If there are no differences, it prompts `There is no difference found.`.

After the `inspect` command is executed, the metadata, context variables, configuration definitions, etc., of the rule file are output based on the `--format` parameter.

## Extended Features

### Custom Check Item Configuration (YAML)

You can define custom check rules through a YAML file and pass it to the `precheck` command using the `--custom-config-path` parameter. This allows you to set expected values and check logic for specific environment variables or configuration items.

**Rule file example (`custom_rules.yaml`):**

```yaml
MY_CUSTOM_ENV:
  expected:
    type: eq
    value: "expected_value"
  reason: "Custom environment variables should be set to specific values."
  severity: high
```

Currently, comparison types such as `eq`/`==`, `lt`/`<`, `le`/`<=`, `gt`/`>`, `ge`/`>=`, `ne`/`!=` are supported. The `value` field supports arithmetic operations and field references (see below).

### Field Reference Syntax

In custom check rules, you can use the `${}` syntax to reference the values of other fields in the configuration file, enabling the definition of complex correlation checks.

**Example:**
Assume the configuration file contains `{"a": {"b": 10, "c": 20}}`. You can create the following rule to check whether `a.b + a.c == 30`:

```yaml
a:
  b:
    expected:
      type: eq
      value: 30 - ${.c}  # Relative reference a.c
    reason: "a.b should equal 30 minus the value of a.c."
  c:
    expected:
      type: eq
      value: 30 - ${a.b} # Absolute reference a.b
    reason: "a.c should equal 30 minus the value of a.b."
```

### Custom-Rule Engine (cmate file)

Starting from version 2025.12.16, msprechecker introduces a new, more powerful rule engine, operated through the `run` and `inspect` subcommands. Rules are defined via files in `cmate` format, supporting more complex assertion logic, conditional branching, modular organization, and rich context control.

- **Built-in rules**: The tool has built-in common check rules for the MindIE and vLLM-Ascend frameworks in the `mindie.cmate` and `vllm.cmate` files under the `msprechecker/preset/` directory. Users can use them directly or as a reference for writing their own.
- **Usage process**:
    1. **View rules**: Use `msprechecker inspect <rule.cmate>` to view the context variables, configuration files, and rule descriptions required by the rule file.
    2. **Execute rules**: Use `msprechecker run <rule.cmate>` with the `-C` (pass context) and `-c` (pass configuration file) parameters to execute checks.
- **Advantages**:
    1. **Extensible**: Users can develop new rule sets based on the `cmate` syntax and share them within teams or the community.
    2. **Flexible**: The rule file supports defining checks for multiple data sources (environment variables, configuration files, system status), and can dynamically control the check logic through context variables.
    3. **Structured output**: The `run` command provides clearly structured, unit-test-like output, making it easy to integrate into CI/CD processes.

For detailed syntax and authoring guides for `cmate` files, please refer to the tool's source code or related development documentation.

## Appendixes

### FAQs

1. **Q: How do I specify a rank table for the vLLM-Ascend framework to perform a check?**

    A: When executing the command, you must explicitly specify the framework using `--scene vllm` so that the tool parses the rank table in the correct format. For example: `msprechecker precheck --scene vllm --rank-table-path /path/to/vllm_hccl.json`

2. **Q: What does the `[WARNING]` that appears during `dump` mean?**

    A: This usually occurs because when flushing certain information to disk (such as a configuration file), it is skipped due to an unprovided path or a nonexistent file. This does not affect the disk flushing of other information. The final `.json` file generated is still valid, only lacking the data for the corresponding part.

3. **Q: In a multi-machine scenario, do I need to run the precheck on every machine?**

    A: Yes. For a PD colocation multi-machine scenario, you need to run the precheck command separately on **each server node participating in the computation**, because the environment configuration may differ on each machine.

4. **Q: What is the difference between the `run` subcommand in the new version and the original `precheck` command?**

    A: `precheck` is a packaged, fixed check process for specific scenarios, ready to use out of the box. The `run` command provides a general-purpose, programmable rule execution engine that allows users to define and execute arbitrarily complex check logic through `cmate` files, offering great flexibility. It is suitable for scenarios with customized check requirements or where users wish to share check rules. The built-in `precheck` functionality of the tool is also being gradually migrated to the new rule engine at the underlying level.
