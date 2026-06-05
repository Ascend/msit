# **msIT**

## What's New

- [2026.1.12]: [License Change Notice of the msit Repository](https://gitcode.com/Ascend/msit/discussions/1).

- [2025.12.31]: MindStudio Inference Tools (msIT) is fully open source, including the following repositories:

    - [MindStudio-Profiler](https://gitcode.com/Ascend/msprof)  
    Provides basic performance tuning capabilities for all Ascend scenarios. It collects CANN and NPU performance data to improve tuning efficiency on Ascend devices.

    - [MindStudio-Profiler-Analyze](https://gitcode.com/Ascend/msprof-analyze)  
    Analyzes collected profile data to quickly identify performance bottlenecks on Ascend devices.

    - [MindStudio-MemScope](https://gitcode.com/Ascend/msmemscope)  
    Dedicated tool for Ascend memory debugging and tuning. It provides multi-dimensional memory data collection, automatic diagnosis, and optimization analysis across the entire network.

    - [MindStudio-Service-Profiler](https://gitcode.com/Ascend/msserviceprofiler)  
    Ascend-compatible service profiler that supports request scheduling and model execution visualization, improving service performance analysis efficiency.

    - [MindStudio-Monitor](https://gitcode.com/Ascend/msmonitor)  
    One-stop online monitoring tool that supports both disk dump and online performance data collection, with performance monitoring and fault location capabilities in cluster scenarios.

    - [MindStudio-ModelSlim](https://gitcode.com/Ascend/msmodelslim)  
    Ascend-native model compression tool with acceleration as its goal, compression as its technique, and Ascend as its foundation. It includes quantization, compression, and other inference optimizations, supporting large language dense models, MoE models, multi-modal understanding models, and multi-modal generation models.

    - [MindStudio-Insight](https://gitcode.com/Ascend/msinsight)  
    Supports multi-dimensional performance analysis across system, operator, and serving scenarios. It provides deep performance data analysis to help developers diagnose performance issues.
    
    - [MindStudio-Probe](https://gitcode.com/Ascend/msprobe)  
    Toolkit for precision debugging during model development. It is a full-scenario precision toolchain for Ascend and helps users improve model precision debugging efficiency.

## Overview

MindStudio Inference Tools (msIT) provides capabilities commonly used in developing inference for both large language models and traditional models, including model compression, debugging, and tuning. It supports performance tuning in inference serving scenarios, helping users achieve optimal inference performance.

## Directory Structure 

The key directories are as follows:

```tex
|—————— msit                     # Inference toolchain msIT
|—————— msmodelslim              # Quantization tool msmodelslim
|—————— msprechecker             # Pre-check tool
|—————— msserviceprofiler        # Service profiler
|—————— test                     # UT test
|—————— README.md                # Repository overview
```

## Quick Start

This section uses a simple model as an example to demonstrate how to use the model quantization, data dump, precision comparison, and performance profiling tools from msIT. For details, see [Quick Start](./docs/en/msit_quick_start.md).

## Function Description

As a unified inference development toolchain for the Ascend platform, msIT contains model quantization, precision debugging, and performance profiling tools. You can choose a tool from the descriptions below to learn more and begin model inference.

### Performance profiling tools

- [**msProf (MindStudio Profiler)**](https://gitcode.com/Ascend/msprof)<br>
    **Data collection tool**: builds basic performance tuning capabilities for all Ascend scenarios, and collects CANN and NPU performance data to improve the tuning efficiency.

- [**msMonitor (MindStudio Monitor)**](https://gitcode.com/Ascend/msmonitor)<br>
    **Online monitoring tool**: one-stop online monitoring tool that supports both disk dump and online performance data collection, with performance monitoring and fault location capabilities in cluster scenarios.

- [**msServiceProfiler (MindStudio Service Profiler)**](https://gitcode.com/Ascend/msserviceprofiler)<br>
    **Service profiler**: Ascend-compatible profiler. It supports request scheduling and model execution visualization, improving service performance analysis efficiency.

- [msprechecker (MindStudio Prechecker Tool)](https://gitcode.com/Ascend/msit/tree/master/msprechecker)<br>
    **Pre-check tool**: msprechecker provides pre-check capabilities for inference scenarios, including environment validation, connectivity testing, and disk dump and comparison during inference. This helps users detect issues before deploying inference services and quickly establish performance baselines during inference.

- [**msprof-analyze (MindStudio Profiler Analyze)**](https://gitcode.com/Ascend/msprof-analyze)<br>
    **Ascend performance analysis tool**: analyzes collected profile data to quickly identify performance bottlenecks on Ascend devices.

- [msInsight (MindStudio Insight)](https://gitcode.com/Ascend/msinsight)<br>
    **MindStudio Insight**: supports multi-dimensional performance analysis across system, operator, and serving scenarios. It provides deep performance data analysis to help developers diagnose performance issues.

### Precision tool
    
- [msProbe (MindStudio Probe)](https://gitcode.com/Ascend/msprobe)<br>
    **Precision debugging tool**: toolkit for precision debugging during model development. It is a full-scenario precision toolchain for Ascend and helps users improve model precision debugging efficiency.

- [**msMemScope (MindStudio MemScope)**](https://gitcode.com/Ascend/msmemscope)<br>
    **Memory tool**: dedicated tool for Ascend memory debugging and tuning. It provides multi-dimensional memory data collection, automatic diagnosis, and optimization analysis across the entire network.

### Quantization tool

- [msModelSlim (MindStudio ModelSlim)](https://gitcode.com/Ascend/msmodelslim)<br>
    **Model compression tool**: The Ascend model compression tool is Ascend-native, with acceleration as its goal, compression as its technique, and Ascend as its foundation. It includes quantization, compression, and other inference optimizations, supporting large language dense models, MoE models, multi-modal understanding models, and multi-modal generation models.

## Security Statement 

Describes msIT-related security information, public network addresses, and communication matrix. For details, see [Security Statement] (./docs/zh/security_statement.md).

## Disclaimer

- This tool is intended for debugging and development purposes only and is not suitable for production environments. Users assume all risks associated with its use and acknowledge the following:

  - [x] For debugging and development only. This tool is designed to assist developers in debugging and is not intended for production or commercial use. The tool and its developers are not liable for any data loss or damage resulting from misuse.

  - [x] Data processing and deletion: Users are responsible for any data generated while using this tool, including but not limited to dumped data. You are advised to delete such data promptly after use to prevent information leakage.

  - [x] Data confidentiality and distribution: Users understand and agree not to disclose or distribute any data generated by this tool. Neither the tool nor its developers are responsible for any information leaks, data breaches, or other negative consequences.

  - [x] User input security: Users are responsible for the security of any commands they enter and for any risks or losses resulting from improper input. The tool and its developers are not liable for issues caused by incorrect command usage.

- Disclaimer scope: This disclaimer applies to all individuals and entities using this tool. By using the tool, you acknowledge and accept this statement and assume all risks and responsibilities arising from its use. If you do not agree, please stop using the tool immediately.

- Before using this tool, **please read and understand the preceding disclaimer**. If you have any questions, contact the developer.

## License

For the license of msIT, see [LICENSE](./docs/LICENSE).

Documentation in the `docs` directory of msIT is subject to the CC-BY 4.0 license. For details, see [LICENSE](./docs/LICENSE).

## Contribution Statement

1. **Error report submission**: If you discover a vulnerability in msIT that is not a security issue, first search the **Issues** in the msIT repository to avoid submitting duplicates. If the vulnerability is not listed, create a new issue. If you discover a security-related problem, do not disclose it publicly. Please refer to the security handling guidelines for details. All error reports must include complete information about the issue.
2. **Security issue handling**: For guidance on handling security issues in this project, please contact the core team via email for instructions.
3. **Resolving existing issues**: Review the issue list of the repository to identify issues that need attention, and attempt to resolve them.
4. **Proposing new features**: Use the **Feature** tag when creating an issue for a new feature. We will review and confirm proposals periodically.
5. **How to contribute**:
    1. Fork the repository of the project.
    2. Clone it to your local machine.
    3. Create a development branch.
    4. Local testing: All unit tests, including any new test cases, must pass before submission.
    5. Submit your code.
    6. Create a pull request (PR).
    7. Code review: Modify the code according to review comments and resubmit your changes. This process may involve multiple iterations.
    8. After your PR is approved by the required number of reviewers, the committer will conduct the final review.
    9. After your PR is approved and all tests pass, the CI system will merge it into the main branch of the project.

## Suggestions and Feedback

You are welcome to contribute to the community. If you have any questions or suggestions, please submit [issues](https://gitcode.com/Ascend/msit/issues). We will reply as soon as possible. Thank you for your support.

## Acknowledgments

msIT is jointly developed by the following Huawei departments:

- Ascend Computing MindStudio Development Dept

Thank you to everyone in the community for your PRs. We warmly welcome contributions to msIT!
