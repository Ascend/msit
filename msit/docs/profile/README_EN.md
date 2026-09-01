# msit profile Usage Instructions

## Introduction

- Performance data collection and analysis for model inference on Ascend devices.

## Tool Installation

- After installing msit, you also need to install the profile tool.

```bash
msit install profile
```

## Feature Introduction

### msprof

Integrates the performance collection and analysis tool msprof, which is used to analyze key performance data of each running phase of an APP project running on the Ascend AI processor. For details, refer to the [msit profile msprof Quick Start Guide](https://gitcode.com/Ascend/msprof/blob/26.1.0/docs/zh/quick_start/msprof_quick_start.md).

### analyze

Supports analyzing collected profiling data in recommendation scenarios and graph mode inference, and outputs performance analysis reports to guide model performance tuning. For details, refer to the [Performance Comparison Quick Start Guide](./analyze/README.md).
