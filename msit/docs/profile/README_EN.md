# Usage Description of the msit profile

## Brief Introduction

 * Collects and analyzes performance data for model inference on Ascend devices.

## Tool installation

 * Install the msit tool. For details, see.[MSIT Integrated Tool User Guide](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md)    
 * After installing the msit, install the profile tool.

```bash
msit install profile
```

## Function Description

### msprof

Integrates the performance collection and analysis tool msprof, which is used to analyze the key performance data in each running phase of the APP project running on the Ascend AI processor.[msit profile msprof quick start guide](./msprof/README.md)    

### analyze

Analyzes the collected profiling data in the context of recommendation scenarios and graph mode inference, and outputs performance analysis reports to guide model performance optimization.[Performance Comparison Quick Start Guide](./analyze/README.md)    
