# Dynamic AIPP

## Introduction

- For an introduction to dynamic AIPP, refer to [What Is AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasatc_16_0016.html).
- Currently, the benchmark tool only supports models with dynamic AIPP configuration for a single input. It supports three scenarios: static shape, dynamic batch, and dynamic HW. It does not support the dynamic shape scenario.

## Running Samples

### --aipp-config Input .config File Template

Take a specific AIPP configuration for the resnet18 model as an example (actual_aipp_conf.config):

```cfg
[aipp_op]
    input_format : RGB888_U8
    src_image_size_w : 256
    src_image_size_h : 256

    crop : 1
    load_start_pos_h : 16
    load_start_pos_w : 16
    crop_size_w : 224
    crop_size_h : 224

    padding : 0
    csc_switch : 0
    rbuv_swap_switch : 0
    ax_swap_switch : 0
    csc_switch : 0

   min_chn_0 : 123.675
   min_chn_1 : 116.28
   min_chn_2 : 103.53
   var_reci_chn_0 : 0.0171247538316637
   var_reci_chn_1 : 0.0175070028011204
   var_reci_chn_2 : 0.0174291938997821
```

- For the field names and value ranges under `[aipp_op]` in the .config file, refer to [Static AIPP Configuration Example](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasatc_16_0019.html) and [Dynamic AIPP Configuration Example](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasatc_16_0020.html) in the CANN manual.
- The `input_format`, `src_image_size_w`, and `src_image_size_h` fields under `[aipp_op]` in the .config file are required fields.
- The benchmark tool itself does not check whether the specific field values in the .config file are suitable for the corresponding model. Errors reported by the acl interface during inference are not benchmark issues.

### 1. Static Shape Scenario Example, Using the resnet18 Model

#### ATC command to convert a static shape model with dynamic AIPP configuration

```cfg
atc --framework=5 --model=./resnet18.onnx --output=resnet18_bs4_dym_aipp --input_format=NCHW --input_shape="image:4,3,224,224" --soc_version=<soc_version> --insert_op_conf=dym_aipp_conf.aippconfig
```

- The content of dym_aipp_conf.aippconfig (same below) is:

```cfg
aipp_op{
    related_input_rank : 0
    aipp_mode : dynamic
    max_src_image_size : 4000000
}
```

#### Benchmark command

```cfg
msit benchmark --om-model resnet18_bs4_dym_aipp.om --aipp-config actual_aipp_conf.config
```

### 2. Dynamic Batch Scenario Example, Using the resnet18 Model

#### ATC command to convert a dynamic batch model with dynamic AIPP configuration

```cfg
atc --framework=5 --model=./resnet18.onnx --output=resnet18_dym_batch_aipp --input_format=NCHW --input_shape="image:-1,3,224,224" --dynamic_batch_size "1,2" --soc_version=<soc_version> --insert_op_conf=dym_aipp_conf.aippconfig
```

#### Benchmark command

```cfg
msit benchmark --om-model resnet18_dym_batch_aipp.om --aipp-config actual_aipp_conf.config --dym-batch 1
```

### 3. Dynamic HW Scenario Example, Using the resnet18 Model

#### ATC command to convert a dynamic HW model with dynamic AIPP configuration

```cfg
atc --framework=5 --model=./resnet18.onnx --output=resnet18_dym_image_aipp --input_format=NCHW --input_shape="image:4,3,-1,-1" --dynamic_image_size "112,112;224,224" --soc_version=<soc_version> --insert_op_conf=dym_aipp_conf.aippconfig
```

#### Benchmark command

```cfg
msit benchmark --om-model resnet18_dym_image_aipp.om --aipp-config actual_aipp_conf.config --dym-hw 112,112
```

## FAQ

For issues during use, refer to the [FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
