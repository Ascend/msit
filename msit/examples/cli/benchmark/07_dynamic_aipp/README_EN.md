# Dynamic AIPP #

## Introduction ##

 *  Reference for Dynamic AIPP[What Is AIPP?](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasatc_16_0016.html)	.
 *  Currently, the benchmark tool supports only a single input model with dynamic AIPP configuration. Only static shape, dynamic batch, and dynamic width and height scenarios are supported. Dynamic shape scenarios are not supported.

## Running an Example ##

### \--aipp-config Input .config file template ###

The following uses the Aipp configuration (actual_aipp_conf.config) corresponding to the ResNet18 model as an example:

```
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

 *  .config File`[aipp_op]`For the names and value ranges of the fields under, see[Static AIPP Configuration Example](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasatc_16_0019.html)	To be associated with the[Dynamic AIPP Configuration Example](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasatc_16_0020.html)	.
 *  .config File`[aipp_op]`lower`input_format`,`src_image_size_w`,`src_image_size_h`Field is required.
 *  The benchmark does not check whether the field values in the .config file adapt to the corresponding model. During inference, the ACL interface reports an error that does not belong to the benchmark problem.

### 1. Example of the static shape scenario (ResNet18 model is used as an example). ###

#### Convert the static shape model with dynamic aipp configuration by running the atc command. ####

```
atc --framework=5 --model=./resnet18.onnx --output=resnet18_bs4_dym_aipp --input_format=NCHW --input_shape="image:4,3,224,224" --soc_version=<soc_version> --insert_op_conf=dym_aipp_conf.aippconfig
```

 *  The content of dym_aipp_conf.aippconfig is as follows:

```
aipp_op{
    related_input_rank : 0
    aipp_mode : dynamic
    max_src_image_size : 4000000
}
```

#### Benchmark Command ####

```
msit benchmark --om-model resnet18_bs4_dym_aipp.om --aipp-config actual_aipp_conf.config
```

### 2. Dynamic batch scenario (ResNet18 model is used as an example.) ###

#### Run the atc command to convert the dynamic batch model with dynamic aipp configuration. ####

```
atc --framework=5 --model=./resnet18.onnx --output=resnet18_dym_batch_aipp --input_format=NCHW --input_shape="image:-1,3,224,224" --dynamic_batch_size "1,2" --soc_version=<soc_version> --insert_op_conf=dym_aipp_conf.aippconfig
```

#### Benchmark Command ####

```
msit benchmark --om-model resnet18_dym_batch_aipp.om --aipp-config actual_aipp_conf.config --dym-batch 1
```

### 3. Dynamic width and height scenario example, using the ResNet18 model as an example. ###

#### Convert the dynamic width and height model with dynamic aipp configuration by running the atc command. ####

```
atc --framework=5 --model=./resnet18.onnx --output=resnet18_dym_image_aipp --input_format=NCHW --input_shape="image:4,3,-1,-1" --dynamic_image_size "112,112;224,224" --soc_version=<soc_version> --insert_op_conf=dym_aipp_conf.aippconfig
```

#### Benchmark Command ####

```
msit benchmark --om-model resnet18_dym_image_aipp.om --aipp-config actual_aipp_conf.config --dym-hw 112,112
```

## FAQ ##

If a problem occurs, refer to.[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)	

