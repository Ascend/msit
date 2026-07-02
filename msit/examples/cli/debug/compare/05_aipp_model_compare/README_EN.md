# Aipp Model Compare #

## Introduction ##

Precision comparison between the original ONNX model and the offline OM model converted after the aipp option is enabled is supported.

## Running an Example ##

### Preparations ###

Use first[ATC tool](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/atc/atlasatc_16_0005.html)	Convert an O M model that does not merge operators.

```
atc --framework 5 --model=./resnet18.onnx --output=resnet18_bs8 --input_format=NCHW \
--input_shape="image:8,3,224,224" --log=debug --soc_version=<soc_version> \
--insert_op_conf=aipp.config --fusion_switch_file=fusionswitch.cfg
```

The fusionswitch.cfg file (operator not integrated) contains the following information:

```
{
    "Switch":{
        "GraphFusion":{
            "ALL":"off"
        },
        "UBFusion":{
            "ALL":"off"
        }
    }
}
```

A configuration example of aipp.config is as follows:

```
aipp_op{
    aipp_mode:static
    input_format : RGB888_U8

    src_image_size_w : 256
    src_image_size_h : 256

    crop: true
    load_start_pos_h : 16
    load_start_pos_w : 16
    crop_size_w : 224
    crop_size_h: 224

    min_chn_0 : 123.675
    min_chn_1 : 116.28
    min_chn_2 : 103.53
    var_reci_chn_0: 0.0171247538316637
    var_reci_chn_1: 0.0175070028011204
    var_reci_chn_2: 0.0174291938997821
}
```

### Command line operations ###

```
msit debug compare -gm ./resnet18.onnx -om ./resnet18_bs8.om -is "image:8,3,224,224"
```

\-gm indicates the benchmark ONNX model (mandatory). (Mandatory) -om: Enter the generated O M model that is not integrated with the operator. (Mandatory) -is indicates the input shape information of the ONNX model. If you need to specify the input (optional), use the -i parameter to specify the input (npy or bin file) of the OM model.

