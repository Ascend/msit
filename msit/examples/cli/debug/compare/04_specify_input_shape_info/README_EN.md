# Specify Input Shape Info #

## Introduction ##

Specifies the input shape information of a model. (This parameter must be specified in dynamic scenarios.)

## Running an Example ##

**Note:**

 *  If a scalar-like vector (shape is ()) exists in the input, you do not need to specify the shape of the vector.
 *  Currently, this function supports only the OM model converted from the input_shape_range parameter of atc.

--------------------

1.  Specify -is or --input-shape for precision comparison.
    
    ```
    msit debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
    -is "image:1,3,224,224"
    ```
    
    If the model is a dynamic shape model, the shape information input by the -is command is used for model inference and precision comparison.
2.  Specify -dr or --dym-shape-range to compare the precision of multiple shapes in a dynamic model. (The priority is higher than that of -is and --input-shape.)

```
msit debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
-dr "image:1,3,224-256,224~226"
```

 *  In the preceding command, input_name must be the node name in the network model before conversion. "~" indicates the range. a-b-c indicates \[a: b:c\]. The hyphen (-) indicates the value of a bit. In the preceding figure, six precision comparison processes are performed, and the input \["image:1,3,224,224","image:1,3,224,225","image:1,3,224,226","image:1,3,256,224","image:1,3,256,225","image:1,3,256,226"\] is compared respectively.

