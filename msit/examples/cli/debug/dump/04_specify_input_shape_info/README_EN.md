# Specify Input Shape Info #

## Introduction ##

Specifies the input shape information of a model. (This parameter must be specified in dynamic scenarios.)

## Running an Example ##

**Note:**

 *  If a scalar vector (shape is ()) exists in the input, you do not need to specify the shape of the vector.

--------------------

1.  Specify -is or --input-shape for precision comparison, for example, input_name1:1,3. Input_name must be the node name defined in the model.
    
    ```
    msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
    -is "image:1,3,224,224"
    ```
    
    If the model is a dynamic shape model, the input shape information is used for model inference and precision comparison.
2.  Specify -dr or --dym-shape-range to compare the precision of multiple shapes in a dynamic model. (This parameter has a higher priority than -is and --input-shape.) The usage of this parameter is the same as that of -is.

```
msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
-dr "image:1,3,224-256,224~226"
```

 *  In the preceding example, input_name is specified as`image`, where input_name must be the node name in the model. "~" indicates the range. a-b-c indicates \[a: b:c\]. The hyphen (-) indicates the value of a bit. The preceding dump process is performed for six times. The dump process is performed when the input value is \["image:1,3,224,224","image:1,3,224,225","image:1,3,224,226","image:1,3,256,224","image:1,3,256,225","image:1,3,256,226"\].

