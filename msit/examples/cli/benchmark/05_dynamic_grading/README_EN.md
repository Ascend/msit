# Dynamic grading #

## Introduction ##

Dynamic rank scenario. Dynamic Batch, Dynamic HW (width and height), and Dynamic Dims scenarios are involved. --dym-batch, --dym-hw, and --dym-dims need to be transferred respectively to specify the actual level information.

## Running an Example ##

1.  Dynamic Batch.
    
    Take gear 1 2 4 8 as an example, set gear 2 and this program will obtain the actual model input group Batch, every 2 inputs into a group for group Batch.
    
    ```
    msit benchmark --om-model ./resnet50_v1_dynamicbatchsize_fp32.om --input=./data/ --dym-batch 2
    ```
2.  Dynamic HW width and height.
    
    in gear 224, 224; For example, if you set 448 and 448 to 224 and 224, the program will obtain the actual model input group Batch.
    
    ```
    msit benchmark --om-model ./resnet50_v1_dynamichw_fp32.om --input=./data/ --dym-hw 224,224
    ```
3.  Dynamic Dims.
    
    For example, if you set level 1, 3, 224, 224, this program will obtain the actual model input group Batch.
    
    ```
    msit benchmark --om-model resnet50_v1_dynamicshape_fp32.om --input=./data/ --dym-dims actual_input_1:1,3,224,224
    ```
4.  Automatically sets the Dims mode (dynamic Dims model).
    
    The shape of the input data of the dynamic dims model may not be fixed. For example, the shape of one input file is 1, 3, 224, and 224, and the shape of the other input file is 1, 3, 300, and 300. If two files are inferred at the same time, you need to set the dynamic Shape parameter twice. Currently, this operation is not supported. For this scenario, the --auto-set-dymdims-mode mode is added. In this mode, the Shape parameter of the model can be automatically set based on the Shape information in the input file.
    
    ```
    msit benchmark --om-model resnet50_v1_dynamicshape_fp32.om --input=./data/ --auto-set-dymdims-mode 1
    ```

 *  Note: In the example, the ./data/ folder stores user input data in .npy format. If no input data is specified, random input data is automatically generated.

## FAQ ##

If a problem occurs, refer to.[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)	

