# Dynamic shapes #

## Introduction ##

Dynamic shape scenario. There are three scenarios: dynamic shape, automatic shape mode (dynamic shape model), and dynamic shape model range test mode. --dym-shape, --auto-set-dymshape-mode, and --dym-shape-range need to be transferred respectively to specify the dynamic shape information.

## Running an Example ##

1.  Dynamic Shape scenario.
    
    For example, if the ATC is set to \[1-8, 3, 200-300, 200-300\] and gears 1, 3, 224, 224, the program will obtain the actual model input group Batch.
    
    Generally, the output size of a dynamic shape is 0. You are advised to set the output memory size by setting the output-size parameter.
    
    ```
    msit benchmark --om-model resnet50_v1_dynamicshape_fp32.om --dym-shape actual_input_1:1,3,224,224 --output-size 10000
    ```
2.  Automatically sets the Shape mode (dynamic Shape model).
    
    The shape of the input data of the dynamic shape model may not be fixed. For example, the shape of one input file is 1, 3, 224, and the shape of another input file is 1, 3, 300, and 300. If two files are inferred at the same time, you need to set the dynamic Shape parameter twice. Currently, this operation is not supported. For this scenario, the --auto-set-dymshape-mode mode is added. In this mode, the Shape parameter of the model can be automatically set based on the Shape information in the input file.
    
    ```
    msit benchmark --om-model ./pth_resnet50_dymshape.om  --output-size 100000 --auto-set-dymshape-mode 1  --input ./dymdata
    ```
    
    **Note that the input file in this scenario must be in the .npy format. If the file is in the .bin format, the actual Shape information cannot be obtained.**
3.  Range test mode of the single-input dynamic Shape model.
    
    Enter the range of the dynamic shape. Perform inference on shapes within the range to obtain their performance indicators.
    
    For example, run the following commands to perform separate inference on 1,3,224,224 1,3,224,225 1,3,224,226:
    
    ```
    msit benchmark --om-model ./pth_resnet50_dymshape.om  --output-size 100000 --dym-shape-range actual_input_1:1,3,224,224~226
    ```
4.  Range test mode of the multi-input dynamic Shape model (command line)
    
    Enter the range of the dynamic shape. Perform inference on shapes within the range to obtain their performance indicators.
    
    For example, if the first input is 1,3,224,224 1,3,224,225 1,3,224,226 and the second input is 1,3,224,224 2,3,224,224 3,3,224,224, run the following commands to perform separate inference:
    
    ```
    msit benchmark --om-model ./pth_resnet50_dymshape_dual_input.om  --output-size 100000 --dym-shape-range "actual_input_1:1,3,224,224~226;actual_input_2:1~3,3,224,224"
    ```
5.  Range test mode of the multi-input dynamic Shape model (\*.info)
    
    Enter the range of the dynamic shape. Perform inference on shapes within this range to obtain their performance indicators.
    
    Taking a two-input model as an example, the inference is made for two sets of inputs respectively.
    The first set of inputs is:
    The first input is 1,3,224,224 1,3,224,225 1,3,224,226 and the second input is 1,3,224,224 2,3,224,224 3,3,224,224.
    The second set of inputs is:
    The first input is 1,3,224,224 2,3,224,224 3,3,224,224 and the second input is 1,3,224,224 1,3,224,225 1,3,224,226.
    Create an info file named dual_input.info, with the following contents:
    actual_input_1:1,3,224,224-226;actual_input_2:1-3,3,224,224
    actual_input_1:1-3,3,224,224;actual_input_2:1,3,224,224-226
    
    The command is as follows:
    
    ```
    msit benchmark --om-model ./pth_resnet50_dymshape_dual_input.om  --output-size 100000 --dym-shape-range dual_input.info
    ```
    
    > Note: actual_input indicates the actual input name of the model.

## FAQ ##

If a problem occurs, refer to.[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)	

