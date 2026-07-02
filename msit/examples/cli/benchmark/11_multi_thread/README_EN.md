# Multi thread #

## Introduction ##

Currently, the benchmark inference tool provides the multi-thread inference function.

## Running an Example ##

1.  Pure inference scenario. By default, the output information is displayed only on the screen.
    
    ```
    msit benchmark -om ./pth_resnet50_bs1.om --pipeline 1
    ```
    
    In the preceding command, -om indicates the path of the OM offline model file.
2.  Data inference is required. By default, the output information is displayed only on the screen.
    
    ```
    msit benchmark -om ./pth_resnet50_bs1.om --input=./data --pipeline 1
    ```
    
    In the preceding command, --input indicates the input path, which is separated by commas (,).
3.  Enables the debug mode.
    
    ```
    msit benchmark -om ./pth_resnet50_bs1.om --input=./data --debug=1 --pipeline 1
    ```
    
    After the debugging mode is enabled, more information is displayed, including:
    
     *  Input and output parameters of the model
        
        ```
        input:
          #0 input_ids (1, 384) int32 1536 1536
          #1 input_mask (1, 384) int32 1536 1536
          #2 segment_ids (1, 384) int32 1536 1536
        output:
          #0 logits:0 (1, 384, 2) float32 3072 3072
        ```
     *  Detailed inference time information
        
        ```
        [DEBUG] model aclExec cost : 2.336000
        ```
     *  Detailed operation information such as model input and output
4.  Save the result scenario.
    
    ```
    msit benchmark -om ./pth_resnet50_bs1.om --input=./data --output=./result/ --pipeline 1
    ```
    
    In the preceding command, --output indicates the path of the folder.
    
     *  Example
    
    ```
    #The input folder is as follows:
    ls ./data/
    196608-0.bin  196608-1.bin  196608-2.bin  196608-3.bin  196608-4.bin  196608-5.bin  196608-6.bin  196608-7.bin  196608-8.bin  196608-9.bin
    ```
    
    ```
    result/
    |-- 2023_01_03-06_35_53
    |   |-- 196608-0_0.bin
    |   |-- 196608-1_0.bin
    |   |-- 196608-2_0.bin
    |   |-- 196608-3_0.bin
    |   |-- 196608-4_0.bin
    |   |-- 196608-5_0.bin
    |   |-- 196608-6_0.bin
    |   |-- 196608-7_0.bin
    |   |-- 196608-8_0.bin
    |   |-- 196608-9_0.bin
    |-- 2023_01_03-06_35_53_summary.json
    ```
5.  Dynamic shape scenario
    
    For example, if the ATC is set to \[1-8, 3, 200-300, 200-300\] and gears 1, 3, 224, 224, the program will obtain the actual model input group Batch.
    
    Generally, the output size of a dynamic shape is 0. You are advised to set the output memory size by setting the outputSize parameter.
    
    ```
    msit benchmark -om ./pth_resnet50_dymshape.om --input ./data/ --dym-shape actual_input_1:1,3,224,224 --output-size 10000 --pipeline 1
    ```
6.  Automatically sets the Shape mode (dynamic Shape model).
    
    The shape of the input data of the dynamic shape model may not be fixed. For example, the shape of one input file is 1, 3, 224, and the shape of another input file is 1, 3, 300, and 300. If two files are inferred at the same time, you need to set the dynamic Shape parameter twice. Currently, this operation is not supported. For this scenario, the --auto-set-dymshape-mode mode is added. In this mode, the Shape parameter of the model can be automatically set based on the Shape information in the input file.
    
    ```
    msit benchmark -om ./pth_resnet50_dymshape.om --input ./data --output-size 10000 --auto-set-dymshape-mode 1 --pipeline 1
    ```
7.  In the multi-computing thread inference scenario.
    
    You can set the --threads parameter to set the number of computing threads during multi-thread inference, implementing parallel computing and improving the inference throughput.
    
    ```
    msit benchmark -om ./pth_resnet50_bs1.om --input ./data --pipeline 1 --threads 2
    ```

