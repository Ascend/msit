# Multi Thread

## Introduction

The benchmark inference tool currently provides multi-thread inference functionality.

## Running Samples

1. Pure inference scenario. By default, the output information is only displayed on the screen.

    ```bash
    msit benchmark -om ./pth_resnet50_bs1.om --pipeline 1
    ```

    where -om is the OM offline model file path.

2. Inference with data scenario. By default, the output information is only displayed on the screen.

    ```bash
    msit benchmark -om ./pth_resnet50_bs1.om --input=./data --pipeline 1
    ```

    where --input is the input path, separated by commas.

3. Debug mode. Enable the debug mode.

    ```bash
    msit benchmark -om ./pth_resnet50_bs1.om --input=./data --debug=1 --pipeline 1
    ```

    After the debug mode is enabled, more log information is displayed, including:
   - Model input and output parameter information

     ```bash
     input:
       #0    input_ids  (1, 384)  int32  1536  1536
       #1    input_mask  (1, 384)  int32  1536  1536
       #2    segment_ids  (1, 384)  int32  1536  1536
     output:
       #0    logits:0  (1, 384, 2)  float32  3072  3072
     ```

   - Detailed inference time information

     ```bash
     [DEBUG] model aclExec cost : 2.336000
     ```

   - Detailed operation information such as model input and output

4. Save results scenario.

    ```bash
    msit benchmark -om ./pth_resnet50_bs1.om --input=./data --output=./result/ --pipeline 1
    ```

    where --output is the save folder path.

   - Sample

    ```bash
    # The input directory contents are as follows
    ls ./data/
    196608-0.bin  196608-1.bin  196608-2.bin  196608-3.bin  196608-4.bin  196608-5.bin  196608-6.bin  196608-7.bin  196608-8.bin  196608-9.bin
    ```

    ```bash
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

5. Dynamic shape scenario

    Take the ATC setting [1\~8,3,200\~300,200\~300] with the gear 1,3,224,224 as an example. This program obtains the actual model input batch.

    The output size of a dynamic shape is usually 0. It is recommended to set the corresponding output memory size through the outputSize parameter.

    ```bash
    msit benchmark -om ./pth_resnet50_dymshape.om --input ./data/ --dym-shape actual_input_1:1,3,224,224 --output-size 10000 --pipeline 1
    ```

6. Auto-set shape mode (dynamic shape model).

    The shape of the input data for a dynamic shape model may vary. For example, one input file has the shape 1,3,224,224, and another input file has the shape 1,3,300,300. If both files are inferred at the same time, the dynamic shape parameter needs to be set twice, which is currently not supported. For this scenario, the --auto-set-dymshape-mode mode is added to automatically set the shape parameters of the model based on the shape information of the input file.

    ```bash
    msit benchmark -om ./pth_resnet50_dymshape.om --input ./data --output-size 10000 --auto-set-dymshape-mode 1 --pipeline 1
    ```

7. Multi-compute thread inference scenario.

    You can set the number of compute threads for multi-thread inference by additionally setting the --threads parameter. This achieves compute-compute parallelism and improves inference throughput.

    ```bash
    msit benchmark -om ./pth_resnet50_bs1.om --input ./data --pipeline 1 --threads 2
    ```
