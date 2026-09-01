# Dynamic Shapes

## Introduction

Dynamic shape scenarios. This mainly includes three scenarios: dynamic shape, auto-set shape mode (dynamic shape model), and dynamic shape model range test mode. You need to pass --dym-shape, --auto-set-dymshape-mode, and --dym-shape-range respectively to specify the dynamic shape information.

## Running Samples

1. Dynamic shape scenario.

    Take the ATC setting [1\~8,3,200\~300,200\~300] with the gear 1,3,224,224 as an example. This program obtains the actual model input batch.

    The output size of a dynamic shape is usually 0. It is recommended to set the corresponding output memory size through the output-size parameter.

    ```bash
    msit benchmark --om-model resnet50_v1_dynamicshape_fp32.om --dym-shape actual_input_1:1,3,224,224 --output-size 10000
    ```

2. Auto-set shape mode (dynamic shape model).

    The shape of the input data for a dynamic shape model may vary. For example, one input file has the shape 1,3,224,224, and another input file has the shape 1,3,300,300. If both files are inferred at the same time, the dynamic shape parameter needs to be set twice, which is currently not supported. For this scenario, the --auto-set-dymshape-mode mode is added to automatically set the shape parameters of the model based on the shape information of the input file.

    ```bash
    msit benchmark --om-model ./pth_resnet50_dymshape.om  --output-size 100000 --auto-set-dymshape-mode 1  --input ./dymdata
    ```

    **Note that the input file in this scenario must be in npy format. If it is a bin file, the actual shape information cannot be obtained.**

3. Single-input dynamic shape model range test mode.

    Enter the range of the dynamic shape. For each shape within the range, inference is performed separately to obtain the respective performance metrics.

    Take the inference of 1,3,224,224, 1,3,224,225, and 1,3,224,226 as an example. The command is as follows:

    ```bash
    msit benchmark --om-model ./pth_resnet50_dymshape.om  --output-size 100000 --dym-shape-range actual_input_1:1,3,224,224~226
    ```

4. Multi-input dynamic shape model range test mode (command line).

    Enter the range of the dynamic shape. For each shape within the range, inference is performed separately to obtain the respective performance metrics.

    Take a dual-input model as an example. The first input is 1,3,224,224, 1,3,224,225, and 1,3,224,226, and the second input is 1,3,224,224, 2,3,224,224, and 3,3,224,224. The command for separate inference is as follows:

    ```bash
    msit benchmark --om-model ./pth_resnet50_dymshape_dual_input.om  --output-size 100000 --dym-shape-range "actual_input_1:1,3,224,224~226;actual_input_2:1~3,3,224,224"
    ```

5. Multi-input dynamic shape model range test mode (*.info).

    Enter the range of the dynamic shape. For each shape within the range, inference is performed separately to obtain the respective performance metrics.

    Take a dual-input model as an example. Inference is performed separately for two groups of inputs.
    <br/>The first group of inputs is:
    <br/>The first input is 1,3,224,224, 1,3,224,225, and 1,3,224,226, and the second input is 1,3,224,224, 2,3,224,224, and 3,3,224,224.
    <br/>The second group of inputs is:
    <br/>The first input is 1,3,224,224, 2,3,224,224, and 3,3,224,224, and the second input is 1,3,224,224, 1,3,224,225, and 1,3,224,226.
    <br/>First create an info file named dual_input.info with the following content:
   <br/> actual_input_1:1,3,224,224\~226;actual_input_2:1\~3,3,224,224
   <br/> actual_input_1:1\~3,3,224,224;actual_input_2:1,3,224,224\~226

    The command is as follows:

    ```bash
    msit benchmark --om-model ./pth_resnet50_dymshape_dual_input.om  --output-size 100000 --dym-shape-range dual_input.info
    ```

    >Note: actual_input is the actual input name of the model
 >
## FAQ

For issues during use, refer to the [FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
