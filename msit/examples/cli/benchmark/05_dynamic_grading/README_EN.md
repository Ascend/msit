# Dynamic Grading

## Introduction

Dynamic grading scenarios. This mainly includes three scenarios: dynamic batch, dynamic HW (height and width), and dynamic dims. You need to pass --dym-batch, --dym-hw, and --dym-dims respectively to specify the actual gear information.

## Running Samples

1. Dynamic batch.

    Take the gears 1, 2, 4, and 8 as an example. Set the gear to 2. This program obtains the actual model input batch, grouping every 2 inputs as a batch.

    ```bash
    msit benchmark --om-model ./resnet50_v1_dynamicbatchsize_fp32.om --input=./data/ --dym-batch 2
    ```

2. Dynamic HW (height and width).

    Take the gears 224,224 and 448,448 as an example. Set the gear to 224,224. This program obtains the actual model input batch.

    ```bash
    msit benchmark --om-model ./resnet50_v1_dynamichw_fp32.om --input=./data/ --dym-hw 224,224
    ```

3. Dynamic dims.

   Take the gear 1,3,224,224 as an example. This program obtains the actual model input batch.

   ```bash
   msit benchmark --om-model resnet50_v1_dynamicshape_fp32.om --input=./data/ --dym-dims actual_input_1:1,3,224,224
   ```

4. Auto-set dims mode (dynamic dims model).

    The shape of the input data for a dynamic dims model may vary. For example, one input file has the shape 1,3,224,224, and another input file has the shape 1,3,300,300. If both files are inferred at the same time, the dynamic shape parameter needs to be set twice, which is currently not supported. For this scenario, the --auto-set-dymdims-mode mode is added to automatically set the shape parameters of the model based on the shape information of the input file.

    ```bash
    msit benchmark --om-model resnet50_v1_dynamicshape_fp32.om --input=./data/ --auto-set-dymdims-mode 1
    ```

- Description: The ./data/ directory in the sample stores the user input data in .npy format. If no input data is specified, random input data is automatically generated.

## FAQ

For issues during use, refer to the [FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
