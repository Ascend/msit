# Specify Input Data

## Introduction

By default, data filled with zeros is constructed and sent to the model for inference. You can specify file input or directory input.

## Running Samples

1. File input scenario.

    Use the --input parameter to specify the model input file. Multiple files are separated by commas.

    This scenario compares the input file size with the actual model input size. If data is missing, data is automatically constructed to complete the batch. This process is called batch grouping.

    ```bash
    msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --input ./1.bin,./2.bin,./3.bin,./4.bin,./5.bin
    ```

   - Description:
    The .bin file stores the tensor data entered by the user. You can generate it using the following method. The size and astype in the sample can be obtained through the debug mode tool. The --input parameter is designed for users to specify input data.

    ```python
    import numpy as np
    np.random.uniform(size=[32,32]).astype('float32').tofile('foo.bin')
    ```

2. Directory input scenario.

    Use the --input parameter to specify the directory where the model input files are located. Multiple directories are separated by commas.

    This scenario groups batches based on the input file size and the actual model input size.

    ```bash
    msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --input ./
    ```

   - Description:
     1. If no .bin file exists in the specified ./ directory, an error is reported. Ensure that .bin data exists in the ./ directory when passing the --input parameter.
     2. The number of model inputs must match the number of directories passed.

    For example, open the model with the Netron software to view the model inputs. The BERT model has three inputs: input_ids, input_mask, and segment_ids. Therefore, you must pass three directories, and the three directories correspond to the three inputs of the model in order.

    - The first directory "./data/SQuAD1.1/input_ids" corresponds to the input of the first parameter "input_ids" of the model
    - The second directory "./data/SQuAD1.1/input_mask" corresponds to the input of the second parameter "input_mask"
    - The third directory "./data/SQuAD1.1/segment_ids" corresponds to the input of the third parameter "segment_ids"

    ```bash
    msit benchmark --om-model ./save/model/BERT_Base_SQuAD_BatchSize_1.om --input ./data/SQuAD1.1/input_ids,./data/SQuAD1.1/input_mask,./data/SQuAD1.1/segment_ids
    ```

## FAQ

For issues during use, refer to the [FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)
