# Specify Input Data #

## Introduction ##

By default, data whose values are all 0s is constructed and sent to model inference. File input or folder input can be specified.

## Running an Example ##

1.  File input scenario.
    
    Use the --input parameter to specify the model input file. Use commas (,) to separate multiple files.
    
    In this scenario, the system compares the file input size with the actual model input size. If data is missing, the system automatically constructs data supplementation, which is called group batch.
    
    ```
    msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --input ./1.bin,./2.bin,./3.bin,./4.bin,./5.bin
    ```
    
     *  Note: The .bin file stores the tensor data entered by the user and can be generated in the following way. In the example, size and astype can be obtained by using the debug tool. The --input parameter is designed for users to specify input data.
    
    ```
    import numpy as np
    np.random.uniform(size=[32,32]).astype('float32').tofile('foo.bin')
    ```
2.  Folder input scenario.
    
    Use the --input parameter to specify the directory where the model input file is located. Separate multiple directories with commas (,).
    
    In this scenario, batch grouping is performed based on the file input size and model input size.
    
    ```
    msit benchmark --om-model ./resnet50_v1_bs1_fp32.om --input ./
    ```
    
     *  Note: 1. If there is no .bin file in the input. / folder, an error will be reported. Ensure that the ./ folder contains .bin data when inputting --input parameter. 2. The number of input models must be consistent with the number of input folders.
    
    For example, open a model using the Netron software to view the input of the model. For example, the bert model has three inputs: input_ids, input_mask, and segment_ids. Therefore, the input parameters must be transferred to three folders. In addition, the three folders correspond to the three inputs of the model respectively, and the sequence must be corresponding.
    
     *  The first folder ./data/SQuAD1.1/input_ids" corresponds to the input of the first parameter input_ids of the model.
     *  The second folder "./data/SQuAD1.1/input_mask" corresponds to the input of the input_mask parameter.
     *  The third folder "./data/SQuAD1.1/segment_ids" corresponds to the input of the third parameter segment_ids.
    
    ```
    msit benchmark --om-model ./save/model/BERT_Base_SQuAD_BatchSize_1.om --input ./data/SQuAD1.1/input_ids,./data/SQuAD1.1/input_mask,./data/SQuAD1.1/segment_ids
    ```

## FAQ ##

If a problem occurs, refer to.[FAQ](https://gitcode.com/Ascend/msit/wiki/benchmark_FAQ%2Fait%20benchmark%20%E4%BD%BF%E7%94%A8%E8%BF%87%E7%A8%8B%20FAQ.md)	

