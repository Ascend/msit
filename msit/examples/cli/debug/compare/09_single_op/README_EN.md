# Single operator comparison #

## Introduction ##

The precision comparison tool supports single-operator parallel comparison of models, avoiding the impact of the previous operator on the input and output of subsequent operators. This function is disabled by default. If this function is enabled, run the following command:`-single True`or the`--single-op True`

## Running an Example ##

**Prerequisites: Dynamic shape models are not supported currently. The npy file may fail to be loaded during the running.**`--onnx-fusion-switch False`To resolve. Note:

 *  Do not set this parameter when using`--dump 为False`. The default value is True.
 *  Don't turn on`--custom-op`
 *  Don't turn on`--locat`Run the following command:

```
msit debug compare -gm {onnx_model_path} -om {om_model_path} -i {input_data_path} -o {output_file_path} -single True
```

## The results ##

The output file is stored in the single_op folder in the output path. The comparison result is summarized in the`single_op_summary.csv`in the file. For details about the comparison results, see.[Procedure for Comparing Result Analysis](../result_analyse/README.md)	

