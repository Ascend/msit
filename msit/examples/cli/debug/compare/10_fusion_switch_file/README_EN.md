# Disabling Convergence Rule Comparison #

## Introduction ##

The precision comparison tool disables the fusion rule comparison mode. Generally, the operator fusion function is enabled by default during offline model precision comparison during offline model conversion. To check the accuracy of the fused operators, you can disable the operator fusion function during model conversion.

(1) For the OM model, generate dump data files that are integrated and compared with dump data files that are integrated.
(2) For the TensorFlow framework, generate integrated dump data files and compare them with the benchmark model.

This function is disabled by default. If this function is enabled, run the following command:`--fusion-switch-file ./fusion_switch.cfg`\[fusion_switch.cfg\] is the configuration file of the Ascend convergence rule. For details about how to configure the configuration, see.[Disabling a Convergence Rule](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000105.html)	

**Note:**

 *  Do not turn off when in use`--dump`Don't turn it on.`--custom-op`,`--locat`,`--single-op`For advanced functions, the input parameter \[-gm, --golden-model\] must be the .onnx model.

## Running an Example ##

For example, if you disable all convergent rules, the configuration method of the fusion_switch.cfg file is as follows:

```
{
    "Switch":{
        "GraphFusion":{
            "ALL":"off"
        },
        "UBFusion":{
            "ALL":"off"
         }
    }
}
```

Place the fusion_switch.cfg file in the current directory by running the following command:

```
msit debug compare -gm {onnx_model_path} -om {om_model_path} -o {output_file_path} --fusion-switch-file ./fusion_switch.cfg
```

## The result ##

 *  Path to the result_\*.csv file of the Tensor comparison result. For details, see.[Procedure for Comparing Result Analysis](../result_analyse/README.md)	.
 *  The output parameters in the CSV table are different from those in the ONNX model. \[GroundTruth\] indicates the operator name of the OM offline model for which operator fusion is disabled.

| Parameter   | Description                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------------------- |
| NPUDump     | Indicates the operator name of the offline model for which the operator fusion function is enabled.  |
| GroundTruth | Indicates the operator name of the offline model for which the operator fusion function is disabled. |

