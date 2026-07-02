# Caffe one-click precision comparison #

## Introduction ##

Supports one-click precision comparison for Caffe benchmark models.

## Example 1 ##

 *  For details about how to install the Caffe environment, see:[MSIT Integrated Tool User Guide](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md)	
 *  Currently, Caffe models do not support dynamic shape model comparison.`yolov2`/`yolov3`/`ssd`For models that need to customize the implementation layer, you need to compile and install the Caffe of a specific version.
 *  Prepare the Caffe model structure file and weight file. For details about the model structure file, see.[Caffe model structure file example](#caffe-模型结构文件示例)	. This file is used to define the model and save the weights initialized randomly. In actual use, the existing model structure file should be used.`.prototxt`and Weight File`.caffemodel`
    
    ```
    import caffe
    
    model_path = "caffe_demo.prototxt"
    weight_path = "caffe_demo.caffemodel"
    
    net = caffe.Net(model_path, caffe.TEST)
    net.save(weight_path)
    ```
 *  Use the`atc`Run the following command to convert the Caffe model to the OM model:
    
    ```
    atc --model=caffe_demo.prototxt --weight=caffe_demo.caffemodel --framework=0 --soc_version=<soc_version> --output=caffe_demo
    ```
 *  Note: Execute the quantization original model (GPU/CPU) vs the quantization offline model (disable the fusion rule) (NPU). Operator fusion is enabled by default for ATC model conversion. To avoid the failure of precision comparison between operators after fusion, disable operator fusion first. For details, see the following section.[How Do I Disable or Enable Convergence Rules?](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/graph/graphubfusionref/atlasrr_30_0003.html#ZH-CN_TOPIC_0000002263813281__section15766181519012)	
    
    ```
    atc --model=resnet50_deploy_model.prototxt --weight=resnet50_deploy_weights.caffemodel --framework=0   \
    --output=caffe_resnet50_off --soc_version=<soc_version>  --fusion_switch_file=fusion_switch.cfg
    ```

Note: To disable the operator convergence function, run the following command:`--fusion_switch_file`Specify the operator convergence rule configuration file (for example, fusion_switch.cfg) and disable the operator convergence in the configuration file. The configuration for disabling the convergent rule configuration file is as follows:

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

 *  **Precision comparison call**
    
    ```
    mkdir -p test  #Output path of dump data and comparison results
    ASCEND_TOOLKIT=$HOME/Ascend/ascend-toolkit/latest  #Must be a writable CANN package path.
    msit debug compare -gm caffe_demo.prototxt -w caffe_demo.caffemodel -om caffe_demo.om -c $ASCEND_TOOLKIT -o ./test
    #...
    #[INFO] Caffe input info:
    #[{'name':'data','shape':(1,3,32,32),'type':'float32'}]
    #...
    #[INFO] b'2023-05-31 05:56:20 (2749324) - [INFO] The command was completed and took 0 seconds.'
    #[INFO] Compare Node_output:0 completely.
    #[INFO] Analyser init parameter csv_path=-/workspace/caffe_dump/test/20230531055610/result_20230531055619.csv
    #[INFO] Analyser call parameter strategy=FIRST_INVALID_OVERALL, max_column_len=30
    #[INFO] None operator with accuracy issue reported
    ```
 *  **Output Directory Structure Reference**[01_basic_usage](../01_basic_usage/README.md)	. The dump data of the Caffe model is stored in the`{output_path}/{timestamp}/dump_data/caffe/`
 *  **The comparison result is located in**`{output_path}/{timestamp}/result_{timestamp}.csv`In, the meaning of the comparison result is the same as that of the basic precision comparison tool. For details about the meaning of each field, see.[Parameters in the complete comparison result](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/modelaccuracy/atlasaccuracy_16_0064.html)	

## Example 2 ##

 *  In the Caffe non-quantized original model vs. quantized offline model scenario, run the following command:`atc`Run the following command to convert the quantized Caffe model to the OM model:
    
    ```
    atc --model=caffe_demo.prototxt --weight=caffe_demo.caffemodel --framework=0 --soc_version=<soc_version> --output=caffe_demo
    ```
 *  Export the JSON file.
    
    ```
    atc --om caffe_demo.om --mode 1 --json=caffe_demo.json
    ```
 *  **Precision comparison call**
 *  Use the Caffe model before quantization.
    
    ```
    mkdir -p test  #Path for storing dump data and comparison results.
    ASCEND_TOOLKIT=$HOME/Ascend/ascend-toolkit/latest  #Must be a writable CANN package path.
    msit debug compare -gm ResNet-50-deploy.prototxt -w ResNet-50-model.caffemodel -om caffe_demo.om -c $ASCEND_TOOLKIT -o ./test -q caffe_demo.json
    ```

## Caffe Model Structure File Example ##

 *  `caffe_demo.prototxt`

```
name: "caffe_Demo"
layer {
  name: "Input_1"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 32
      dim: 32
    }
  }
}

layer {
        bottom: "data"
        top: "conv1"
        name: "conv1"
        type: "Convolution"
        convolution_param {
                num_output: 64
                kernel_size: 7
                pad: 3
                stride: 2
        }
}

layer {
        bottom: "conv1"
        top: "conv1"
        name: "bn_conv1"
        type: "BatchNorm"
        batch_norm_param {
                use_global_stats: true
        }
}

layer {
        bottom: "conv1"
        top: "conv1"
        name: "conv1_relu"
        type: "ReLU"
}

layer {
        bottom: "conv1"
        top: "pool5"
        name: "pool5"
        type: "Pooling"
        pooling_param {
                kernel_size: 16
                stride: 1
                pool: AVE
        }
}

layer {
        bottom: "pool5"
        top: "fc1000"
        name: "fc1000"
        type: "InnerProduct"
        inner_product_param {
                num_output: 1000
        }
}
```

