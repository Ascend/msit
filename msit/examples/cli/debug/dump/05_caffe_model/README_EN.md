# Caffe Dump #

## Introduction ##

Tensor data dumping can be performed on a specified Caffe model.

## Example 1 ##

 *  For details about how to install the Caffe environment, see:[MSIT Integrated Tool User Guide](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md)	
 *  Currently, Caffe models do not support dynamic shape model comparison.`yolov2`/`yolov3`/`ssd`For models that need to customize the implementation layer, you need to compile and install the Caffe of a specific version.
 *  Prepare the Caffe model structure file and weight file. For details about the model structure file, see.[Caffe model structure file example](#caffe-模型结构文件示例)	. This file is used to define the model and save the weights that are initialized randomly. In actual use, the existing model structure file is used.`.prototxt`and Weight File`.caffemodel`
    
    ```
    import caffe
    
    model_path = "caffe_demo.prototxt"
    weight_path = "caffe_demo.caffemodel"
    
    net = caffe.Net(model_path, caffe.TEST)
    net.save(weight_path)
    ```
 *  **Precision comparison call**
    
    ```
    mkdir -p test  #Path for storing dump data and comparison results.
    ASCEND_TOOLKIT=$HOME/Ascend/ascend-toolkit/latest  #Must be a writable CANN package path.
    msit debug dump -m caffe_demo.prototxt -w caffe_demo.caffemodel -dp cpu -c $ASCEND_TOOLKIT -o ./test
    ```
 *  **Output Directory Structure Reference**[01_basic_usage](../01_basic_usage/README.md)	. The dump data of the Caffe model is stored in the`{output_path}/{timestamp}/dump_data/caffe/`

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

