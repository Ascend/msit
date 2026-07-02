# Saved model compare #

## Introduction ##

Supports precision comparison between benchmark models in save_model format and OM models.

## Use Example ##

 *  1. Example of the saved_model compare command. The path must be an absolute path.
    
    ```
    msit debug compare -gm /home/HwHiAiUser/prouce_data/resnet_offical_saved_model -om /home/HwHiAiUser/prouce_data/model/resnet50.om
    -saved_model_signature serving -saved_model_tag_set serve -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test -is "模型的输入shape"
    ```
    
     *  `-gm, –-golden-model`Specify the original saved_model model path.
     *  `-om, –-om-model`Specifies the offline model (.om) path of the Ascend AI processor.
     *  `-saved_model_signature`(Optional) Signature required for loading the saved_model model in the TensorFlow 2.6 framework.
     *  `-saved_model_tag_set`(Optional) Label for loading the saved_model model as a session in the TensorFlow 2.6 framework. You can load different parts of the model based on the label.
     *  `-c, –-cann-path`(Optional) Specify`CANN`Path after the package is installed. If the path is not specified, the system environment variables are obtained by default.`ASCEND_TOOLKIT_HOME`Obtained from`CANN`Package path. If the package path does not exist, the default value is`/usr/local/Ascend/ascend-toolkit/latest`
     *  `-o, –-output`(Optional) Output file path. The default value is the current path.
     *  `-is, --input-shape`(Optional) Input shape of the model

