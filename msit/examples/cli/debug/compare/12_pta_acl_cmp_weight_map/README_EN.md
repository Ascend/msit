# Precision Comparison Between PTA and ACL Acceleration Library Based on Weight Mapping #

 *  Compare the kernel data in the acceleration library and the weight (input) values of the PT/PTA API based on the weight (input) values to roughly determine the kernel and PTA APIs that are mapped.
 *  Step: Obtain the weight of the acceleration library and the PTA framework, and calculate the MD5 value. If the MD5 values are the same, it is determined that the corresponding acceleration library kernel has a mapping relationship with the PTA API.
 *  Limitation: This function applies only to matching with weights and mapping relationships.

## Comparison Process ##

![acl_pta_workflow.png](acl_pta_workflow.png)	

## Interface Description ##

 *  **set_dump_path(dump_path=".", dump_tag="ait_dump", backend="pt"): indicates the dump data directory. In multi-card inference, ensure that each process can invoke this function.**

| Parameter name | Meaning                                     | Mandatory or not | Instructions for use                                                                                                                                                         |
| -------------- | ------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dump_path      | Path for storing dump data.                 | No.              | Data type: str. You are advised to set this parameter when the data of different dialogs needs to be dumped. Otherwise, the data of the previous dialog will be overwritten. |
| dump_tag       | Setting the name of the dump data directory | No.              | Example parameter: dump_tag="dialog_0". The default dump data directory is named ait_dump.                                                                                   |
| backend        | Inference back end                          | No.              | Data type: str. The options are \[pt and acl\]. pt indicates pytorch-npu or pytorch-gpu inference. acl indicates acceleration library inference.                             |

 *  **register_hook(model, op_list=None, dump_start_token_id=0, dump_end_token_id=-1) registers hooks for models to obtain output data in the middle of the model. This function is required only for PyTorch-npu (GPU) inference.**

| Parameter name      | Meaning                              | Mandatory or Not | Instructions for use                                                                                                                                                                                                                               |
| ------------------- | ------------------------------------ | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model               | Model to hook                        | Yes              | Data type: torch.nn.Module                                                                                                                                                                                                                         |
| op_list             | Operator type that requires hooking. | No.              | Data type: list. The default value is None, indicating that all ops in the model are hooked. The element is`torch.nn.Module`Subcategories of, such as`op_list=[torch.nn.Linear]`If op_list is set, only the specified op is hooked.                |
| dump_start_token_id | Start token ID of dump data.         | No.              | Data type: int. The default value is 0. You can also specify a value greater than 0 to indicate the start token ID. If the encoder is not invoked on the acceleration library side, that is, no encoder dump data exists, set this parameter to 1. |
| dump_end_token_id   | End token ID of dump data.           | No.              | Data type: int. The default value is - 1, indicating that the end token is not limited. You can also specify a value greater than 0 to indicate the end token ID. For example, if the value is 3, dump the token.`[0, 1, 2]`Data of three tokens   |

## Introduction to the CLI ##

```
msit debug compare aclcmp --golden-path {PTA 侧 dump 数据} --my-path {加速库侧 dump 数据}
```

 *  `golden-path`Specifies the dump data path on the PTA.
 *  `my-path`Specifies the dump data path in the acceleration library.

--------------------

# ChatGLM-6B Usage Example #

 *  Install the msit tool before using it. For details, see.[Installing the msit tool](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md)	,`ChatGLM-6B`The model definition is located in the acceleration library.`pytorch/examples/chatglm6b/modeling_chatglm_model_xxxyy.py`xxxyy indicates the chip type used by the user.
 *  Obtaining Model Weight and Configuration File[Huggingface THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)	Save the file to a place other than the acceleration library path. Otherwise, the subsequent compilation will be affected.
 *  in which`ascend-transformer-boost`Need to be specified at compile time as`debug`The mode

## PTA dump data ##

 *  In the`main.py`Adding a Model After Creating a Model`register_hook`, and`set_dump_path`Configure the dump path to save the data in the forward invoking. The weight or bias used at each layer is used as the MD5 value to match the data in the acceleration library.
    
    ```
    import torch
    from msquickcmp.pta_acl_cmp.pt_dump.hook import register_hook, set_dump_path
    
    #Ensure that the model weight and configuration file are secure and reliable. The following code can be used as long as it is safe. Otherwise, set'trust_remote_code' to False
    model = AutoModel.from_pretrained("./", trust_remote_code=True).half().npu()
    
    #Add After Model Initialization
    #Set dump_start_token_id based on the actual acceleration library code. In the current version, the encoder is not executed on the acceleration library. Therefore, set dump_start_token_id to 1.
    register_hook(model, dump_start_token_id=1)
    set_dump_path(dump_path=".", dump_tag="ait_dump", backend="pt")
    ```
 *  Execute the inference script.`bash run.sh patches/models/modeling_chatglm_model_xxxyy.py`. Enter the same input as the data dump in the acceleration library, and check whether the generated data is stored in the`{dump_path}/{dump_tag}/{进程 ID}`In the following figure, xxxyy indicates the chip type used by the user.

## Dump data in the acceleration database ##

 *  In the`main.py`Setting in`set_dump_path`Designated`backend="acl"`, and also specify`LD_PRELOAD`For msit's`libsavetensor.so`Overwrite the original acceleration library`SaveTensor`This interface is used to save the intensor as the MD5 value for matching data on the PTA.
    
    ```
    from msquickcmp.pta_acl_cmp.pt_dump.hook import set_dump_path
    set_dump_path(backend="acl")
    ```
 *  Configure other`ATB`Dump related environment variables and run the inference script.
    
    ```
    MSQUICKCMP_PATH=`python3 -c 'import msquickcmp; print(msquickcmp.__path__[0])'`
    export LD_PRELOAD=$MSQUICKCMP_PATH/libsavetensor.so:$LD_PRELOAD
    
    export ATB_SAVE_TENSOR=1  #Enables the acceleration library dump function. The default value is 0.
    export ATB_SAVE_TENSOR_RANGE=0,1000  #Specifies the maximum number of tokens in data dump in the acceleration library. The default value is 0. The value 1 indicates that only the first token is dumped.
    bash run.sh patches/models/modeling_chatglm_model_xxxyy.py  #xxxyy indicates the type of the chip used by the user.
    ```
 *  Generate data in`atb_temp/tensors/{进程 ID}_{线程ID}`Lower
    
    ```
    ls atb_temp/tensors/ -1t
    #25518 _ 25518
    ls atb_temp/tensors/25518_25518/
    #0 1 2 3 4 5 6 7 8
    ```
 *  If an error occurs`undefined symbol: EVP_md5`, which may be used by Python in the environment.`libssl.so`Compiled with`libsavetensor.so`The system used when`libssl.so`Inconsistent. You can try specifying`export LD_PRELOAD=libssl.so:$LD_PRELOAD`resolves

## msIT Precision Comparison Based on Weight Mapping ##

 *  Assigned separately`--golden-path`is the dump data path on the PTA,`--my-path`To accelerate the dump data path in the database, the mapping is automatically established based on the MD5 value matching relationship of the weight and the comparison result is output.`cmp_report.csv`File
    
    ```
    msit debug compare aclcmp --golden-path ait_dump/25115/ --my-path atb_temp/tensors/25518_25518
    ```
    
    ![chatglm6b_cmp_result.png](chatglm6b_cmp_result.png)	
    
     *  Output result`token_id`The value starts from 0 because the PTA side specifies the`dump_start_token_id=1`,`goden_data_path` `token_id==0`The corresponding path is`1`,`acl_data_path`Corresponding to`0`
     *  In the comparison result of this example, the acceleration library performs the fusion operation on the first Linear + activation at the FFn layer. As a result, the linear operator nodes on the matched PTA are similar.
 *  **Calculate the matching degree between operators. In the comparison result, only operators with the same MD5 weight can be matched. In addition, a small number of nodes may be matched due to the conversion of weight data formats in the actual calculation. Therefore, the matching degree is used only as the general scope of precision exception problems.**
    
    ```
    #Count the number of operations in the acceleration library based on dump data.
    find ./atb_temp/tensors/25518_25518/1 -wholename '*Operation/*Operation/after' | wc -l
    #644
    ```
    
    In addition, in the example, the number of operations matching weight MD5 for a single token in the CSV table is calculated as follows:`170`, Proportion`26.4%`
    
    ![matched_pie.png](matched_pie.png)	

--------------------

# BLOOM-7B Usage Example #

 *  Install the msit tool before using it. For details, see.[Installing the msit tool](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md)	,`BLOOM-7B`Model definition in the acceleration library`pytorch/examples/bloom7b`
 *  Obtaining Model Weight and Configuration File[Huggingface bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)	Save the file to a place other than the acceleration library path. Otherwise, the subsequent compilation will be affected.
 *  **In this example, the volume of dump data on the acceleration database side is large. You can run the following command:`register_hook`of the`dump_end_token_id`The parameter limits the number of dump tokens on the PTA side.`ATB_SAVE_TENSOR_RANGE`Limit the number of dump tokens in the acceleration library.**

## Dump data on the PTA. ##

 *  In the`run_bloom_npu.py`of the`main`In the function, add after the model is created.`register_hook`, and`set_dump_path`Configure the dump path to save the data in the forward invoking.
    
    ```
    from msquickcmp.pta_acl_cmp.pt_dump.hook import register_hook, set_dump_path
    ...
    model, tokenizer = load_model(args)
    
    #This parameter is added after the model is initialized. In the current version, the encoder is executed on the sample acceleration library, and dump_start_token_id does not need to be specified.
    #If dump_end_token_id is set to 5, only the data of the token [0, 1, 2, 3, 4] is dumped.
    register_hook(model, dump_end_token_id=5)
    set_dump_path(dump_path=".", dump_tag="ait_dump", backend="pt")
    ...
    
    #Modify the script to perform only one inference.
    #- seq_lens = [2**x for x in range(5, 11)]
    #- max_new_tokens_list = [2**x for x in range(5, 11)]
    seq_lens = [2**x for x in range(5, 6)]
    max_new_tokens_list = [2**x for x in range(5, 6)]  #The output is 32 tokens.
    ```
    
    Execute the inference script.`bash run.sh -p modeling_bloom.py --run --device 0`to view the generated data in the`{dump_path}/{dump_tag}/{进程 ID}`Lower

## Dump data in the acceleration database ##

 *  In the`run_bloom_npu.py`Setting in`set_dump_path`Designated`backend="acl"`, and also specify`LD_PRELOAD`For msit's`libsavetensor.so`Overwrite the original acceleration library`SaveTensor`This interface is used to save the tensor as the MD5 value for matching data on the PTA.
    
    ```
    from msquickcmp.pta_acl_cmp.pt_dump.hook import set_dump_path
    set_dump_path(backend="acl")
    ```
    
    Configure other`ATB`Dump related environment variables, execute the inference script, and specify`ATB_SAVE_TENSOR_RANGE=0,5`Only dump is allowed.`[0, 1, 2, 3, 4]`Token data
    
    ```
    MSQUICKCMP_PATH=`python3 -c 'import msquickcmp; print(msquickcmp.__path__[0])'`
    export LD_PRELOAD=$MSQUICKCMP_PATH/libsavetensor.so:$LD_PRELOAD
    
    export ATB_SAVE_TENSOR=1  #Enables the acceleration library dump function. The default value is 0.
    export ATB_SAVE_TENSOR_RANGE=0,5  #Only the token data before the dump is `[0, 1, 2, 3, 4]`. The default value is 0. The value 1 indicates that only the first token is dumped.
    bash run.sh -p patches/models/modeling_bloom_model_performance.py --run --device 0
    ```
    
    Generate data in`$ASDOPS_LOG_TO_FILE_DIR/tensors/{进程 ID}_{线程ID}`Lower, where`$ASDOPS_LOG_TO_FILE_DIR`Set this parameter when the acceleration library is configured. The default value is.`"atb_temp"`
    
    ```
    ls $ASDOPS_LOG_TO_FILE_DIR/tensors/ -1t
    #18621 _ 18621
    ls $ASDOPS_LOG_TO_FILE_DIR/tensors/18621_18621/
    #0 1 2 3 4
    ```
    
    If an error occurs`undefined symbol: EVP_md5`, which may be used by Python in the environment.`libssl.so`Compiled with`libsavetensor.so`The system used when`libssl.so`Inconsistent. You can try specifying`export LD_PRELOAD=libssl.so:$LD_PRELOAD`resolves

## msIT Precision Comparison Based on Weight Mapping ##

 *  Assigned separately`--golden-path`is the dump data path on the PTA,`--my-path`To accelerate the dump data path in the database, the mapping is automatically established based on the MD5 value matching relationship of the weight, and the comparison result is output.`cmp_report.csv`File
    
    ```
    msit debug compare aclcmp --golden-path ait_dump/21219 --my-path atb_temp/tensors/18621_18621
    ```
    
    ![bloom7b_cmp_result.png](bloom7b_cmp_result.png)	
 *  **Calculate the matching degree between operators. Only operators with the same MD5 weight can be matched in the comparison result. In addition, a small number of nodes may be matched because the weight data format is converted in the actual calculation. Therefore, the matching degree is used only as the scope of precision exception problems.**
    
    ```
    #Count the number of operations in the acceleration library based on dump data.
    find ./atb_temp/tensors/18621_18621/1 -wholename '*Operation/*Operation/after' | grep -v 'GraphOperation/after' | wc -l
    #690
    ```
    
    In this example, the number of operations matching the weight MD5 of a single token in the CSV table is calculated as follows:`182`, Proportion`26.4%`

