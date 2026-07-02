# Instructions for Inserting dump_data Codes #

## How to Use ##

The usage method consists of two steps: dump data and compare data. The details are as follows:

 *  The tool provides the dump_data function for data dump, which needs to be inserted into the model script.
 *  After data dump is complete, two JSON files are generated in the specified directory. Use the corresponding JSON paths as input parameters to the tool for comparison.

The tool command is as follows:

`msit debug compare aclcmp --golden-path path_to_golden_data.json --my-path path_to_acl_data.json --output output_dir`

### Dump data ###

#### Function prototype ####

```
dump_data(data_src, data_id, data_val, tensor_path, token_id)
```

#### Function Description ####

 *  The function is used to flush the data to be dumped to the disk (high-level flushing path:`当前目录/{PID}_cmp_dump_data/{data_src}_tensor/{token_id}/xxx.npy`)
 *  At the same time, data information is written to a`metadata.json`Medium, used for subsequent comparison (path:`当前目录/{PID}_cmp_dump_data/{data_src}_tensor/metadata.json`)

#### Parameter Description ####

| Parameter name | Description                                                                                                                                                                                                                                                            | Mandatory or not |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| data_src       | Indicates whether the dump data is benchmark data (CPU/GPU/NPU) or data to be compared in the acceleration database. The value can be`"golden"`or the`"acl"`,`"golden"`represents benchmark data,`"acl"`Indicates the data to be compared in the acceleration library. | Yes              |
| data_id        | Unique ID of data, which is used to match data_id (corresponding relationship) of data on the other side to implement data comparison.                                                                                                                                 | Yes              |
| data_val       | In the high-level scenario, the variable name can be directly transferred and the tensor is obtained and then flushed to disks. This parameter is mandatory in the high-level scenario.                                                                                | No.              |
| tensor_path    | In low-level scenarios, variable names cannot be obtained directly. Set the variable names based on the dump directory structure path of the acceleration library. This parameter is mandatory in low-level scenarios.                                                 | No.              |
| token_id       | In the high-level scenario, a directory with token_id is generated and dump data is generated in the directory. In the low-level scenario, data with a specified token_id is generated.                                                                                | Yes              |

**Precautions**

 *  data_id is the unique data matching ID.
 *  Reference format for setting tensor_path directories`{model_index}_{model_name}/{layer_index}_{layer_name}/{op_index}_{op_name}/after/Output0.bin`, such as`"0_ChatGlm6BModelEncoderTorch/0_ChatGlm6BLayerEncoderOperationGraphRunner/3_SelfAttentionOpsChatglm6bRunner/after/outTensor0.bin"`
 *  token_id is used only for data differentiation, not for data identification. (In the low-level scenario, only the acceleration library data of the specified token round is dumped.). The value needs to be set by the user. It is recommended that the value can be directly increased in the single Encoder/Decoder structure. In the Encoder-Decoder structure, the value can be differentiated or the value can be processed based on the first word inference.

#### Use Example ####

##### 1. Add model code. #####

 *  Import the corresponding function at the beginning of the .py file.

```
from msquickcmp.pta_acl_cmp.compare import dump_data
```

##### Insert dump_data code in the position of the data to be compared. #####

 *  The dump_data has high-level and low-level scenarios.

###### High-level Insert Code ######

```
#Where you need to compare the data
#=============================golden================================
#Indicates the global auto-increment data_id and token rounds, respectively.
global data_id, token_id
for i, layer in enumerate(self.layers):

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    layer_ret = layer(
        hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
        layer_id=torch.tensor(i),
        layer_past=past_key_values[i],
        use_cache=use_cache,
        output_attentions=output_attentions
    )

    hidden_states = layer_ret[0]

    if use_cache:
        presents = presents + (layer_ret[1],)

    if output_attentions:
        all_self_attentions = all_self_attentions + \
            (layer_ret[2 if use_cache else 1],)
#Compare the last output.
dump_data("golden", data_id, hidden_states, token_id=token_id)
data_id += 1
#=============================golden================================



#=============================acl================================
#Same meaning as above
global data_id, token_id
acl_model_out = self.acl_encoder_operation.execute(
    hidden_states, position_ids, self.cos, self.sin, attention_mask, seq_len)
#Compare the last encoder output.
dump_data("acl", data_id, acl_model_out[0], token_id=token_id)
data_id += 1
#=============================acl================================
```

###### Low-level insertion code ######

```
#Where you need to compare the data
#=============================golden================================
#To set low-level on the PT side, you only need to add dump_data to the declaration of the corresponding function.
def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids,
            attention_mask: torch.Tensor,
            layer_id,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
    #Indicates the global auto-increment data_id and token rounds, respectively.
    global data_id, token_id
    """
    Other codes
    """
    #Residual connection.
    alpha = (2 * self.num_layers) ** 0.5
    hidden_states = attention_input * alpha + attention_output

    mlp_input = self.post_attention_layernorm(hidden_states)
    dump_data("golden", data_id, mlp_input, token_id=token_id)
    data_id += 1
#=============================golden================================



#=============================acl================================
#The low-level acceleration library needs to invoke dump_data before invoking the execute function to inform the acceleration library of the data to be dumped.
#Indicates the global auto-increment data_id and token rounds, respectively.
global data_id, token_id
if past_key_values[0] is None:
    for i in range(num_layers):
        tensor_path = "0_ChatGlm6BModelEncoderTorch/{i}_ChatGlm6BLayerEncoderOperationGraphRunner/6_NormOpsRunner/after/outTensor0.bin"
        dump_data("acl", data_id, tensor_path=tensor_path, token_id=token_id)
        data_id += 1
    acl_model_out = self.acl_encoder_operation.execute(
        hidden_states, position_ids, self.cos, self.sin, attention_mask, seq_len)
#=============================acl================================
```

##### 2. Model inference #####

###### Preparing the running environment for the acceleration library ######

```
source /usr/local/Ascend/ascend-toolkit/set_env
cd output/acltransformer #Acceleration Library Directory Location
source set_env.sh

#Used when low-level data in the acceleration library needs to be dumped.
site_packages_path=$(python3 -c "import site; print(site.getsitepackages()[0])")
export LD_PRELOAD="${site_packages_path}/msquickcmp/libsavetensor.so":$LD_PRELOAD
export ATB_SAVE_TENSOR=1 #Enable the dump function of the acceleration library. The value 0 indicates that the dump function is disabled (default value). The value 1 indicates that the dump function is enabled.
export ATB_SAVE_TENSOR_START=0 #Number of data rounds when the acceleration library interface dump starts;
export ATB_SAVE_TENSOR_END=1 #Number of data rounds after the acceleration library interface dump is complete.
export ATB_SAVE_TENSOR_RUNNER=NormOpsRunner #Runner trustlist of the acceleration library dump data. The default value is empty. If the value is empty, full dump is performed. If the value is not empty, only the runner data contained in the trustlist is dumped.
```

###### Model Inference Execution ######

Run the original running mode.

 *  Example:

```
bash run.sh patches/model/modeling_chatglm_model.py
```

##### 3. Data comparison #####

After the execution is complete, data is flushed to disks and a metadata.json file is generated in the following path:`当前目录/{PID}_cmp_dump_data/{data_src}_tensor/metadata.json`), transfer the metadata.json paths on both sides to the input parameter of the tool and specify the output path.`output_dir`Run the following command to complete the comparison:`msit debug compare aclcmp --golden-path path_to_golden_data.json --my-path path_to_acl_data.json --output output_dir`

After the comparison is complete,`output_dir`A new one will be generated next.`cmp_report.csv`to save the final comparison result.

 *  Comparison result:![cmp_report.csv](./cmp_report.png)	

