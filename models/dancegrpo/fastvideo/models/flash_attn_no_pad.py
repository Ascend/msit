from einops import rearrange


def flash_attn_no_pad(qkv,
                      key_padding_mask,
                      causal=False,
                      dropout_p=0.0,
                      softmax_scale=None):
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    # 去引入的flash_attn功能，因为dancegrpo-flux并无引用，且NPU不支持该模块，当前去掉引入，后续移除文件
    pass
