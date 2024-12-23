import math
import numpy as np
import torch
import torch_npu
from typing import Optional, Tuple, Union
from transformers import Qwen2ForCausalLM, LlamaForCausalLM, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaAttention
import types

def patch_with_omni_attn_pattern(
    model: Union[LlamaForCausalLM, Qwen2ForCausalLM],  # these two model series are well tested
    pattern: np.ndarray,
    sink: int = 128,
    recent: int = 256,
):
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        device = next(module.parameters()).device
        module.forward = types.MethodType(omni_attention_forward, module)
        module.sink = sink
        module.recent = recent

        layer_pattern = torch.tensor(pattern[idx], device=device, dtype=int).repeat_interleave(module.num_key_value_groups)
        module.register_buffer("stream_head_mask", layer_pattern < 0.5)


def omni_attention_forward(
    self: Union[LlamaAttention, Qwen2Attention],
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[DynamicCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    position_ids = position_ids.to(value_states.device)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=2,
    )

    if past_key_value is not None:
        # (B, S, N, D) -> (B, N, S, D)
        key_states, value_states = key_states.transpose(1, 2), value_states.transpose(1, 2)
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # (B, N, S, D) -> (B, S, N, D)
        key_states, value_states = key_states.transpose(1, 2), value_states.transpose(1, 2)
    kv_seq_len = key_states.shape[1]

    if q_len == kv_seq_len:
        # pre-filling: use flash attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
        )
    else:
        # decoding
        omni_mask = torch.zeros(bsz,
                                query_states.shape[2],
                                q_len,
                                kv_seq_len,
                                dtype=bool,
                                device=query_states.device)
        if kv_seq_len > self.sink + self.recent:
            omni_mask[:, self.stream_head_mask, :, self.sink : kv_seq_len-self.recent] = True
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
            atten_mask=omni_mask,
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def flash_attn_func(
    q,
    k,
    v,
    causal=True,
    dropout_p=0.0,
    softmax_scale=None,
    atten_mask=None,
):
    """dropout_p should be set to 0.0 during evaluation.
    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to applsy causal attention mask (e.g., for auto-regressive modeling).
    """
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    device = q.device
    head_num = q.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if atten_mask is None:
        atten_mask = torch.ones(seqlen_q, seqlen_k, dtype=bool, device=device)
        atten_mask = torch.triu(atten_mask, diagonal=seqlen_k-seqlen_q+1)
    output = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num,
        "BSND",
        keep_prob=1.0,
        scale=softmax_scale,
        atten_mask=atten_mask,
    )[0]
    return output