# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from ascend_utils.common.security.path import get_valid_read_path

READ_ONLY_PERMISSION = 0o400
READ_WRITE_PERMISSION = 0o600
MAX_READ_FILE_SIZE_16G = 17179869184  # 16G, 16 * 1024 * 1024 * 1024


def remove_zero_and_shift(matrix):
    n, m = matrix.shape

    # Step 1: 找到每行第一个 0 的位置（即要删除的位置）
    # 如果某行没有 0，则默认保留所有元素（但根据题意，应该每行都有一个 0）
    zero_pos = (matrix == 0).int().argmax(dim=1)  # [n,]

    # Step 2: 构造掩码，标记要保留的元素（排除每行的第一个 0）
    # 生成一个 [n, m] 的坐标矩阵，标记每列是否等于 zero_pos
    col_indices = torch.arange(m, device=matrix.device).expand(n, -1)  # [n, m]
    mask = (col_indices != zero_pos.unsqueeze(1))  # [n, m]

    # Step 3: 用掩码筛选元素（自动展平，需要重新调整形状）
    filtered = matrix[mask].view(n, m - 1)  # [n, m-1]

    # Step 4: 在最后一列补 0
    result = torch.cat([filtered, torch.zeros(n, 1, device=matrix.device)], dim=1)  # [n, m]

    return result.to(matrix)


class SharedHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        normalized_states = self.norm(hidden_states)
        logits = self.head(normalized_states)
        return logits


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MTPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hnorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.shared_head = SharedHead(config)

        self.eh_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False
        )
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )


class MTPModel(PreTrainedModel):
    def __init__(self, config, model, mtp_layer: nn.Module):
        super().__init__(config)
        self.model = model.model
        self.vocab_size = config.vocab_size
        self.lm_head = model.lm_head
        self.post_init()

        mtp_decoder = model.model.layers[config.num_hidden_layers - 1]

        mtp_decoder.enorm = mtp_layer.enorm
        mtp_decoder.hnorm = mtp_layer.hnorm
        mtp_decoder.shared_head = mtp_layer.shared_head
        mtp_decoder.eh_proj = mtp_layer.eh_proj
        mtp_decoder.embed_tokens = mtp_layer.embed_tokens

        self.model.config.num_hidden_layers -= 1
        self.config.num_hidden_layers -= 1

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        old_layers = self.model.layers
        try:
            self.model.layers = old_layers[:self.config.num_hidden_layers]
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        finally:
            self.model.layers = old_layers

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        ####################### MTP LAYER ######################
        input_ids_mtp = remove_zero_and_shift(input_ids)
        position_ids = torch.arange(
            0,
            input_ids_mtp.shape[-1],
            dtype=torch.long,
            device=input_ids.device,
        ) + 1
        position_ids = position_ids.unsqueeze(0)
        logits[:, -1, :].argmax(dim=1)
        input_ids_mtp[:, -1] = logits[:, -1, :].argmax(dim=1)

        mtp_layer = old_layers[self.config.num_hidden_layers]
        input_embeds_mtp = mtp_layer.embed_tokens(input_ids_mtp)
        input_embeds_mtp = mtp_layer.enorm(input_embeds_mtp)
        hidden_states_mtp = mtp_layer.hnorm(hidden_states)
        hidden_states_mtp = torch.cat([input_embeds_mtp, hidden_states_mtp], dim=-1)
        hidden_states_mtp = mtp_layer.eh_proj(hidden_states_mtp)

        # transformers==4.48.2
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        attention_mask_mtp = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_ids.shape[:2]),
            input_embeds_mtp,
            0,
        )
        layer_outputs_mtp = mtp_layer(
            hidden_states_mtp,
            attention_mask=attention_mask_mtp,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        _ = mtp_layer.shared_head(layer_outputs_mtp[0])
        ####################### MTP LAYER ######################

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def warp_mtp_model(config, base_model, model_path):
    """
    从预训练模型加载MTP层权重，并封装为MTP_Model
    
    Args:
        config (Config): 模型配置对象
        base_model (PreTrainedModel): 原始基础模型
        model_path (str): 预训练模型路径

    Returns:
        MTP_Model: 封装后的MTP模型
    """
    mtp_layer = MTPLayer(config)
    mtp_safetensor = os.path.join(model_path, "model-00163-of-000163.safetensors")
    mtp_safetensor = get_valid_read_path(
        mtp_safetensor,
        size_max=MAX_READ_FILE_SIZE_16G,
        is_dir=False,
        check_user_stat=True
    )
    mtp_weight = load_file(mtp_safetensor, device="cpu")
    new_state_dict = {}
    for key, value in mtp_weight.items():
        new_key = key.replace('model.layers.61.', '')
        if new_key in mtp_layer.state_dict().keys():
            new_state_dict[new_key] = value
    mtp_layer.load_state_dict(new_state_dict)
    warpped_model = MTPModel(config, base_model, mtp_layer)
    return warpped_model
