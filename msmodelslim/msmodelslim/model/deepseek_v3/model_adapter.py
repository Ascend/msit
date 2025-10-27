# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os.path
from importlib import import_module
from typing import Any, List, Optional, Tuple, Generator, Dict

import torch
import torch.nn as nn
from torch import distributed as dist

from msmodelslim.app import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func, \
    TransformersForwardBreak
from msmodelslim.model.factory import ModelFactory
from msmodelslim.model.transformers import TransformersModel
from msmodelslim.quant import ir as qir
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.security.model import SafeGenerator
from msmodelslim.utils.security import json_safe_load, json_safe_dump
from .convert_fp8_to_bf16 import auto_convert_model_fp8_to_bf16
from .mtp_quant_module import warp_mtp_model, remove_zero_and_shift
from ..interface_hub import ModelInfoInterface, ModelSlimPipelineInterfaceV1, IterSmoothInterface, \
    FlexSmoothQuantInterface, FA3QuantAdapterInterface, FA3QuantPlaceHolder, QuaRotInterface, AscendV1SaveInterface
from .quarot import get_ln_fuse_map, get_rotate_map


@ModelFactory.register("DeepSeek-V3")
@ModelFactory.register("DeepSeek-V3-0324")
@ModelFactory.register("DeepSeek-R1")
@ModelFactory.register("DeepSeek-R1-0528")
@ModelFactory.register("DeepSeek-V3.1")
@logger_setter("msmodelslim.model.deepseek_v3")
class DeepSeekV3ModelAdapter(TransformersModel,
                             ModelInfoInterface,  # support naive quantization
                             ModelSlimPipelineInterfaceV1,  # support modelslim v1
                             IterSmoothInterface,  # support iter smooth
                             FlexSmoothQuantInterface,  # support flex smooth quant
                             FA3QuantAdapterInterface,  # support FA3 activation quant placeholders
                             QuaRotInterface,
                             AscendV1SaveInterface,
                             ):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'deepseek_v3'

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        # load one more layer for mtp
        self.config.num_hidden_layers += 1
        # load model to cpu and cpu
        model = SafeGenerator.get_model_from_pretrained(model_path=str(self.model_path),
                                                        config=self.config,
                                                        trust_remote_code=self.trust_remote_code,
                                                        device_map="cpu",
                                                        torch_dtype="auto",
                                                        attn_implementation='eager')
        # auto convert fp8 to bf16
        auto_convert_model_fp8_to_bf16(model, str(self.model_path))
        # warp mtp into model
        model = warp_mtp_model(self.config, model, str(self.model_path))
        return model

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(model)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:
        def pack_request(name: str, request_args: Tuple, request_kwargs: Dict):
            module = model.get_submodule(name)
            return ProcessRequest(name, module, request_args, request_kwargs)

        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "decoderlayer" in module.__class__.__name__.lower()
        ]

        # 存储第一个transformer block的输入
        first_block_input: Optional[Tuple] = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs,)
            raise TransformersForwardBreak()

        hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)]

        # 执行一次前向传播以获取输入
        try:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                model(*inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        except Exception as e:
            raise e
        finally:
            for hook in hooks:
                hook.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

        # 循环处理每个transformer block
        current_inputs = first_block_input

        if dist.is_initialized():
            dist.barrier()

        mtp_idx = self.config.num_hidden_layers

        args, kwargs = current_inputs
        for name, block in transformer_blocks[:mtp_idx]:
            output = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = output[0]
            args = (hidden_states,)

        def wrap_device(module: nn.Module):
            def auto_module(arg):
                module.to('npu')
                result = module(arg.to('npu'))
                module.to('cpu')
                return result

            return auto_module

        hidden_states = model.model.norm(*args)
        logits = wrap_device(model.lm_head)(hidden_states)
        logits = logits.float()

        ####################### MTP LAYER ######################
        input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs[0]
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

        mtp_layer = model.model.layers[self.config.num_hidden_layers]
        input_embeds_mtp = wrap_device(mtp_layer.embed_tokens)(input_ids_mtp)
        input_embeds_mtp = wrap_device(mtp_layer.enorm)(input_embeds_mtp)
        hidden_states_mtp = wrap_device(mtp_layer.hnorm)(hidden_states)
        hidden_states_mtp = torch.cat([input_embeds_mtp, hidden_states_mtp], dim=-1)
        hidden_states_mtp = wrap_device(mtp_layer.eh_proj)(hidden_states_mtp)

        attention_mask = inputs['attention_mask'] if isinstance(inputs, dict) else inputs[1]

        # transformers==4.48.2
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        attention_mask_mtp = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_ids.shape[:2]),
            input_embeds_mtp,
            0,
        )

        del input_embeds_mtp
        del attention_mask
        del input_ids_mtp
        del input_ids
        del logits
        del hidden_states
        hidden_states_mtp = hidden_states_mtp.detach()
        attention_mask_mtp = attention_mask_mtp.detach()

        _ = yield pack_request(f"model.layers.{mtp_idx}",
                               (hidden_states_mtp,),
                               {
                                   "attention_mask": attention_mask_mtp,
                                   "position_ids": position_ids,
                                   "past_key_value": None,
                                   "output_attentions": False,
                                   "use_cache": False,
                               })

        # the remaining part just ignore
        return

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        def pre_forward_hook(module, args, kwargs):
            kwargs['use_cache'] = need_kv_cache
            return args, kwargs

        model.model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            # OKV_b融合的映射配置：o_proj -> kv_b_proj
            okv_b_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.kv_b_proj",  # KV_b投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]  # 输出投影层
            )

            # Norm-Linear融合的映射配置1：q_a_proj, kv_a_proj_with_mqa -> input_layernorm
            norm_linear_mapping_config1 = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[f"model.layers.{layer_idx}.self_attn.q_a_proj",
                         f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"]  # 注意力层的Q_a,KV_a投影
            )

            # Norm-Linear融合的映射配置2：q_b_proj -> q_a_layernorm
            norm_linear_mapping_config2 = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.q_a_layernorm",  # q_a_layernorm
                targets=[f"model.layers.{layer_idx}.self_attn.q_b_proj"]  # q_b投影
            )

            # 为当前layer添加4个配置
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=okv_b_mapping_config,
                    fusion=FusionConfig(
                        fusion_type="kv",
                        num_attention_heads=self.config.num_attention_heads,
                        num_key_value_heads=self.config.num_key_value_heads,
                        custom_config={
                            'qk_nope_head_dim': self.config.qk_nope_head_dim,
                            'v_head_dim': self.config.v_head_dim,
                        }
                    ),
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config1
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config2
                ),
            ])

        return adapter_config

    # ===== FA3QuantAdapterInterface =====
    def inject_fa3_placeholders(self, root_name: str, root_module: nn.Module, should_inject) -> None:
        """为 DeepSeekV3 注意力模块安装 FA3 占位，并包裹 forward 调用这些占位。

        - 在每个 Attention 模块下注入子模块：fa3_q, fa3_k, fa3_v
        - 包裹其 forward，在计算 q_nope 与 compressed_kv 后，依次调用占位：
            q_nope = self.fa3_q(q_nope)
            compressed_kv = self.fa3_k(compressed_kv.unsqueeze(1)).squeeze(1)
            _ = self.fa3_v(compressed_kv.unsqueeze(1)).squeeze(1)
        """

        def _wrap_attention_forward(attn_mod: nn.Module):
            # 动态导入以获取 apply_rotary_pos_emb（采用 import_module，更清晰稳健）
            deepseek_module = import_module(attn_mod.forward.__module__)
            apply_rotary_pos_emb = deepseek_module.apply_rotary_pos_emb

            def new_forward(
                    self,
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_value: Optional[Any] = None,
                    output_attentions: bool = False,
                    use_cache: bool = False,
                    **kwargs,
            ):
                # 参考默认适配器 deepseek mla 前向的关键路径（保留原计算流，仅插入占位调用）
                bsz, q_len, _ = hidden_states.size()

                if getattr(self, 'q_lora_rank', None) is None:
                    q = self.q_proj(hidden_states)
                else:
                    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
                q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
                q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

                compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
                compressed_kv, k_pe = torch.split(
                    compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
                compressed_kv = self.kv_a_layernorm(compressed_kv)
                k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
                kv_seq_len = k_pe.shape[-2]

                if past_key_value is not None:
                    if getattr(self, 'layer_idx', None) is None:
                        raise ValueError(
                            f"The cache structure has changed since version v4.36. "
                            f"If you are using {self.__class__.__name__} "
                            f"for auto-regressive decoding with k/v caching, "
                            f"please make sure to initialize the attention class "
                            "with a layer index."
                        )
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

                cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
                q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos}
                    compressed_kv = compressed_kv.unsqueeze(1)
                    k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
                    compressed_kv = compressed_kv.squeeze(1)

                kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)

                q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]
                out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]

                q_nope = torch.matmul(q_nope, q_absorb)

                # ===== 插入 FA3 占位 =====
                if hasattr(self, 'fa_q'):
                    q_nope = self.fa_q(q_nope)
                if hasattr(self, 'fa_k'):
                    compressed_kv = self.fa_k(compressed_kv.unsqueeze(1)).squeeze(1)
                if hasattr(self, 'fa_v'):
                    _ = self.fa_v(compressed_kv.unsqueeze(1)).squeeze(1)
                # ========================

                attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT))
                attn_weights = attn_weights * self.softmax_scale

                if attention_mask is None:
                    raise ValueError("Attention mask cannot be None")
                attn_weights = attn_weights + attention_mask

                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
                attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout,
                                                           training=self.training)
                attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
                attn_output = torch.matmul(attn_output, out_absorb.mT)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
                attn_output = self.o_proj(attn_output)
                if not output_attentions:
                    attn_weights_out = None
                else:
                    attn_weights_out = attn_weights
                return attn_output, attn_weights_out, past_key_value

            attn_mod.forward = new_forward.__get__(attn_mod, attn_mod.__class__)

        # 遍历并注入
        for name, module in root_module.named_modules():
            if 'Attention' in module.__class__.__name__ and should_inject(f"{root_name}.{name}" if root_name else name):
                # 为该注意力模块注入占位
                root_module.set_submodule(f"{name}.fa_q", FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f"{name}.fa_k", FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f"{name}.fa_v", FA3QuantPlaceHolder(ratio=1.0))
                _wrap_attention_forward(module)

    def get_ln_fuse_map(self):
        return {}, get_ln_fuse_map(self.config)
    
    def get_bake_names(self):
        return [], []
    
    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _ = get_rotate_map(self.config, block_size)
        return [pre_run], [pair for pair in rot_pairs.values()]

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """
        根据 MideIE 要求在deepseek w4a8和w4a8c8场景下，config.json 中添加以下字段
        - mtp_quantize: w8a8_dynamic
        - quantize: w8a8_dynamic
        - moe_quantize: w4a8_dynamic
        - mla_quantize: w8a8(w4a8c8场景下使用w8a8) or w8a8_dynamic(w4a8场景下使用w8a8_dynamic)

        Args:
            model: 模型
            save_directory: 导出件的保存目录
        """
        use_w4a8 = False
        use_c8 = False
        
        for _, module in model.named_modules():
            if isinstance(module, qir.W4A8DynamicFakeQuantLinear):
                use_w4a8 = True
            if isinstance(module, qir.FakeQuantActivationPerHead):
                use_c8 = True
            if use_w4a8 and use_c8:
                break

        if use_w4a8:
            config_file = os.path.join(save_directory, "config.json")
            config_data = json_safe_load(config_file, check_user_stat=False)

            config_data["mtp_quantize"] = "w8a8_dynamic"
            config_data["quantize"] = "w8a8_dynamic"
            config_data["moe_quantize"] = "w4a8_dynamic"
            if use_c8:
                config_data["mla_quantize"] = "w8a8"
            else:
                config_data["mla_quantize"] = "w8a8_dynamic"
                
            json_safe_dump(config_data, config_file, indent=2, check_user_stat=False)

        return
        