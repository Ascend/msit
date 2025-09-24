#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from pydantic import field_validator

import msmodelslim.quant.ir as qir
from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.exception import SchemaValidateError
from .quarot_interface import QuaRotAdapter
from .quarot_utils import QuaRotUtils


class QuaRotProcessorConfig(AutoProcessorConfig):
    type: Literal["quarot"] = "quarot"
    online: bool = False
    block_size: int = -1
    down_proj_online_layers: List[int] = []
    max_tp_size: int = 4

    @field_validator('max_tp_size')
    @classmethod
    def validate_max_tp_size(cls, v):
        """校验 max_tp_size：必须大于等于1且为2的幂"""
        if v < 1 or not QuaRotUtils.is_power_of_two(v):
            raise SchemaValidateError(f"max_tp_size must be a positive power of 2 or equal to 1, got {v}")
        return v

    @field_validator('block_size')
    @classmethod
    def validate_block_size(cls, v):
        """校验 block_size：取值范围为-1或大于0且为2的幂的整数"""
        if v == -1:
            return v
        if v <= 0 or not QuaRotUtils.is_power_of_two(v):
            raise SchemaValidateError(f"block_size must be -1 or a positive power of 2, got {v}")
        return v


@QABCRegistry.register(dispatch_key=QuaRotProcessorConfig, abc_class=AutoSessionProcessor)
class QuaRotProcessor(AutoSessionProcessor):

    def __init__(self, model: nn.Module, config: QuaRotProcessorConfig, adapter: QuaRotAdapter, **kwargs) -> None:
        super().__init__(model)
        self.config = config
        self.model = model
        self.adapter = adapter
        self.rot: Optional[torch.Tensor] = None
        self.rot_att_v: Optional[torch.Tensor] = None
        self.rot_online_down_proj: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.rot_online_o_proj: Optional[torch.Tensor] = None

        self.num_attn_heads: int = self.adapter.get_num_attention_heads()
        self.num_kv_heads: int = self.adapter.get_num_key_value_heads()

        self.online_rotation_info: Optional[qir.QuarotOnlineRotationInfo] = None

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True

    def pre_run(self) -> None:
        # 计算旋转矩阵
        model_dim = self.adapter.get_hidden_dim()
        head_dim = self.adapter.get_head_dim()
        device = self.model.device
        self.rot = QuaRotUtils.create_rot(model_dim, block_size=self.config.block_size, device=device)
        self.rot_att_v = QuaRotUtils.create_rot(head_dim, block_size=self.config.block_size, device=device)

        if self.config.online:
            rot1, rot2, self.rot_online_o_proj = self._add_online_rotations(self.model)
            identity = torch.eye(int(self.adapter.get_head_dim())).to(self.rot_online_o_proj)
            self.rot_online_down_proj = (rot1, rot2)
            self.online_rotation_info = qir.QuarotOnlineRotationInfo(self.rot_online_o_proj, identity, rot1, rot2,
                                                                     self.config.max_tp_size)
        # 层融合
        QuaRotUtils.fuse_ln_linear(
            self.model.get_submodule(self.adapter.get_pre_head_layernorm()),
            [self.model.get_submodule(self.adapter.get_lm_head())])
        # 旋转
        QuaRotUtils.rotate_embedding(self.model.get_submodule(self.adapter.get_embedding()), self.rot)
        if hasattr(self.model.config, 'tie_word_embeddings') and not self.model.config.tie_word_embeddings:
            QuaRotUtils.rotate_head(self.model.get_submodule(self.adapter.get_lm_head()), self.rot)

    def preprocess(self, request: BatchProcessRequest) -> None:

        # 获取norm_linear、linear_linear、ov_pair
        norm_linear_pairs = self.adapter.get_layer_wise_norm_liner_pair(request.module)
        ov_pairs = self.adapter.get_layer_wise_ov_pair(request.module)
        up_down_pairs = self.adapter.get_layer_wise_up_down_pair(request.module)

        # 层融合
        for norm_layer, linear_layers in norm_linear_pairs.items():
            QuaRotUtils.fuse_ln_linear(norm_layer, linear_layers)

        # 插入旋转矩阵
        layer_idx = int(request.name.split('.')[-1])

        QuaRotUtils.rotate_attention_mlp_input(norm_linear_pairs, self.rot)
        QuaRotUtils.rotate_attention_ov_output(ov_pairs, self.rot, self.rot_att_v, self.num_kv_heads)
        QuaRotUtils.rotate_mlp_output(up_down_pairs, self.rot)

        online_oproj_rotation = True if self.rot_online_o_proj is not None else False
        QuaRotUtils.rotate_o_proj_input(ov_pairs, self.rot_att_v, self.rot_online_o_proj, online=online_oproj_rotation,
                                        num_attn_heads=self.num_attn_heads)

        if self.rot_online_down_proj and layer_idx in self.config.down_proj_online_layers:
            QuaRotUtils.rotate_down_proj(up_down_pairs, *self.rot_online_down_proj)

        if self.config.online:
            if layer_idx in self.config.down_proj_online_layers:
                for _, down_proj in up_down_pairs.items():
                    # 使用with_kwargs注册Kronecker HookIR实例作为hook（用于down_proj）
                    hook_ir = qir.QuarotKroneckerRotationHookIR(
                        str(layer_idx),
                        self.online_rotation_info
                    )
                    down_proj.register_forward_pre_hook(hook_ir)

            if online_oproj_rotation:
                for o_proj, _ in ov_pairs.items():
                    # 使用with_kwargs注册普通HookIR实例作为hook（用于o_proj）
                    hook_ir = qir.QuarotHeadsRotationHookIR(
                        str(layer_idx),
                        self.online_rotation_info
                    )
                    o_proj.register_forward_pre_hook(hook_ir)

    def post_run(self) -> None:
        pass

    def _add_online_rotations(self, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(model.config, 'intermediate_size'):
            raise ValueError("Model config must contain 'intermediate_size' for online rotation")
        size_1, size_2 = QuaRotUtils.get_decompose_dim(model.config.intermediate_size)
        rot1 = QuaRotUtils.create_rot(size_1, self.config.max_tp_size, device=model.device)
        rot2 = QuaRotUtils.create_rot(size_2, -1, device=model.device)

        num_heads = self.num_attn_heads
        rot_online_o_proj = QuaRotUtils.create_rot(num_heads, self.config.max_tp_size, device=model.device)

        return rot1, rot2, rot_online_o_proj
