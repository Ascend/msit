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
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger
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


def _get_available_device(index: int = 0) -> torch.device:
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device("npu:{}".format(index))
    elif hasattr(torch, 'cuda') and torch.cuda.is_available():
        return torch.device("cuda:{}".format(index))
    else:
        return torch.device("cpu")


def _get_full_module_name(target_module: nn.Module, request: BatchProcessRequest) -> str:
    """
    通过遍历request.module获取完整的模块名称。

    Args:
        target_module: 目标模块
        request: 处理请求，包含模块前缀信息

    Returns:
        完整的模块名称

    Raises:
        UnsupportedError: 如果找不到完整的模块名称
    """
    for name, module in request.module.named_modules():
        if module is target_module:
            # 拼接完整路径：request.name + 相对路径
            full_name = f"{request.name}.{name}" if name else request.name
            return full_name

    # 如果找不到，抛出UnsupportedError
    raise UnsupportedError(f"Cannot find full module name for {target_module}")


def _convert_hookir_to_wrapper(module: nn.Module) -> None:
    """
    将模块中的HookIR转换为Wrapper

    Args:
        module: 要处理的模块
    """
    # 遍历模块中的所有子模块
    for name, sub_module in module.named_modules():
        if hasattr(sub_module, '_forward_pre_hooks'):
            # 遍历模块的所有前向钩子
            for hook in sub_module._forward_pre_hooks.values():
                # 检查是否是HookIR类型
                if isinstance(hook, qir.HookIR):
                    # 将hook_ir转换为wrapper
                    wrapper = hook.wrapper_module(sub_module)
                    # 将wrapper替换模块
                    module.set_submodule(name, wrapper)
                    get_logger().info(f"Converted {type(hook)} to wrapper for module: {name}")


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
        device = _get_available_device()

        get_logger().info(f"Creating rotation matrices on device: {device}")

        self.rot = QuaRotUtils.create_rot(model_dim, block_size=self.config.block_size, device=device)
        self.rot_att_v = QuaRotUtils.create_rot(head_dim, block_size=self.config.block_size, device=device)

        get_logger().info(f"Rotation matrices created on device: {device}")

        if self.config.online:
            get_logger().info(f"Creating online rotation matrices on device: {device}")
            rot1, rot2, self.rot_online_o_proj = self._add_online_rotations(self.model)
            identity = torch.eye(int(self.adapter.get_head_dim())).to(self.rot_online_o_proj)
            self.rot_online_down_proj = (rot1, rot2)
            self.online_rotation_info = qir.QuarotOnlineRotationInfo(self.rot_online_o_proj, identity, rot1, rot2,
                                                                     self.config.max_tp_size)
            get_logger().info(f"Online rotation matrices created on device: {device}")

        # 层融合
        get_logger().info(f"Fusing layer norm and linear on device: {device}")
        QuaRotUtils.fuse_ln_linear(
            self.model.get_submodule(self.adapter.get_pre_head_layernorm()),
            [self.model.get_submodule(self.adapter.get_lm_head())])
        get_logger().info(f"Layer norm and linear fused on device: {device}")
        # 旋转
        get_logger().info(f"Rotating embedding on device: {device}")
        QuaRotUtils.rotate_embedding(self.model.get_submodule(self.adapter.get_embedding()), self.rot, device)
        get_logger().info(f"Embedding rotated on device: {device}")

        if hasattr(self.model.config, 'tie_word_embeddings') and not self.model.config.tie_word_embeddings:
            get_logger().info(f"Rotating head on device: {device}")
            QuaRotUtils.rotate_head(self.model.get_submodule(self.adapter.get_lm_head()), self.rot, device)
            get_logger().info(f"Head rotated on device: {device}")

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
                    # 获取完整的模块名称
                    full_module_name = _get_full_module_name(down_proj, request)
                    # 使用完整的layer_name注册Kronecker HookIR实例作为hook（用于down_proj）
                    hook_ir = qir.QuarotKroneckerRotationHookIR(
                        full_module_name,
                        self.online_rotation_info
                    )
                    hook_handle = down_proj.register_forward_pre_hook(hook_ir)
                    hook_ir.set_hook_handle(hook_handle)

            if online_oproj_rotation:
                for o_proj, _ in ov_pairs.items():
                    # 获取完整的模块名称
                    full_module_name = _get_full_module_name(o_proj, request)
                    # 使用完整的layer_name注册普通HookIR实例作为hook（用于o_proj）
                    hook_ir = qir.QuarotHeadsRotationHookIR(
                        full_module_name,
                        self.online_rotation_info
                    )
                    hook_handle = o_proj.register_forward_pre_hook(hook_ir)
                    hook_ir.set_hook_handle(hook_handle)

    def post_run(self) -> None:
        """遍历模型中的HookIR，如果是自己添加的HookIR则进行转换"""
        _convert_hookir_to_wrapper(self.model)

    def _add_online_rotations(self, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(model.config, 'intermediate_size'):
            raise ValueError("Model config must contain 'intermediate_size' for online rotation")
        size_1, size_2 = QuaRotUtils.get_decompose_dim(model.config.intermediate_size)
        rot1 = QuaRotUtils.create_rot(size_1, self.config.max_tp_size, device=model.device)
        rot2 = QuaRotUtils.create_rot(size_2, -1, device=model.device)

        num_heads = self.num_attn_heads
        rot_online_o_proj = QuaRotUtils.create_rot(num_heads, self.config.max_tp_size, device=model.device)

        return rot1, rot2, rot_online_o_proj
