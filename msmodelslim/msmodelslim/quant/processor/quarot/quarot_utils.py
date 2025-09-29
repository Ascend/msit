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

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from msmodelslim.quant.processor.quarot.hadamard import random_hadamard_matrix
from msmodelslim.utils.exception import UnsupportedError


class QuaRotUtils:
    """QuaRot 工具类，包含所有静态工具方法"""

    @staticmethod
    def is_power_of_two(n: int) -> bool:
        """检查一个数是否为2的幂"""
        return n > 0 and (n & (n - 1)) == 0

    @staticmethod
    def get_decompose_dim(n: int) -> Tuple[int, int]:
        """获取分解维度"""
        sup_list = {1, 2} | {4 * i for i in range(1, 65)}
        max_sup = max(sup_list)

        min_a = int(math.sqrt(n))
        if min_a * min_a < n:
            min_a += 1
        for a in range(min_a, max_sup + 1):
            tmp = a * a - n
            if tmp < 0:
                continue
            b = int(math.sqrt(tmp))
            if b * b == tmp and (a - b) in sup_list and (a + b) in sup_list:
                return a - b, a + b

        raise UnsupportedError(f"Can not decompose {n}")

    @staticmethod
    def create_rot(
            size: int,
            block_size: int = -1,
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """创建旋转矩阵"""
        if block_size == -1:
            transform_dim = size
        else:
            transform_dim = block_size

        rot = random_hadamard_matrix(transform_dim, dtype, device)

        if block_size != -1:
            rot = rot.repeat(size // block_size, 1, 1)
            rot = torch.block_diag(*rot)

        return rot

    @staticmethod
    def fuse_ln_linear(ln: nn.Module, linear_list: List[nn.Module]) -> None:
        """将 LayerNorm 和 Linear 层融合，并将 LayerNorm 权重设置为1.0"""
        for linear in linear_list:
            linear_dtype = linear.weight.dtype
            curr_weight = linear.weight.to(dtype=torch.float32)

            if hasattr(ln, 'bias'):
                if linear.bias is None:
                    linear.bias = torch.nn.Parameter(
                        torch.zeros(linear.out_features, dtype=linear_dtype, device=linear.weight.device))
                ln_bias = ln.bias.to(dtype=torch.float32)
                linear.bias.data.copy_(linear.bias.data + torch.matmul(curr_weight, ln_bias))
                linear.bias.to(linear_dtype)

            linear.weight.data.copy_(curr_weight * ln.weight.to(dtype=torch.float32)).to(linear_dtype)

            del curr_weight

        # 将 LayerNorm 权重设置为1.0
        ln.weight.data.fill_(1.0)

        return

    @staticmethod
    def rotate_embedding(embedding_module: nn.Module, rot: torch.Tensor, device: torch.device) -> None:
        """旋转嵌入层权重"""
        # embedding_module 就是嵌入层 (通常是 nn.Embedding)
        ori_device = embedding_module.weight.device
        embedding_module.to(device=device)
        dtype = embedding_module.weight.dtype
        device = embedding_module.weight.device
        weight = embedding_module.weight.to(device=device, dtype=torch.float32)
        embedding_module.weight.data.copy_(torch.matmul(weight, rot.to(device)).to(device=device, dtype=dtype))
        embedding_module.to(device=ori_device)
        del weight

        return

    @staticmethod
    def rotate_head(lm_head_module: nn.Module, rot: torch.Tensor, device: torch.device) -> None:
        """旋转输出头权重"""
        ori_device = lm_head_module.weight.device
        lm_head_module.to(device=device)
        
        dtype = lm_head_module.weight.dtype
        device = lm_head_module.weight.device

        weight = lm_head_module.weight.to(device=device, dtype=torch.float32)
        lm_head_module.weight.data.copy_(torch.matmul(weight, rot.to(device)).to(device=device, dtype=dtype))
        lm_head_module.to(device=ori_device)
        del weight

        return

    @staticmethod
    def rotate_attention_mlp_input(norm_linear_pairs: Dict[nn.Module, List[nn.Module]],
                                   rot: torch.Tensor) -> None:
        """旋转注意力和MLP的输入"""
        for _, linear_layers in norm_linear_pairs.items():
            for linear in linear_layers:
                dtype = linear.weight.dtype
                device = linear.weight.device
                weight = linear.weight.to(device=device, dtype=torch.float32)
                linear.weight.data.copy_(torch.matmul(weight, rot.to(device)).to(device=device, dtype=dtype))
                del weight
        return

    @staticmethod
    def rotate_attention_ov_output(ov_pairs: Dict[nn.Module, nn.Module], rot: torch.Tensor,
                                   rot_att_v: torch.Tensor, num_kv_heads: int) -> None:
        """旋转注意力o层和v层的输出"""
        rot_transformed = torch.block_diag(*[rot_att_v] * num_kv_heads)
        for o_proj, v_proj in ov_pairs.items():
            dtype = o_proj.weight.dtype
            device = o_proj.weight.device

            # 旋转o_proj
            weight = o_proj.weight.to(device=device, dtype=torch.float32)
            o_proj.weight.data.copy_(torch.matmul(rot.T.to(device), weight).to(device=device, dtype=dtype))

            if o_proj.bias is not None:
                bias = o_proj.bias.to(device=device, dtype=torch.float32)
                o_proj.bias.data.copy_(torch.matmul(rot.T.to(device), bias).to(device=device, dtype=dtype))

            # 旋转v_proj
            weight_v = v_proj.weight.to(device=device, dtype=torch.float32)
            v_proj.weight.data.copy_(
                torch.matmul(rot_transformed.T.to(device), weight_v).to(device=device, dtype=dtype))

            if v_proj.bias is not None:
                bias_v = v_proj.bias.to(device=device, dtype=torch.float32)
                v_proj.bias.data.copy_(
                    torch.matmul(rot_transformed.T.to(device), bias_v).to(device=device, dtype=dtype))

            del weight, weight_v

        return

    @staticmethod
    def rotate_mlp_output(up_down_pairs: Dict[nn.Module, nn.Module], rot: torch.Tensor) -> None:
        """旋转MLP输出层权重（down_proj）"""
        for _, down_proj in up_down_pairs.items():
            dtype = down_proj.weight.dtype
            device = down_proj.weight.device

            weight = down_proj.weight.to(device=device, dtype=torch.float32)
            down_proj.weight.data.copy_(torch.matmul(rot.T.to(device), weight).to(device=device, dtype=dtype))

            if down_proj.bias is not None:
                bias = down_proj.bias.to(device=device, dtype=torch.float32)
                down_proj.bias.data.copy_(torch.matmul(rot.T.to(device), bias).to(device=device, dtype=dtype))

            del weight

        return

    @staticmethod
    def rotate_o_proj_input(ov_pairs: Dict[nn.Module, nn.Module], rot: torch.Tensor,
                            rot_online: Optional[torch.Tensor] = None,
                            online: bool = False, num_attn_heads: int = 0) -> None:
        """旋转o_proj层的输入"""
        for o_proj, _ in ov_pairs.items():
            dtype = o_proj.weight.dtype
            device = o_proj.weight.device
            rot = rot.to(device)

            if online and rot_online is not None:
                rot_online = rot_online.to(device)
                h_full = torch.kron(rot_online, rot)
                weight = o_proj.weight.to(device=device, dtype=torch.float32)
                o_proj.weight.data.copy_(torch.matmul(weight, h_full.to(device)).to(device=device, dtype=dtype))
                del h_full
            else:
                rot_block = torch.block_diag(*[rot] * num_attn_heads)
                weight = o_proj.weight.to(device=device, dtype=torch.float32)
                o_proj.weight.data.copy_(torch.matmul(weight, rot_block.to(device)).to(device=device, dtype=dtype))

            del weight

        return

    @staticmethod
    def rotate_down_proj(up_down_pairs: Dict[nn.Module, nn.Module], rot1: torch.Tensor,
                         rot2: torch.Tensor) -> None:
        """旋转down_proj层权重"""
        for _, down_proj in up_down_pairs.items():
            dtype = down_proj.weight.dtype
            device = down_proj.weight.device

            init_shape = down_proj.weight.data.shape
            weight = down_proj.weight.data.view(-1, rot1.shape[0], rot2.shape[0])
            weight = torch.matmul(weight, rot2.to(device, dtype)).to(device=device, dtype=dtype)
            weight = torch.matmul(rot1.T.to(device, dtype), weight).reshape(init_shape).to(device=device, dtype=dtype)
            down_proj.weight.data.copy_(weight)

            del weight

        return
