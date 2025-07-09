#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from functools import lru_cache
from typing import Tuple, Callable, List

import torch


@lru_cache(maxsize=None)
def get_kia_fake_quantize() -> Callable:
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import fake_quantize as kia_fake_quantize
    return kia_fake_quantize


@lru_cache(maxsize=None)
def get_kia_init_weight_quant_normal() -> Callable:
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import \
        init_weight_quant_normal as kia_init_weight_quant_normal
    return kia_init_weight_quant_normal


@lru_cache(maxsize=None)
def get_kia_linear_quantization_params() -> Callable:
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import \
        linear_quantization_params as kia_linear_quantization_params
    return kia_linear_quantization_params


@lru_cache(maxsize=None)
def get_kia_handle_progressive_quant() -> Callable:
    from msmodelslim.pytorch.lowbit.atomic_power_outlier import \
        handle_progressive_quant as kia_handle_progressive_quant
    return kia_handle_progressive_quant


def linear_quantization_params(bit: int, 
                               x_min: torch.Tensor, 
                               x_max: torch.Tensor,
                               intergral_zero_point: bool,
                               q_signed: bool, 
                               sym: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算量化参数
    
    参数:
        bit: 量化的位数
        x_min: 最小值
        x_max: 最大值
        intergral_zero_point: 是否使用整数偏移
        q_signed: 是否是有符号量化
        sym: 是否是对称量化
    
    返回值:
        Tuple[torch.Tensor, torch.Tensor]: 分别是量化缩放因子，量化零点
    """
    return get_kia_linear_quantization_params()(bit, x_min, x_max, intergral_zero_point, q_signed, sym)


def fake_quantize(tensor: torch.Tensor,
                  scale: torch.Tensor,
                  zero_point: torch.Tensor,
                  bits: int = 8,
                  is_signed: bool = True,
                  round_opt: bool = False,
                  is_fp=False,
                  dequant=True,
                  group_size=-1,
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    进行伪量化操作
    
    参数:
        tensor: 需要伪量化的张量
        scale: 量化的比例因子
        zero_point: 量化的零点
        bits: 量化的位数
        is_signed: 是否是有符号量化
        round_opt: 是否使用四舍五入
        is_fp: 是否使用浮点量化
        dequant: 是否使用反量化
        group_size: 分组大小
    
    返回值:
        Tuple[torch.Tensor, torch.Tensor]: 分别是伪量化后的张量和反量化后的张量

    """

    return get_kia_fake_quantize()(tensor, scale, zero_point, bits, is_signed, round_opt, is_fp, dequant, group_size)


def init_weight_quant_normal(weight: torch.Tensor,
                             bits: int = 8,
                             is_sym=True,
                             is_signed: bool = True,
                             intergral_zp=True,
                             admm=(False, 1000),
                             round_opt=False,
                             mm_tensor=True,
                             fake_quant=True,
                             hqq=False,
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    初始化权重量化
    
    参数:
        weight: 权重张量
        bits: 量化的位数
        is_sym: 是否是对称量化
        is_signed: 是否是有符号量化
        intergral_zp: 是否使用整数零点
        admm: 是否使用ADMM算法
        round_opt: 是否使用四舍五入
        mm_tensor: 是否使用PerTensor量化
        fake_quant: 是否使用伪量化
        hqq: 是否使用HQQ算法
    返回值:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 分别是量化后的权重，反量化后的权重，量化比例因子，量化零点
    """
    return get_kia_init_weight_quant_normal()(weight, bits, is_sym, is_signed, intergral_zp, admm, round_opt, mm_tensor,
                                              fake_quant, hqq)


def handle_progressive_quant(
        weight: torch.Tensor,
        group_size: int = 0,
        num_bits: int = 4,
        per_channel: bool = True,
        w_sym: bool = True,
        use_hqq: bool = False,
) -> Tuple[
    torch.Tensor,  # 反量化权重
    List[torch.Tensor],  # 缩放因子
    torch.Tensor,  # 量化后（整型）权重
    List[torch.Tensor]  # 零点
]:
    """
    对权重 w4 进行分阶段量化
    
    参数:
        weight（torch.Tensor）: 待量化的权重
        group_size（int）: 分组大小，0 表示不进行分组
        num_bits（int）: 量化位数，默认为 4
        per_channel（bool）: 是否是通道量化，默认为 True
        w_sym（bool）: 是否是对称量化，默认为 True
        use_hqq（bool）: 是否使用HQQ算法，默认为 False

    返回值:
        Tuple[
            torch.Tensor,  # 反量化权重
            List[torch.Tensor],  # 缩放因子
            torch.Tensor,  # 量化后（整型）权重
            List[torch.Tensor]  # 零点
        ]
    """
    if weight.numel() < group_size:
        raise ValueError(f"Cannot perform group quantization with group size {group_size} "
                         f"for weight with {weight.numel()} elements")
    
    return get_kia_handle_progressive_quant()(weight, group_size, num_bits, per_channel, w_sym, use_hqq)

