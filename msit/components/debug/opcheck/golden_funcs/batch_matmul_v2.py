# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
from typing import Sequence
from components.debug.opcheck.conversion.dtype_convert import bfloat16_conversion
from components.debug.opcheck.utils import  ceil_div, align
from components.debug.opcheck.graph_parser import OpInfo


def get(container: Sequence, idx: int, out_of_range=None):
    """
    If a container contains only one element, return that element for whatever the idx is
    :param container: A container
    :param idx: index
    :param out_of_range: Out of Range return
    :return: element
    """
    if len(container) == 1:
        return container[-1]
    else:
        if idx >= len(container) or (idx < 0 and abs(idx) > len(container)):
            if out_of_range:
                return out_of_range
            else:
                return False
        return container[idx]


def get_quant_pre(scale, offset):
    # convert float32 to uint32
    import struct
    scale_binary = struct.pack('f', scale)
    scale_int = int.from_bytes(scale_binary, byteorder='little')
    # round to nearest, tie to even
    offset_round = round(offset)
    offset_round_clip = min(max(-256, offset_round), 255)
    offset_binary = struct.pack('i', offset_round_clip)
    offset_int = int.from_bytes(offset_binary, byteorder='little') & 0x1FF # get complement of int9
    quant_pre_u64 = (1 << 46) + (offset_int << 37) + scale_int
    return quant_pre_u64


def f32_2_s9(array):
    array_round = np.round(array)
    array_round_clip = np.clip(array_round, -256, 255)
    return array_round_clip


def fractal_shape(dtype):
    """
    >>> fractal_shape('int8')
    (16, 32)
    >>> fractal_shape('float16')
    (16, 16)
    >>> fractal_shape('float32')
    (16, 8)
    """
    import re
    res = re.match(r'[^\d]+(\d+)', dtype)
    assert res is not None
    bit_of_dtype = int(res[1])
    assert (32 * 8) % bit_of_dtype == 0
    return 16, (32 * 8) // bit_of_dtype


def shape_nd_to_Nz(shape, dtype='float16', before_mmad=True):
    """
    >>> shape_nd_to_Nz([3,17])
    [2, 1, 16, 16]
    >>> shape_nd_to_Nz([4,5,3,17])
    [4, 5, 2, 1, 16, 16]
    >>> shape_nd_to_Nz([3,17], dtype='int8')
    [1, 1, 16, 32]
    >>> shape_nd_to_Nz([16,27], dtype='int32')
    [4, 1, 16, 8]
    >>> shape_nd_to_Nz([16,27], dtype='int32', before_mmad=False)
    [2, 1, 16, 16]
    """
    assert (dtype, before_mmad) in (
        ('float16', True), ('float32', False), ('int8', True),
        ('int32', False), ('int32', True), ('float64', False), (('float32', True))
    ), f"Please implement shape ND to FRACTAL_NZ with dtype {dtype} on {shape} {'before mmad' if before_mmad else 'after mmad'}"

    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]
    if before_mmad:
        a0, b0 = fractal_shape(dtype)
    else:
        a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def transpose_input(x, trans):
    if trans:
        array_trans = gen_axes_for_transpose(len(x.shape) - 2, [1, 0])
        return x.transpose(*array_trans)
    return x


def nd_to_fractal_nz(data: np.ndarray, trans_align_nd=False):
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    if data.dtype == "int8":
        m0, n0 = 16, 32
    else:
        m0, n0 = 16, 16
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    if trans_align_nd:
        return data
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def hf_32_input_gerenate(context: OpInfo, input_fp32):   
    input_hf32 = input_fp32.view(np.int32)
    input_hf32 = np.right_shift(np.right_shift(input_hf32, 11) + 1, 1)
    input_hf32 = np.left_shift(input_hf32, 12)
    input_hf32 = input_hf32.view(np.float32)
    return input_hf32


def helper_mm_and_bmm(context: OpInfo, name_parameter):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x1, x2, bias, *_ = context.param.get("original_input_arrays")
    trans_a = context.param.get("name_parameter")[0]
    trans_b = context.param.get("name_parameter")[1]
    format_bias = get(context.param.get("stc_input_formats"), 2)
    format_out = context.param.get("output_formats")[0]

    if context.param.get("impl_mode") == "enable_hi_float_32_execution":
        x1 = hf_32_input_gerenate(context, x1)
        x2 = hf_32_input_gerenate(context, x2)

    if x1.dtype == 'float32':
        a_data = x1.astype('float64')
        b_data = x2.astype('float64')
    else:
        a_data = x1.astype('float32')
        b_data = x2.astype('float32')

    import torch
    a = torch.from_numpy(a_data)
    b = torch.from_numpy(b_data)
    if trans_a:
        if len(a.shape) == 2:
            a = a.t()
        else:
            a = a.transpose(-1,-2)
    if trans_b:
        if len(b.shape) == 2:
            b = b.t()
        else:
            b = b.transpose(-1,-2)
    res_pt = torch.matmul(a, b).numpy()
    if bias is not None:
        if x1.dtype == 'float32':
            bias = bias.astype('float64')
        else:
            bias = bias.astype('float32')
        if format_bias == 'NC1HWC0':
            res_pt = nd_to_fractal_nz(res_pt, True)
            bias = bias.transpose(0, 1, 4, 2, 3).reshape(bias.shape)
            res_pt = torch.from_numpy(res_pt)
            bias = torch.from_numpy(bias)
            res_pt = torch.add(res_pt, bias).numpy()
        else:
            if format_out == "FRACTAL_NZ":
                res_pt = nd_to_fractal_nz(res_pt, True)
                bias_shape = bias.shape[0]
                align_bias = align(bias_shape, 16)
                bias = np.pad(bias, (0, align_bias - bias_shape), 'constant', constant_values=(0, 0))
                res_pt = torch.from_numpy(res_pt)
                bias = torch.from_numpy(bias)
                res_pt = torch.add(res_pt, bias).numpy()
            else:
                res_pt = torch.from_numpy(res_pt)
                bias = torch.from_numpy(bias)
                res_pt = torch.add(res_pt, bias).numpy()

    if format_out == 'FRACTAL_NZ':
        res_pt = nd_to_fractal_nz(res_pt)
    output_dtype = bfloat16_conversion(context.param.get("output_dtypes"))
    return res_pt.astype(output_dtype[0], copy=False)

def _batch_matmul(context: OpInfo):
    return helper_mm_and_bmm(context, ["adj_x1", "adj_x2"])