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
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
from components.debug.opcheck.utils import broadcast_to_maxshape
from components.debug.opcheck.graph_parser import OpInfo


def broadcast_inputs_shape(x, mean, context):
    shape_x = x.shape
    format_x = context.output_formats[0]

    if format_x in ("ND", "NCHW"):
        if len(shape_x) == 1:
            index_c = 0
        else:
            index_c = 1
    elif format_x == "NHWC":
        if len(shape_x) == 1:
            index_c = 0
        else:
            index_c = 3
    else:
        c1 = shape_x[1]
        c0 = shape_x[4]
    shape_mean = mean.shape
    if format_x in ("ND", "NCHW", "NHWC"):
        shape_mean = [1] * len(shape_x[:index_c]) + list(shape_mean) \
            + [1] * len(shape_x[index_c + 1:])
    else:
        shape_mean = [1, c1, 1, 1, c0]
    return shape_mean


def _fused_scale_bias_compute(x, mean, variance, scale, bias):
    shape_x = x.shape
    shape_mean = mean.shape
    shape_list = broadcast_to_maxshape([shape_x, shape_mean])
    x_broadcast = np.broadcast_to(x, shape_list[-1])
    mean_broadcast = np.broadcast_to(mean, shape_list[-1])
    var_broadcast = np.broadcast_to(variance, shape_list[-1])
    mean_add = np.add(x_broadcast, mean_broadcast)
    res_y = np.multiply(var_broadcast, mean_add)
    scale_broad = np.broadcast_to(scale, shape_list[-1])
    bias_broad = np.broadcast_to(bias, shape_list[-1])
    res_y = res_y.astype("float32")
    scale_broad = scale_broad.astype("float32")
    res_tmp = np.multiply(res_y, scale_broad)
    return np.add(res_tmp, bias_broad)


def _fused_scale_compute(x, mean, variance, scale):
    shape_x = x.shape
    shape_mean = mean.shape
    shape_list = broadcast_to_maxshape([shape_x, shape_mean])
    x_broadcast = np.broadcast_to(x, shape_list[-1])
    mean_broadcast = np.broadcast_to(mean, shape_list[-1])
    var_broadcast = np.broadcast_to(variance, shape_list[-1])
    mean_add = np.add(x_broadcast, mean_broadcast)
    res_y = np.multiply(var_broadcast, mean_add)
    scale_broad = np.broadcast_to(scale, shape_list[-1])
    res_y = res_y.astype("float32", copy=False)
    scale_broad = scale_broad.astype("float32", copy=False)
    return np.multiply(res_y, scale_broad)


def __fused_compute(x, mean, variance):
    shape_x = x.shape
    shape_mean = mean.shape
    shape_list = broadcast_to_maxshape([shape_x, shape_mean])
    x_broadcast = np.broadcast_to(x, shape_list[-1])
    mean_broadcast = np.broadcast_to(mean, shape_list[-1])
    var_broadcast = np.broadcast_to(variance, shape_list[-1])
    mean_add = np.add(x_broadcast, mean_broadcast)
    return np.multiply(var_broadcast, mean_add)

def _bninference_d(context: OpInfo):
    x, mean, variance, scale, bias = context.param.get("input_arrays")
    dtype = x.dtype
    if str(dtype) == "bfloat16": #tf.bfloat16.as_numpy_dtype:
        x = x.astype("float32")
        mean = mean.astype("float32")
        variance = variance.astype("float32")
        if scale is not None:
            scale = scale.astype("float32")
            bias = bias.astype("float32")
    mean.shape = broadcast_inputs_shape(x, mean, context)
    variance.shape = mean.shape
    if scale is not None and bias is not None:
        scale.shape = mean.shape
        bias.shape = mean.shape
        res = _fused_scale_bias_compute(x, mean, variance, scale, bias)
    elif scale is not None and bias is None:
        scale.shape = mean.shape
        res = _fused_scale_compute(x, mean, variance, scale)
    else:
        res = __fused_compute(x, mean, variance)
    return res.astype(dtype, copy=False)
    
