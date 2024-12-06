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

import numpy as np
import functools
from components.debug.opcheck.graph_parser import OpInfo


def _approximate_equal(context: OpInfo):
    input0, input1 = context.param.get("input_arrays")
    reshape_input = [1]
    reshape_input[0] = functools.reduce(lambda x, y: x * y, input0.shape[:])
    input_x = input0.reshape(reshape_input)
    input_y = input1.reshape(reshape_input)
    input_x = input_x.astype('float32')
    input_y = input_y.astype('float32')
    res_sub = np.subtract(input_x, input_y)
    res_abs = np.abs(res_sub)
    res_abs = res_abs.astype(input_x.dtype)
    tol = context.param.get("tolerance")
    tol_data = np.broadcast_to(np.array(tol, input_x.dtype),
                               input_x.shape)
    zero_rb = np.broadcast_to(np.array(0, "float16"), input_x.shape)
    one_rb = np.broadcast_to(np.array(1, "float16"), input_x.shape)
    res = np.where(res_abs <= tol_data, one_rb, zero_rb)

    return res.astype(context.param.get("output_dtypes")[0])