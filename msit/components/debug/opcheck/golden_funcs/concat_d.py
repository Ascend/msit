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

import numpy
from components.debug.opcheck.conversion.shape_convert import update_axis_for_npu_inner_format
from components.debug.opcheck.graph_parser import OpInfo


def concat_d(context: OpInfo):
    ori_shape = context.param.get("stc_ori_inputs")[0]
    axis = context.param.get("concat_dim")
    input_format = context.param.get("stc_input_formats")[0]
    ori_format = context.param.get("stc_input_ori_formats")[0]
    concat_dim = update_axis_for_npu_inner_format(ori_shape, axis, input_format, ori_format)
    return numpy.concatenate(context.param.get("input_arrays"), axis=concat_dim)