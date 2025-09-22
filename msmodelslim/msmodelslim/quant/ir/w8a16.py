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

import torch
import torch.nn as nn
import torch.nn.functional as F

from msmodelslim.core import QDType, QParam, QStorage, calculate_qparam
from msmodelslim.core import dequantize, fake_quantize
from msmodelslim.core.QAL import QABCRegistry
from msmodelslim.core.QAL import QScope, QScheme
from msmodelslim.quant.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter
from .const import (
    fp16_placeholder_sym,
    fp16_placeholder_asym,
    int8_per_channel_sym,
    int8_per_channel_asym
)


@QABCRegistry.multi_register(
    dispatch_key=[
        (fp16_placeholder_sym, int8_per_channel_sym),
        (fp16_placeholder_sym, int8_per_channel_asym),
        (fp16_placeholder_asym, int8_per_channel_sym),
        (fp16_placeholder_asym, int8_per_channel_asym),
    ],
    abc_type=AutoFakeQuantLinear
)
@logger_setter('msmodelslim.quant.ir.w8a16')
class W8A16PerChannelFakeQuantLinear(AutoFakeQuantLinear):
    """
    W8A16 量化方式的伪量化IR。
    
    W8A16 量化方式可以用以下参数描述：
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight_offset: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.int32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int8
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
    """

    def __init__(
            self,
            x_q_param: QParam,
            w_q_param: QParam,
            w_q: QStorage,
            bias: torch.Tensor
    ):
        super().__init__()

        self.w_sym = w_q_param.scheme.symmetric
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"], requires_grad=False)
        self.weight_offset = nn.Parameter(w_q_param.ext["offset"], requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def __repr__(self) -> str:
        return f"W8A16PerChannelFakeQuantLinear(symmetric={self.w_sym})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=self.w_sym), 
            ext={"scale": self.weight_scale, "offset": self.weight_offset}
            )
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT8, value=self.weight.data).T, w_q_param).T
        return F.linear(x, weight_q_dq.value, self.bias)

