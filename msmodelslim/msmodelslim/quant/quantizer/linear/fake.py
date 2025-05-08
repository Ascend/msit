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


import torch
from torch import nn
from torch.nn import functional as F

from msmodelslim.quant.quantizer.base.fake import BaseFakeQuantizer
from msmodelslim.quant.quantizer.linear.config import LinearQuantConfig


class W8A8LinearFakeQuantizer(BaseFakeQuantizer):
    def __init__(self,
                 cfg: LinearQuantConfig,
                 input_scale: torch.Tensor,
                 input_offset: torch.Tensor,
                 deq_scale: torch.Tensor,
                 quant_bias: torch.Tensor,
                 weight: torch.Tensor,
                 ):
        super().__init__(cfg)
        self.cfg = cfg
        self.weight = nn.Parameter(weight)
        self.input_scale = nn.Parameter(input_scale)
        self.input_offset = nn.Parameter(input_offset)
        self.deq_scale = nn.Parameter(deq_scale, requires_grad=False)
        self.quant_bias = nn.Parameter(quant_bias, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.input_scale + self.input_offset
        x = F.linear(x, self.weight)
        x = (x + self.quant_bias) * self.deq_scale
        return x


class WXA16LinearFakeQuantizer(BaseFakeQuantizer):
    def __init__(self,
                 cfg: LinearQuantConfig,
                 weight: torch.Tensor,
                 bias: torch.Tensor,
                 weight_scale: torch.Tensor,
                 weight_offset: torch.Tensor,
                 ):
        super().__init__(cfg)
        self.cfg = cfg
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.weight_scale = nn.Parameter(weight_scale)
        self.weight_offset = nn.Parameter(weight_offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.weight_scale + self.weight_offset
        x = F.linear(x, weight, self.bias)
        return x
