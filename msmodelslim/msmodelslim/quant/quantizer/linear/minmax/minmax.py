#  -*- coding: utf-8 -*-
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

from typing import Tuple, Optional

import torch
from torch import nn

from msmodelslim.quant.kia.utils import fake_quantize, init_weight_quant_normal
from msmodelslim.quant.quantizer.base.const import WeightQuantMethod, WeightQuantScope
from msmodelslim.quant.quantizer.linear.base import BaseLinearQuantizer, BaseWeightQuantizer
from msmodelslim.quant.quantizer.linear.base import LINEAR_QUANTIZER_REGISTRY
from msmodelslim.quant.quantizer.linear.config import WeightQuantConfig, LinearQuantConfig


class MinMaxWeightQuantizer(BaseWeightQuantizer):
    def __init__(self, cfg: WeightQuantConfig):
        super().__init__(cfg)

        self.register_buffer('weight_scale', torch.zeros(1))
        self.register_buffer('weight_offset', torch.zeros(1))

    def get_scale_offset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.weight_scale, self.weight_offset

    def get_quantized_weight_and_bias(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        int_weight, dequant_weight = fake_quantize(
            weight,
            self.weight_scale,
            self.weight_offset,
            self.cfg.bits,
        )
        return int_weight, dequant_weight

    def quant(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        return fake_quantize(weight, self.weight_scale, self.weight_offset)

    def forward(self,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, dequant_weight, self.weight_scale, self.weight_offset = init_weight_quant_normal(
            weight,
            bits=self.cfg.bits,
            mm_tensor=self.cfg.scope == WeightQuantScope.PER_TENSOR,
            fake_quant=True
        )

        return dequant_weight, bias


@LINEAR_QUANTIZER_REGISTRY.register()
class MinMaxLinearQuantizer(BaseLinearQuantizer):

    def __init__(self, module: nn.Module, cfg: LinearQuantConfig):
        super().__init__(module, cfg)

    @staticmethod
    def match(module: nn.Module, cfg: LinearQuantConfig) -> bool:
        return isinstance(module, nn.Linear) and cfg.w_cfg.method == WeightQuantMethod.MINMAX

    def _create_weight_quantizer(self, cfg: WeightQuantConfig) -> nn.Module:
        return MinMaxWeightQuantizer(cfg)
