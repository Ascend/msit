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
from typing import Tuple, Optional

import torch
from torch import nn as nn
from torch import distributed as dist

from msmodelslim.quant.kia.utils import linear_quantization_params, fake_quantize
from msmodelslim.quant.quantizer.activation.base import ActQuantConfig, OBSERVER_REGISTRY, \
    OBSERVER_STRATEGY_REGISTRY


class ActivationQuantizer(nn.Module):

    @classmethod
    def from_config(cls, config: ActQuantConfig, **kwargs):
        return cls(config, **kwargs)

    def __init__(self, config: ActQuantConfig, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.sync = sync
        self.group = group
        self.config = config
        self.observer = OBSERVER_REGISTRY.get_by_name(config.scope.type.value)(config)
        self.strategy = OBSERVER_STRATEGY_REGISTRY.get_by_name(config.method.type.value)(config)
        self.observer.set_strategy(self.strategy)

    def get_scale_offset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        min_val, max_val = self.observer.get_min_max()
        scale, zero_point = linear_quantization_params(self.config.base.bits,
                                                       min_val,
                                                       max_val,
                                                       True,
                                                       self.config.base.signed,
                                                       self.config.base.symmetric)
        return scale, zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observer.update(x, sync=self.sync, group=self.group)
        scale, zero_point = self.get_scale_offset()
        _, dequant_x = fake_quantize(x, scale, zero_point, self.config.base.bits, self.config.base.signed)
        return dequant_x
