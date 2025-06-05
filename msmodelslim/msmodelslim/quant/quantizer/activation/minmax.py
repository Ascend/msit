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

from typing import Tuple

import torch
from torch import distributed as dist

from msmodelslim.quant.quantizer.activation.base import ObserverStrategy, OBSERVER_STRATEGY_REGISTRY, \
    ActQuantMethodConfig
from msmodelslim.quant.quantizer.base.const import QuantMethod


class ActMinMaxConfig(ActQuantMethodConfig):
    type: QuantMethod = QuantMethod.MINMAX


@OBSERVER_STRATEGY_REGISTRY.register(QuantMethod.MINMAX.value)
class MinMaxStrategy(ObserverStrategy):

    def __init__(self, config: ActMinMaxConfig = ActMinMaxConfig()):
        self.config = config
        self.min_val = None
        self.max_val = None

    def update(self, x: torch.Tensor):

        if self.min_val is None:
            self.min_val = torch.min(x)
        else:
            self.min_val = torch.min(self.min_val, torch.min(x))

        if self.max_val is None:
            self.max_val = torch.max(x)
        else:
            self.max_val = torch.max(self.max_val, torch.max(x))

    def update_with_group(self, x: torch.Tensor, group: dist.ProcessGroup):
        self.update(x)

        if not dist.is_initialized():
            raise RuntimeError("MinMaxStrategy's update_with_group requires distributed enabled")

        dist.all_reduce(self.min_val, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(self.max_val, op=dist.ReduceOp.MAX, group=group)

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.min_val is None or self.max_val is None:
            raise RuntimeError(
                "Trying to get stats but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated."
                "Please check your model and quant config.")

        return self.min_val, self.max_val

    def reset(self):
        self.min_val = None
        self.max_val = None
