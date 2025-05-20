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
from torch import distributed as dist

from msmodelslim.quant.quantizer.activation.base import StatisticsStrategy, STATISTIC_STRATEGY_REGISTRY
from msmodelslim.quant.quantizer.base.const import ActivationQuantMethod


@STATISTIC_STRATEGY_REGISTRY.register(ActivationQuantMethod.MINMAX.value)
class MinMaxStatistic(StatisticsStrategy):

    def __init__(self):
        self.min_val = None
        self.max_val = None

    def update_stats(self, x: torch.Tensor, reduce_dims: list[int], keep_dims: bool = False, sync_stats: bool = False):

        if self.min_val is None:
            self.min_val = torch.amin(x, dim=reduce_dims, keepdim=keep_dims)
        else:
            self.min_val = torch.min(self.min_val, torch.amin(x, dim=reduce_dims, keepdim=keep_dims))

        if self.max_val is None:
            self.max_val = torch.amax(x, dim=reduce_dims, keepdim=keep_dims)
        else:
            self.max_val = torch.max(self.max_val, torch.amax(x, dim=reduce_dims, keepdim=keep_dims))

        if sync_stats and dist.is_initialized():
            dist.all_reduce(self.min_val, op=dist.ReduceOp.MIN)
            dist.all_reduce(self.max_val, op=dist.ReduceOp.MAX)

    def get_stats(self) -> torch.Tensor:

        if self.min_val is None or self.max_val is None:
            raise RuntimeError(
                "Trying to get stats but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated."
                "Please check your model and quant config.")

        return torch.stack([self.min_val, self.max_val], dim=0)

    def clear_stats(self):
        self.min_val = None
        self.max_val = None
