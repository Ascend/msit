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
import torch.distributed as dist

from msmodelslim.quant.quantizer.activation.base import BaseObserver, ActQuantScopeConfig
from msmodelslim.quant.quantizer.activation.base import OBSERVER_REGISTRY
from msmodelslim.quant.quantizer.base.const import QuantScope


class PerTensorConfig(ActQuantScopeConfig):
    type: QuantScope = QuantScope.PER_TENSOR


# 按Tensor统计（全局）
@OBSERVER_REGISTRY.register(QuantScope.PER_TENSOR.value)
class PerTensorObserver(BaseObserver):
    """
    按张量统计的观察者，对所有维度进行统计。
    
    该类观察者会对输入张量的所有维度进行统计，得到一个全局的统计结果。
    适用于需要对整个张量进行统一统计的场景。
    """

    def __init__(self, config: PerTensorConfig):
        super().__init__()


class PerAxisConfig(ActQuantScopeConfig):
    type: QuantScope = QuantScope.PER_AXIS
    axis: int = 0


@OBSERVER_REGISTRY.register(QuantScope.PER_AXIS.value)
class PerAxisObserver(BaseObserver):
    """
    按轴统计的观察者，对指定轴以外的维度进行统计。
    
    该类观察者会对输入张量的指定轴以外的维度进行统计，得到一个按轴分布的统计结果。
    适用于需要对张量的特定维度进行独立统计的场景。
    
    Note:
        x.shape = (1, 2, 3, 4), axis = 1
        
        => scale.shape = (2)
        => zero_point.shape = (2)
    """

    def __init__(self, config: PerAxisConfig):
        """
        初始化按轴统计的观察者。
        
        参数:
            strategy: 统计策略，用于收集和处理统计信息
            axis: 遍历某个轴进行per_tensor
        """
        super().__init__()
        self.axis = config.axis
        self.observer_count = 0
        self.per_tensor_observers = []
        self.uninitialized = torch.tensor([1])

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):

        if self.axis >= x.ndim:
            raise ValueError(f"Axis {self.axis} is out of bounds for tensor of dimension {x.ndim}")

        x = x.clone().detach().permute(self.axis, *[d for d in range(x.ndim) if d != self.axis])

        self._init_observers(x)

        for i in range(self.observer_count):
            self.per_tensor_observers[i].update(x[i], sync, group)

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.uninitialized.item():
            raise RuntimeError("Observer not initialized")

        min_val = torch.stack(
            [per_tensor_observer.get_min_max()[0] for per_tensor_observer in self.per_tensor_observers])
        max_val = torch.stack(
            [per_tensor_observer.get_min_max()[1] for per_tensor_observer in self.per_tensor_observers])

        return min_val, max_val

    def _init_observers(self, x: torch.Tensor):
        if self.uninitialized.item():
            self.observer_count = x.shape[0]
            self.per_tensor_observers = [PerTensorObserver(self.config) for _ in range(self.observer_count)]
            self.uninitialized = torch.tensor([0])


class PerTokenConfig(ActQuantScopeConfig):
    type: QuantScope = QuantScope.PER_TOKEN


@OBSERVER_REGISTRY.register(QuantScope.PER_TOKEN.value)
class PerTokenObserver(BaseObserver):
    """
    按Token统计的观察者，对每个Token进行统计。
    
    该类观察者会对输入张量的每个Token进行统计，得到一个按Token分布的统计结果。
    适用于需要对每个Token进行独立统计的场景。
    """

    def __init__(self, config: PerTokenConfig = PerTokenConfig()):
        super().__init__()
        self.sync_stats = False

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
        """
        更新观察者的统计信息。

        根据输入张量和内部确定的缩减维度，更新统计策略的状态。

        参数:
            x: 输入张量，包含需要观察的数据
        """
        self.strategy.reset()
        super().update(x, sync, group)
