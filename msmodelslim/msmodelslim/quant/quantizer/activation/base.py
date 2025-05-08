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

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from pydantic import BaseModel
from torch import nn as nn

from msmodelslim.utils.registry import Registry
from msmodelslim.quant.kia.utils import fake_quantize, linear_quantization_params
from msmodelslim.quant.quantizer.base.const import ActivationQuantMethod, ActivationQuantScope


class StatisticsStrategy(ABC):
    """
    统计策略的抽象基类，定义了更新和获取统计值的基本接口。

    统计策略用于收集和处理张量数据的统计信息，如均值、方差等。
    所有具体的统计策略实现都应该继承此类并实现抽象方法。
    """

    @abstractmethod
    def update_stats(self, x: torch.Tensor, reduce_dims: list[int], keep_dims: bool = True):
        """
        更新统计值。

        根据输入张量和指定的缩减维度，更新内部统计状态。

        参数:
            x: 输入张量，包含需要统计的数据
            reduce_dims: 需要缩减的维度列表，指定在哪些维度上进行统计
        """
        pass

    @abstractmethod
    def get_stats(self) -> torch.Tensor:
        """
        获取统计结果。

        返回当前统计策略计算得到的统计值。

        返回:
            torch.Tensor: 统计结果张量
        """
        pass

    @abstractmethod
    def clear_stats(self):
        """
        清除统计状态。

        重置统计状态，为下一次统计做准备。
        """
        pass


STATISTIC_STRATEGY_REGISTRY = Registry[StatisticsStrategy]()


class BaseObserver:
    """
    观察者的基类，用于观察和收集张量数据的统计信息。

    观察者负责根据指定的统计策略，收集和处理张量数据的统计信息。
    所有具体的观察者实现都应该继承此类并实现抽象方法。
    """

    def __init__(self, strategy: Optional[StatisticsStrategy] = None):
        """
        初始化观察者。

        参数:
            strategy: 统计策略，用于收集和处理统计信息
        """
        self.strategy = strategy

    def set_strategy(self, strategy: StatisticsStrategy):
        """
        设置统计策略。

        参数:
            strategy: 新的统计策略
        """
        self.strategy = strategy

    def update(self, x: torch.Tensor):
        """
        更新观察者的统计信息。

        根据输入张量和内部确定的缩减维度，更新统计策略的状态。

        参数:
            x: 输入张量，包含需要观察的数据
        """
        reduce_dims = self._get_reduce_dims(x)
        self.strategy.update_stats(x, reduce_dims)

    def get_stats(self):
        """
        获取观察者收集的统计信息。

        返回统计策略计算得到的统计结果。

        返回:
            dict: 包含统计指标名称和对应值的字典
        """
        return self.strategy.get_stats()

    def _get_reduce_dims(self, x: torch.Tensor) -> list[int]:
        """
        获取需要缩减的维度列表。

        此方法由子类实现，用于指定在哪些维度上进行统计。

        参数:
            x: 输入张量，用于确定其维度信息

        返回:
            list[int]: 需要缩减的维度列表
        """
        raise NotImplementedError()



OBSERVER_REGISTRY = Registry[BaseObserver]()


class ActivationQuantConfig(BaseModel):
    bits: int = 8
    method: ActivationQuantMethod = ActivationQuantMethod.MINMAX
    scope: ActivationQuantScope = ActivationQuantScope.PER_TENSOR
    symmetric: bool = False
    signed: bool = True


class ActivationQuantizer(nn.Module):
    def __init__(self, config: ActivationQuantConfig):
        super().__init__()
        self.config = config
        self.statistics = None
        self.observer = None

    def set_observer(self, observer: BaseObserver):
        self.observer = observer

    def set_statistics(self, statistics: StatisticsStrategy):
        self.statistics = statistics
        self.observer.set_strategy(self.statistics)

    def get_scale_offset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        min_val, max_val = self.statistics.get_stats()
        scale, zero_point = linear_quantization_params(self.config.bits, min_val, max_val, self.config.signed,
                                                       self.config.symmetric)
        return scale, zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observer.update(x)
        scale, zero_point = self.get_scale_offset()
        _, dequant_x = fake_quantize(x, scale, zero_point, self.config.bits, self.config.signed)
        return dequant_x
