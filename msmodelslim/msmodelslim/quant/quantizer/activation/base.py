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
from torch import distributed as dist

from msmodelslim.quant.quantizer.base.const import QuantMethod, QuantScope
from msmodelslim.utils.registry import Registry


class ActQuantMethodConfig(BaseModel):
    type: QuantMethod


class ActQuantScopeConfig(BaseModel):
    type: QuantScope


class ActQuantBaseConfig(BaseModel):
    bits: int = 8
    signed: bool = True
    symmetric: bool = False


class ActQuantConfig(BaseModel):
    base: ActQuantBaseConfig
    scope: ActQuantScopeConfig
    method: ActQuantMethodConfig


class ObserverStrategy(ABC):
    """
    统计策略的抽象基类，定义了更新和获取统计值的基本接口。

    统计策略用于收集和处理张量数据的统计信息，如均值、方差等。
    所有具体的统计策略实现都应该继承此类并实现抽象方法。
    """

    @abstractmethod
    def update(self, x: torch.Tensor):
        """
        更新统计值。

        根据输入张量，更新内部统计状态。

        参数:
            x: 输入张量，待统计数据
        """
        pass

    @abstractmethod
    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取最小值和最大值。
        
        返回:
            tuple: 包含最小值和最大值的元组
                min_val: 最小值, 形状为 (1,)
                max_val: 最大值, 形状为 (1,)
        """
        pass

    @abstractmethod
    def reset(self):
        """
        重置统计。
        """
        pass

    def update_with_group(self, x: torch.Tensor, group: dist.ProcessGroup):
        """
        分布式量化时，在指定进程组中更新统计值。

        参数:
            x: 输入张量，待统计数据
            group: 通信组，用于同步更新统计信息的进程组

        返回:
            tuple: 包含最小值和最大值的元组
                min_val: 最小值, 形状为 (1,)
                max_val: 最大值, 形状为 (1,)
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support update_with_group method.")


OBSERVER_STRATEGY_REGISTRY = Registry[ObserverStrategy]()


class BaseObserver:
    """
    观察者的基类，用于观察和收集张量数据的统计信息。

    观察者负责根据指定的统计策略，收集和处理张量数据的统计信息。
    所有具体的观察者实现都应该继承此类并实现抽象方法。
    """

    def __init__(self):
        """
        初始化观察者。

        参数:
            strategy: 统计策略，用于收集和处理统计信息
        """
        self.strategy = None

    def set_strategy(self, strategy: ObserverStrategy):
        """
        设置统计策略。

        参数:
            strategy: 新的统计策略
        """
        self.strategy = strategy

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
        """
        更新观察者的统计信息。

        根据输入张量和内部确定的缩减维度，更新统计策略的状态。

        参数:
            x: 输入张量，包含需要观察的数据
            sync: 是否同步更新统计信息
            group: 通信组，用于同步更新统计信息的进程组
        """

        if sync:
            self.strategy.update_with_group(x, group)
        else:
            self.strategy.update(x)

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取观察者收集的统计信息。

        返回统计策略计算得到的统计结果。

        返回:
            tuple: 包含最小值和最大值的元组
        """
        return self.strategy.get_min_max()


OBSERVER_REGISTRY = Registry[BaseObserver]()
