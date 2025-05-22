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

import torch

from msmodelslim.quant.quantizer.activation.base import BaseObserver
from msmodelslim.quant.quantizer.activation.base import OBSERVER_REGISTRY
from msmodelslim.quant.quantizer.base.const import ActivationQuantScope


# 按Tensor统计（全局）
@OBSERVER_REGISTRY.register(ActivationQuantScope.PER_TENSOR.value)
class PerTensorObserver(BaseObserver):
    """
    按张量统计的观察者，对所有维度进行统计。
    
    该类观察者会对输入张量的所有维度进行统计，得到一个全局的统计结果。
    适用于需要对整个张量进行统一统计的场景。
    """

    def __init__(self):
        super().__init__(sync_stats=True)

    def _get_reduce_dims(self, x: torch.Tensor) -> list[int]:
        """
        获取需要缩减的维度列表。
        
        对于PerTensorObserver，缩减所有维度，得到一个标量统计结果。
        
        参数:
            x: 输入张量，用于确定其维度信息
            
        返回:
            list[int]: 包含所有维度的列表，表示在所有维度上进行统计
        """
        return list(range(x.ndim))  # 缩减所有维度


# 按轴统计（如Channel/Head）
class PerAxisObserver(BaseObserver):
    """
    按轴统计的观察者，对指定轴以外的维度进行统计。
    
    该类观察者会对输入张量的指定轴以外的维度进行统计，得到一个按轴分布的统计结果。
    适用于需要对张量的特定维度进行独立统计的场景。
    """

    def __init__(self, axis: int):
        """
        初始化按轴统计的观察者。
        
        参数:
            strategy: 统计策略，用于收集和处理统计信息
            axis: 需要保留的轴，其他轴将被缩减
        """
        super().__init__()
        self.axis = axis

    def _get_reduce_dims(self, x: torch.Tensor) -> list[int]:
        """
        获取需要缩减的维度列表。
        
        对于PerAxisObserver，缩减除了指定轴以外的所有维度。
        
        参数:
            x: 输入张量，用于确定其维度信息
            
        返回:
            list[int]: 除了指定轴以外的所有维度的列表
        """
        return [d for d in range(x.ndim) if d != self.axis]


@OBSERVER_REGISTRY.register(ActivationQuantScope.PER_HEAD.value)
class PerHeadObserver(PerAxisObserver):
    """
    按注意力头统计的观察者，对除了头维度以外的维度进行统计。
    
    该类观察者会对输入张量的头维度以外的维度进行统计，得到一个按头分布的统计结果。
    适用于需要对注意力机制中的头进行独立统计的场景。
    """

    def __init__(self, axis: int = 1):
        """
        初始化按头统计的观察者。
        
        参数:
            strategy: 统计策略，用于收集和处理统计信息
            axis: 头维度，默认为1
        """
        super().__init__(axis)


@OBSERVER_REGISTRY.register(ActivationQuantScope.PER_TOKEN.value)
class PerTokenObserver(PerTensorObserver):
    """
    按Token统计的观察者，对每个Token进行统计。
    
    该类观察者会对输入张量的每个Token进行统计，得到一个按Token分布的统计结果。
    适用于需要对每个Token进行独立统计的场景。
    """

    def __init__(self):
        super().__init__()
        self.sync_stats = False

    def update(self, x: torch.Tensor):
        """
        更新观察者的统计信息。

        根据输入张量和内部确定的缩减维度，更新统计策略的状态。

        参数:
            x: 输入张量，包含需要观察的数据
        """
        self.strategy.clear_stats()
        super().update(x)
