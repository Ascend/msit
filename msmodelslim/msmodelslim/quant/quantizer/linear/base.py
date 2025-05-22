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

import abc
from typing import Tuple, List, Type, Optional, Set

import torch
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F

from msmodelslim.quant.quantizer.activation.base import ActivationQuantConfig, ActivationQuantizer
from msmodelslim.quant.quantizer.activation.factory import ActivationQuantizerFactory
from msmodelslim.quant.quantizer.base.fake import BaseFakeQuantizer
from msmodelslim.quant.quantizer.base.quantizer import BaseQuantizer
from msmodelslim.quant.quantizer.linear.config import WeightQuantConfig, LinearQuantConfig


class BaseWeightQuantizer(nn.Module):
    """
    权重量化器的基类
    
    该类定义了权重量化的基本接口，包括量化方法和前向传播方法。
    所有具体的权重量化器都应该继承这个类并实现其抽象方法。
    """

    def __init__(self, cfg: WeightQuantConfig):
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def quant(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        量化权重

        参数:
            weight: 权重张量
            bias: 偏置张量
            x: 输入张量

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 量化后的权重和偏置
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        量化权重并执行前向传播

        参数:
            weight: 权重张量
            bias: 偏置张量
            x: 输入张量

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 量化然后反量化后的权重和偏置
        """
        raise NotImplementedError()


class BaseLinearQuantizer(BaseQuantizer):
    """
    线性层量化器的基类
    
    该类通过替换 nn.Linear 实现量化功能。在模型的前向传播过程中：
    1. 首先使用 input_quantizer 对输入进行量化和反量化
    2. 然后使用 weight_quantizer 对权重进行量化和反量化
    3. 最后使用量化后的权重和偏置进行线性运算
    
    尽管它持有一个 nn.Linear 实例，以及 input_quantizer 和 weight_quantizer，
    但它仍被视为模型树中的叶节点，为此我们重写了 named_modules 方法。
    """

    def __init__(self, module: nn.Linear, cfg: LinearQuantConfig):
        super().__init__(module, cfg)
        self.cfg = cfg

    def named_modules(self, memo: Optional[Set['nn.Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        """
        重写named_modules方法，使其作为叶节点
        
        参数:
            memo: 用于存储已访问模块的集合
            prefix: 模块名称前缀
            remove_duplicate: 是否移除重复项
        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self

    @abc.abstractmethod
    def deploy(self, *args, **kwargs) -> BaseFakeQuantizer:
        """
        部署量化器
        
        返回:
            BaseFakeQuantizer: 部署后的伪量化器
        """
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        参数:
            x: 输入张量
            
        返回:
            torch.Tensor: 量化后的输出张量
        """
        if x.numel() > 0:
            x = self.input_quantizer(x)
        weight, bias = self.weight_quantizer(self.fp_weight, self.fp_bias, x)
        return F.linear(x, weight, bias)

    def _setup(self, module: nn.Module, cfg: LinearQuantConfig, *args, **kwargs):
        """
        设置量化器
        
        参数:
            module: 要量化的模块
            cfg: 量化配置
        """
        super()._setup(module, cfg, *args, **kwargs)
        self.fp_weight = module.weight
        self.fp_bias = getattr(module, 'bias', None)
        self.input_quantizer = self._create_input_quantizer(cfg.a_cfg)
        self.weight_quantizer = self._create_weight_quantizer(cfg.w_cfg)

    def _create_input_quantizer(self, cfg: ActivationQuantConfig) -> ActivationQuantizer:
        """
        创建输入量化器
        
        参数:
            cfg: 激活量化配置
            
        返回:
            ActivationQuantizer: 创建的输入量化器
        """
        _ = self
        return ActivationQuantizerFactory.create(cfg)

    @abc.abstractmethod
    def _create_weight_quantizer(self, cfg: WeightQuantConfig) -> BaseWeightQuantizer:
        """
        创建权重量化器
        
        参数:
            cfg: 权重量化配置
            
        返回:
            BaseWeightQuantizer: 创建的权重量化器
        """
        raise NotImplementedError()


class QuantizerRegistry:
    """
    量化器注册表
    
    该类提供了一个中央注册表，用于存储和管理不同类型的量化器。
    支持通过装饰器方式注册量化器，并提供查询和获取量化器的功能。
    量化器通过match方法判断是否适用于给定的配置。
    """

    def __init__(self):
        # 存储所有注册的量化器
        self._quantizers: List[Type[BaseLinearQuantizer]] = []

    def register(self):
        """
        量化器注册装饰器
        
        该装饰器用于注册量化器类。
        
        返回:
            装饰器函数，用于注册量化器类
        """

        def decorator(quantizer_cls: Type[BaseLinearQuantizer]) -> Type[BaseLinearQuantizer]:
            # 注册量化器
            self._quantizers.append(quantizer_cls)
            return quantizer_cls

        return decorator

    def get_quantizer(self, module: nn.Module, config: BaseModel) -> Optional[Type[BaseLinearQuantizer]]:
        """
        根据配置获取量化器类
        
        该方法会遍历所有注册的量化器，调用其match方法判断是否适用于给定的配置。
        返回第一个匹配的量化器类。
        
        参数:
            module: 要量化的模块
            config: 量化配置参数
            
        返回:
            Type[BaseLinearQuantizer]: 量化器类，如果不存在则返回None
        """
        for quantizer_cls in self._quantizers:
            if hasattr(quantizer_cls, 'match') and quantizer_cls.match(module, config):
                return quantizer_cls
        return None

    def get_all_quantizers(self) -> List[Type[BaseLinearQuantizer]]:
        """
        获取所有注册的量化器
        
        返回:
            List[Type[BaseLinearQuantizer]]: 量化器类列表
        """
        return self._quantizers.copy()

    def clear(self):
        """
        清空注册表
        
        该方法会清空所有注册的量化器。
        """
        self._quantizers.clear()


# 创建全局量化器注册表实例
LINEAR_QUANTIZER_REGISTRY = QuantizerRegistry()
