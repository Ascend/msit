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
from typing import Tuple, List, Type, Optional

import torch
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F

from msmodelslim.quant.quantizer.activation.base import ActivationQuantConfig, ActivationQuantizer
from msmodelslim.quant.quantizer.activation.factory import ActivationQuantizerFactory
from msmodelslim.quant.quantizer.base.const import ActivationQuantScope
from msmodelslim.quant.quantizer.base.fake import BaseFakeQuantizer
from msmodelslim.quant.quantizer.base.quantizer import BaseQuantizer
from msmodelslim.quant.quantizer.linear.config import WeightQuantConfig, LinearQuantConfig
from msmodelslim.quant.quantizer.linear.fake import W8A8LinearFakeQuantizer


class BaseWeightQuantizer(nn.Module):
    def __init__(self, cfg: WeightQuantConfig):
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def quant(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> \
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
        量化权重

        参数:
            weight: 权重张量
            bias: 偏置张量
            x: 输入张量

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 量化然后反量化后的权重和偏置
        """
        raise NotImplementedError()



class BaseLinearQuantizer(BaseQuantizer):

    def __init__(self, module: nn.Module, cfg: LinearQuantConfig):
        super().__init__(module, cfg)
        self.cfg = cfg
        self.module = module
        self.input_quantizer = self._create_input_quantizer(cfg.a_cfg)
        self.weight_quantizer = self._create_weight_quantizer(cfg.w_cfg)

    def deploy(self, *args, **kwargs) -> BaseFakeQuantizer:
        if (self.cfg.w_cfg.bits == 8 and self.cfg.a_cfg.bits == 8
                and self.cfg.a_cfg.method != ActivationQuantScope.PER_TOKEN):
            return self._deploy_w8a8()
        elif (self.cfg.w_cfg.bits == 8 and self.cfg.a_cfg.bits == 8
              and self.cfg.a_cfg.method == ActivationQuantScope.PER_TOKEN):
            return self._deploy_w8a8_dynamic()
        elif self.cfg.a_cfg.bits == 16:
            return self._deploy_wxa16()
        else:
            raise ValueError(f"Unsupported weight and activation bits: {self.cfg.w_cfg.bits} and {self.cfg.a_cfg.bits}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quantizer(x)
        weight, bias = self.weight_quantizer(x)
        return F.linear(x, weight, bias)

    @abc.abstractmethod
    def _create_weight_quantizer(self, cfg: WeightQuantConfig) -> BaseWeightQuantizer:
        pass

    def _setup(self, module: nn.Module, cfg: LinearQuantConfig, *args, **kwargs):
        super()._setup(module, cfg, *args, **kwargs)
        self.fp_weight = module.weight
        self.fp_bias = getattr(module, 'bias', None)
        self.input_quantizer = self._create_input_quantizer(cfg.a_cfg)
        self.weight_quantizer = self._create_weight_quantizer(cfg.w_cfg)

    def _deploy_w8a8(self) -> BaseFakeQuantizer:
        input_scale, input_offset = self.input_quantizer.get_scale_offset()
        weight_scale, _ = self.weight_quantizer.get_scale_offset()
        quant_weight, _ = self.weight_quantizer.quant(self.fp_weight, self.fp_bias)
        deq_scale = input_scale * weight_scale
        fp_weight_bias = self.fp_bias if self.fp_bias is not None else torch.zeros_like(self.fp_weight)
        correction = quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)
        quant_bias = torch.round(fp_weight_bias / deq_scale - correction).to(torch.int32)
        return W8A8LinearFakeQuantizer(self.cfg, input_scale, input_offset, deq_scale, quant_bias, quant_weight)

    def _deploy_w8a8_dynamic(self) -> BaseFakeQuantizer:
        raise NotImplementedError()


    def _deploy_wxa16(self) -> BaseFakeQuantizer:
        raise NotImplementedError()


    def _create_input_quantizer(self, cfg: ActivationQuantConfig) -> ActivationQuantizer:
        _ = self
        return ActivationQuantizerFactory.create(cfg)


class QuantizerRegistry:
    """
    量化器注册表，用于管理和注册量化器。

    该类提供了一个中央注册表，用于存储和管理不同类型的量化器。
    支持通过装饰器方式注册量化器，并提供查询和获取量化器的功能。
    量化器通过match方法判断是否适用于给定的配置。
    """

    def __init__(self):
        # 存储所有注册的量化器
        self._quantizers: List[Type[BaseLinearQuantizer]] = []

    def register(self):
        """
        量化器注册装饰器。

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
        根据配置获取量化器类。

        该方法会遍历所有注册的量化器，调用其match方法判断是否适用于给定的配置。
        返回第一个匹配的量化器类。

        参数:
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
        获取所有注册的量化器。

        返回:
            List[Type[BaseLinearQuantizer]]: 量化器类列表
        """
        return self._quantizers.copy()

    def clear(self):
        """
        清空注册表。

        该方法会清空所有注册的量化器。
        """
        self._quantizers.clear()


LINEAR_QUANTIZER_REGISTRY = QuantizerRegistry()
