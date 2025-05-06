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

from pydantic import BaseModel
from torch import nn


class BaseQuantizer(nn.Module):
    """
    量化器的基类，提供量化功能的基础接口和通用实现。
    
    量化器用于将模型中的浮点参数转换为低精度表示，以减少模型大小和计算开销。
    所有具体的量化器实现都应该继承此类并实现抽象方法。
    """

    def __init__(self, module: nn.Module, cfg: BaseModel, *args, **kwargs):
        """
        初始化量化器。
        
        参数:
            *args: 传递给父类的可变位置参数
            **kwargs: 传递给父类的可变关键字参数
        """
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self._setup(module, cfg)

    @staticmethod
    def match(module: nn.Module, cfg: BaseModel) -> bool:
        """
        判断模块是否匹配量化配置。
        
        此方法用于判断模块是否符合量化配置的要求。
        
        """
        pass

    @abc.abstractmethod
    def deploy(self, *args, **kwargs) -> nn.Module:
        """
        部署量化器，生成用于推理的伪量化模块。
        
        此方法将量化器转换为可直接用于推理的伪量化模块。
        部署后，量化器不再进行量化校准，而是直接执行伪量化推理。
        
        返回:
            nn.Module: 用于替换原始模块的伪量化模块
        """
        pass

    def _setup(self, module: nn.Module, cfg: BaseModel, *args, **kwargs):
        """
        设置量化器参数和状态。

        在安装量化器时，此方法用于初始化量化器，设置必要的参数和状态。
        子类应重写此方法以实现具体的设置逻辑。

        参数:
            module: 被替换的原始模块
            cfg: 量化配置
            *args: 可变位置参数
            **kwargs: 可变关键字参数
        """
        pass
