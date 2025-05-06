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

from typing import Type

from torch import nn

from msmodelslim import logger
from msmodelslim.quant.model.base import BaseModelAdapter
from msmodelslim.utils.registry import Registry

MODEL_ADAPTER_REGISTRY = Registry[Type[BaseModelAdapter]]()


def create_model_adapter(model: nn.Module) -> BaseModelAdapter:
    """
    创建模型适配器实例。

    该方法会根据模型中的config.model_type或模型的类型或名称，创建适用于该模型的适配器实例。
    搜索顺序：1. model.config.model_type 2. 模型类型 3. 模型类名

    参数:
        model: 模型实例

    返回:
        BaseModelAdapter: 适用于该模型的适配器实例

    异常:
        ValueError: 如果没有找到适用于该模型的适配器
    """
    # 1. 首先尝试使用model.config.model_type
    if hasattr(model, 'config') and hasattr(model.config, 'model_type') and model.config.model_type is not None:
        adapter_cls = MODEL_ADAPTER_REGISTRY.get_by_name(model.config.model_type)
        if adapter_cls is not None:
            return adapter_cls(model)

    # 2. 然后尝试使用模型类型
    model_type = type(model)
    adapter_cls = MODEL_ADAPTER_REGISTRY.get_by_type(model_type)
    if adapter_cls is not None:
        return adapter_cls(model)

    # 3. 最后尝试使用模型类名
    model_name = model_type.__name__
    adapter_cls = MODEL_ADAPTER_REGISTRY.get_by_name(model_type.__name__)
    if adapter_cls is not None:
        return adapter_cls(model)

    # 如果都找不到，则返回默认适配器
    logger.warning(
        f"Can't find adapter for model type {type(model)} or name {type(model).__name__}, use default instance instead")

    return BaseModelAdapter(model)
