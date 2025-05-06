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

from typing import Dict

from pydantic import BaseModel
from torch import nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.quant.processor.quant.base import LinearQuantProcessor
from msmodelslim.quant.processor.quant.w8a8 import W8A8QuantConfig
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY, PROCESSOR_CONFIG_REGISTRY


class W8A8DynamicQuantConfig(W8A8QuantConfig):
    def quant_type(self) -> QuantType:
        return QuantType.W8A8_DYNAMIC


@PROCESSOR_CONFIG_REGISTRY.register_by_name("w8a8_dynamic")
class W8A8DynamicProcessorConfig(BaseModel):
    cfg_map: Dict[str, W8A8DynamicQuantConfig] = {}


@PROCESSOR_REGISTRY.register_by_name("w8a8_dynamic")
class W8A8DynamicProcessor(LinearQuantProcessor):

    def __init__(self, model: nn.Module, cfg: W8A8DynamicProcessorConfig):
        super().__init__(model, cfg.cfg_map)

    def is_data_free(self) -> bool:
        return False
