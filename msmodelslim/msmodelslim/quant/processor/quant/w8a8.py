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
from msmodelslim.quant.processor.quant.base import LinearQuantProcessor, BaseSessionQuantConfig
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY, PROCESSOR_CONFIG_REGISTRY
from msmodelslim.quant.quantizer.linear.config import LinearQuantConfig
from msmodelslim.utils.config_map import ConfigMap


class W8A8QuantConfig(BaseSessionQuantConfig, LinearQuantConfig):
    def quant_type(self) -> QuantType:
        return QuantType.W8A8


@PROCESSOR_CONFIG_REGISTRY.register_by_name("w8a8")
class W8A8ProcessorConfig(BaseModel):
    cfg_map: Dict[str, W8A8QuantConfig]


@PROCESSOR_REGISTRY.register_by_name("w8a8")
class W8A8Processor(LinearQuantProcessor):

    def __init__(self, model: nn.Module, cfg: W8A8ProcessorConfig):
        cfg.model_validate(cfg)
        self.cfg_manager = ConfigMap[W8A8QuantConfig](cfg.cfg_map)
        super().__init__(model, self.cfg_manager)

    def is_data_free(self) -> bool:
        return False
