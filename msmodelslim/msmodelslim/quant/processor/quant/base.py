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

from pydantic import BaseModel
from torch import nn

from msmodelslim import logger
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.utils.config_map import ConfigMap
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.quant.processor.base import SessionBaseProcessor
from msmodelslim.quant.processor.const import ProcessStage
from msmodelslim.quant.quantizer.linear.base import LINEAR_QUANTIZER_REGISTRY
from msmodelslim.quant.quantizer.linear.config import LinearQuantConfig


class LinearQuantProcessor(SessionBaseProcessor):
    def __init__(self, model: nn.Module, cfg_map: ConfigMap[LinearQuantConfig]):
        super().__init__(model)
        self.cfg_map = cfg_map

    def is_data_free(self) -> bool:
        return False

    def stage(self) -> ProcessStage:
        return ProcessStage.FORWARD_QUANT

    def preprocess(self, request: BatchProcessRequest) -> None:
        self._install_quantizer(request.name, request.module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._deploy(request.name, request.module)

    def _install_quantizer(self, prefix: str, module: nn.Module) -> None:
        for name, module in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix != "" else name
            if isinstance(module, nn.Linear):
                self._process_linear(full_name, module)

    def _deploy(self, prefix: str, module: nn.Module) -> None:
        for name, module in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix != "" else name
            if hasattr(module, "deploy"):
                self.model.set_submodule(full_name, module.deploy())

    def _process_linear(self, full_name: str, module: nn.Linear) -> None:
        if full_name in self.cfg_map:
            logger.info(f"Linear layer {full_name} with use config {self.cfg_map[full_name]}")
            quantizer_cls = LINEAR_QUANTIZER_REGISTRY.get_quantizer(module, self.cfg_map[full_name])

            if quantizer_cls is None:
                raise ValueError(
                    f"No quantizer found for linear layer {full_name} with config {self.cfg_map[full_name]}")

            quantizer = quantizer_cls(module, self.cfg_map[full_name])
            self.model.set_submodule(full_name, quantizer)


class BaseSessionQuantConfig(BaseModel):

    def quant_type(self) -> QuantType:
        return QuantType.UNKNOWN


class FloatQuantConfig(BaseSessionQuantConfig):
    def quant_type(self) -> QuantType:
        return QuantType.FLOAT
