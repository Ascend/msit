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

import torch
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.quant.processor.quant.base import LinearQuantProcessor, BaseSessionQuantConfig
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY, PROCESSOR_CONFIG_REGISTRY
from msmodelslim.quant.quantizer.base.const import WeightQuantMethod
from msmodelslim.quant.quantizer.base.fake import BaseFakeQuantizer, FAKE_QUANTIZER_REGISTRY
from msmodelslim.quant.quantizer.linear.base import BaseLinearQuantizer, BaseWeightQuantizer, LINEAR_QUANTIZER_REGISTRY
from msmodelslim.quant.quantizer.linear.config import LinearQuantConfig, WeightQuantConfig
from msmodelslim.quant.quantizer.linear.minmax.minmax import MinMaxWeightQuantizer
from msmodelslim.quant.quantizer.base.const import ActivationQuantScope
from msmodelslim.utils.config_map import ConfigMap


@FAKE_QUANTIZER_REGISTRY.register_by_name('w8a8')
class W8A8LinearFakeQuantizer(BaseFakeQuantizer):
    def __init__(self,
                 cfg: LinearQuantConfig,
                 input_scale: torch.Tensor,
                 input_offset: torch.Tensor,
                 deq_scale: torch.Tensor,
                 quant_bias: torch.Tensor,
                 weight: torch.Tensor,
                 ):
        super().__init__(cfg)
        self.cfg = cfg
        self.weight = nn.Parameter(weight.to(torch.int8), requires_grad=False)
        self.input_scale = nn.Parameter(input_scale.to(torch.float32), requires_grad=False)
        self.input_offset = nn.Parameter(input_offset.to(torch.float32), requires_grad=False)
        self.deq_scale = nn.Parameter(deq_scale.to(torch.float32), requires_grad=False)
        self.quant_bias = nn.Parameter(quant_bias.to(torch.int32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.input_scale + self.input_offset).round().to(torch.int8)
        x = F.linear(x, self.weight)
        x = (x + self.quant_bias) * self.deq_scale
        return x


@LINEAR_QUANTIZER_REGISTRY.register()
class W8A8LinearQuantizer(BaseLinearQuantizer):

    def __init__(self, module: nn.Linear, cfg: LinearQuantConfig):
        super().__init__(module, cfg)

    @staticmethod
    def match(module: nn.Module, cfg: LinearQuantConfig) -> bool:
        return cfg.a_cfg.bits == 8 and cfg.w_cfg.bits == 8 and cfg.a_cfg.scope != ActivationQuantScope.PER_TOKEN

    def deploy(self, *args, **kwargs) -> BaseFakeQuantizer:
        input_scale, input_offset = self.input_quantizer.get_scale_offset()
        weight_scale, _ = self.weight_quantizer.get_scale_offset()
        quant_weight, _ = self.weight_quantizer.quant(self.fp_weight, self.fp_bias)
        deq_scale = input_scale * weight_scale
        fp_weight_bias = self.fp_bias if self.fp_bias is not None else torch.zeros_like(self.fp_weight)
        correction = quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)
        quant_bias = torch.round(fp_weight_bias / deq_scale - correction).to(torch.int32)
        return W8A8LinearFakeQuantizer(self.cfg, input_scale, input_offset, deq_scale, quant_bias, quant_weight)

    def _create_weight_quantizer(self, cfg: WeightQuantConfig) -> BaseWeightQuantizer:
        if cfg.method == WeightQuantMethod.MINMAX:
            return MinMaxWeightQuantizer(cfg)


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
