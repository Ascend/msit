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

from collections import OrderedDict
from typing import Optional, List

import torch
from pydantic import BaseModel, Field
from torch import nn
from torch.nn import functional as F

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.quant.processor.quant.base import LinearQuantProcessor
from msmodelslim.quant.processor.quant.w8a8 import W8A8QuantConfig
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY, PROCESSOR_CONFIG_REGISTRY
from msmodelslim.quant.quantizer.activation.base import ActivationQuantConfig
from msmodelslim.quant.quantizer.base.const import WeightQuantMethod, ActivationQuantScope, ActivationQuantMethod, \
    WeightQuantScope
from msmodelslim.quant.quantizer.base.fake import BaseFakeQuantizer, FAKE_QUANTIZER_REGISTRY
from msmodelslim.quant.quantizer.linear.base import LINEAR_QUANTIZER_REGISTRY, BaseLinearQuantizer, BaseWeightQuantizer
from msmodelslim.quant.quantizer.linear.config import LinearQuantConfig, WeightQuantConfig
from msmodelslim.quant.quantizer.linear.minmax.minmax import MinMaxWeightQuantizer
from msmodelslim.utils.config_map import ConfigMap, ConfigSet


@FAKE_QUANTIZER_REGISTRY.register_by_name('w8a8_dynamic')
class W8A8DynamicLinearFakeQuantizer(BaseFakeQuantizer):
    def __init__(self,
                 cfg: LinearQuantConfig,
                 weight_scale: torch.Tensor,
                 weight_offset: torch.Tensor,
                 weight: torch.Tensor,
                 bias: Optional[torch.Tensor] = None,
                 ):
        super().__init__(cfg)
        self.cfg = cfg
        self.weight_scale = nn.Parameter(weight_scale.to(torch.float32), requires_grad=False)
        self.weight_offset = nn.Parameter(weight_offset.to(torch.float32), requires_grad=False)
        self.weight = nn.Parameter(weight.to(torch.int8), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


@LINEAR_QUANTIZER_REGISTRY.register()
class W8A8DynamicLinearQuantizer(BaseLinearQuantizer):

    def __init__(self, module: nn.Linear, cfg: LinearQuantConfig):
        super().__init__(module, cfg)
        self.forward_called = False
        self.register_buffer('dequant_weight', None)
        self.register_buffer('dequant_bias', None)

    @staticmethod
    def match(module: nn.Module, cfg: LinearQuantConfig) -> bool:
        return cfg.a_cfg.bits == 8 and cfg.w_cfg.bits == 8 and cfg.a_cfg.scope == ActivationQuantScope.PER_TOKEN

    def deploy(self, *args, **kwargs) -> BaseFakeQuantizer:
        with torch.device(self.fp_weight.device):
            if not self.forward_called:
                self.weight_quantizer.forward(self.fp_weight, self.fp_bias)
            quant_weight, bias = self.weight_quantizer.quant(self.fp_weight, self.fp_bias)
            weight_scale, weight_offset = self.weight_quantizer.get_scale_offset()
        return W8A8DynamicLinearFakeQuantizer(self.cfg, weight_scale, weight_offset, quant_weight, bias)

    def _create_weight_quantizer(self, cfg: WeightQuantConfig) -> BaseWeightQuantizer:
        if cfg.method == WeightQuantMethod.MINMAX:
            return MinMaxWeightQuantizer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() > 0:
            x = self.input_quantizer(x)
        if not self.forward_called:
            self.dequant_weight, self.dequant_bias = self.weight_quantizer.forward(self.fp_weight, self.fp_bias)
            self.forward_called = True
        return F.linear(x, self.dequant_weight, self.dequant_bias)


class W8A8DynamicQuantConfig(W8A8QuantConfig):
    a_cfg: ActivationQuantConfig = Field(default=ActivationQuantConfig(bits=8,
                                                                       method=ActivationQuantMethod.MINMAX,
                                                                       scope=ActivationQuantScope.PER_TOKEN))
    w_cfg: WeightQuantConfig = Field(default=WeightQuantConfig(bits=8,
                                                               method=WeightQuantMethod.MINMAX,
                                                               scope=WeightQuantScope.PER_CHANNEL))

    quant_type: QuantType = Field(default=QuantType.W8A8_DYNAMIC)


@PROCESSOR_CONFIG_REGISTRY.register_by_name("w8a8_dynamic")
class W8A8DynamicProcessorConfig(BaseModel):
    disable_names: List[str] = Field(default=[])
    cfg_map: OrderedDict[str, W8A8DynamicQuantConfig] = Field(default=OrderedDict())


@PROCESSOR_REGISTRY.register_by_name("w8a8_dynamic")
class W8A8DynamicProcessor(LinearQuantProcessor):

    def __init__(self, model: nn.Module, cfg: W8A8DynamicProcessorConfig, **kwargs):
        cfg.model_validate(cfg)
        self.cfg_manager = ConfigMap[W8A8DynamicQuantConfig](cfg.cfg_map)
        self.disable_set = ConfigSet[str](cfg.disable_names)
        super().__init__(model, self.cfg_manager, self.disable_set)

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True
