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
from typing import Optional, List, Tuple, Union 

import torch
from pydantic import BaseModel, Field
from torch import nn
from torch.nn import functional as F

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.quant.processor.quant.base import LinearQuantProcessor, BaseSessionQuantConfig
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY, PROCESSOR_CONFIG_REGISTRY
from msmodelslim.quant.quantizer.activation.base import ActQuantConfig, ActQuantBaseConfig
from msmodelslim.quant.quantizer.activation.minmax import ActMinMaxConfig
from msmodelslim.quant.quantizer.activation.observer import PerTokenConfig
from msmodelslim.quant.quantizer.base.const import QuantScope, QuantMethod
from msmodelslim.quant.quantizer.linear.config import WeightQuantBaseConfig, \
    WeightQuantMethodConfig, WeightQuantScopeConfig
from msmodelslim.quant.quantizer.base.fake import BaseFakeQuantizer, FAKE_QUANTIZER_REGISTRY
from msmodelslim.quant.quantizer.linear.base import LINEAR_QUANTIZER_REGISTRY, BaseLinearQuantizer, BaseWeightQuantizer
from msmodelslim.quant.quantizer.linear.config import LinearQuantConfig, WeightQuantConfig
from msmodelslim.quant.quantizer.linear.minmax.minmax import MinMaxWeightQuantizer
from msmodelslim.utils.config_map import ConfigMap, ConfigSet
from msmodelslim.quant.kia.utils import handle_progressive_quant


class W4A8WeightGroupStageQuantizer(MinMaxWeightQuantizer):
    def __init__(self, cfg: WeightQuantConfig):
        super().__init__(cfg)
        self.register_buffer('weight_scale_second', None)
        self.register_buffer('weight_offset_second', None)

    
    def get_scale_offset(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self._check_scale_offset()
        return [self.weight_scale, self.weight_scale_second], [self.weight_offset, self.weight_offset_second]


    def quant(self,
              weight: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_scale_offset()
        _, _, quant_weight, _ = handle_progressive_quant(weight=weight,
                                                         group_size=self.cfg.scope.group_size,
                                                         num_bits=self.cfg.base.bits,
                                                         per_channel=not (
                                                             self.cfg.scope.type == QuantScope.PER_CHANNEL
                                                         ),
                                                         w_sym=self.cfg.base.symmetric,
                                                         use_hqq=self.cfg.method.type == QuantMethod.HQQ)
        return quant_weight, bias
    

    def forward(
            self,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dequant_weight, scales, quant_weight, offsets = \
            handle_progressive_quant(weight=weight,
                                    group_size=self.cfg.scope.group_size,
                                    num_bits=self.cfg.base.bits,
                                    per_channel=not (
                                        self.cfg.scope.type == QuantScope.PER_TENSOR
                                    ),
                                    w_sym=self.cfg.base.symmetric,
                                    use_hqq=self.cfg.method.type == QuantMethod.HQQ)
        self.weight_scale = scales[0]
        self.weight_offset = offsets[0]
        self.weight_scale_second = scales[1]
        self.weight_offset_second = offsets[1]

        return dequant_weight, bias

    def _check_scale_offset(self):
        is_none = self.weight_scale is None or self.weight_offset is None
        is_none_second = self.weight_scale_second is None or self.weight_offset_second is None
        if is_none or is_none_second:
            raise RuntimeError("Weight scale/scale_second and weight offset/offset_second "
                               "must be initialized before getting scale/offset")


@FAKE_QUANTIZER_REGISTRY.register_by_name('w4a8_dynamic')
class W4A8DynamicLinearFakeQuantizer(BaseFakeQuantizer):
    def __init__(self,
                 cfg: LinearQuantConfig,
                 weight: torch.Tensor,
                 weight_scale: Union[torch.Tensor, List[torch.Tensor]],
                 weight_offset: Union[torch.Tensor, List[torch.Tensor]],
                 bias: Optional[torch.Tensor] = None,
                 ):
        super().__init__(cfg)
        self.cfg = cfg

        if isinstance(weight_scale, list) and isinstance(weight_offset, list):
            self.weight_scale = nn.Parameter(weight_scale[0].to(torch.float32), requires_grad=False)
            self.weight_offset = nn.Parameter(weight_offset[0].to(torch.float32), requires_grad=False)
            self.weight_scale_second = nn.Parameter(weight_scale[1].to(torch.float32), requires_grad=False)
            self.weight_offset_second = nn.Parameter(weight_offset[1].to(torch.float32), requires_grad=False)
        else:
            self.weight_scale = nn.Parameter(weight_scale.to(torch.float32), requires_grad=False)
            self.weight_offset = nn.Parameter(weight_offset.to(torch.float32), requires_grad=False)
            self.weight_scale_second = None
            self.weight_offset_second = None

        self.weight = nn.Parameter(weight.to(torch.int8), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32), requires_grad=False)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    

@LINEAR_QUANTIZER_REGISTRY.register()
class W4A8DynamicLinearQuantizer(BaseLinearQuantizer):
    
    def __init__(self, module: nn.Linear, cfg: LinearQuantConfig):
        super().__init__(module, cfg)
        self.forward_called = False
        self.register_buffer('dequant_weight', None)
        self.register_buffer('dequant_bias', None)
    

    @staticmethod
    def match(module: nn.Module, cfg: LinearQuantConfig) -> bool:
        return (cfg.a_cfg.base.bits == 8 and cfg.w_cfg.base.bits == 4 and 
                cfg.a_cfg.scope.type == QuantScope.PER_TOKEN and 
                cfg.w_cfg.scope.type in [QuantScope.PER_GROUP] and
                cfg.is_stage_quant)
    

    def deploy(self, *args, **kwargs) -> BaseFakeQuantizer:
        with torch.device(self.fp_weight.device):
            if not self.forward_called:
                self.weight_quantizer.forward(self.fp_weight, self.fp_bias)
            quant_weight, bias = self.weight_quantizer.quant(self.fp_weight, self.fp_bias)
            weight_scale, weight_offset = self.weight_quantizer.get_scale_offset()
        return W4A8DynamicLinearFakeQuantizer(cfg=self.cfg,
                                              weight=quant_weight,
                                              weight_scale=weight_scale, 
                                              weight_offset=weight_offset, 
                                              bias=bias)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() > 0:
            x = self.input_quantizer(x)
        if not self.forward_called:
            self.dequant_weight, self.dequant_bias = self.weight_quantizer.forward(self.module.weight, self.module.bias)
            self.forward_called = True
        return F.linear(x, self.dequant_weight, self.dequant_bias)
    

    def _create_weight_quantizer(self, cfg: WeightQuantConfig) -> BaseWeightQuantizer:
        return W4A8WeightGroupStageQuantizer(cfg)


class WeightQuantScopePerGroupConfig(WeightQuantScopeConfig):
    type: QuantScope = QuantScope.PER_GROUP
    group_size: int = 256


class W4A8DynamicQuantConfig(BaseSessionQuantConfig, LinearQuantConfig):
    quant_type: QuantType = Field(
        default=QuantType.W4A8_DYNAMIC
    )

    a_cfg: ActQuantConfig = Field(
        default=ActQuantConfig(
            base=ActQuantBaseConfig(),
            method=ActMinMaxConfig(),
            scope=PerTokenConfig(type=QuantScope.PER_TOKEN)
        )
    )
    w_cfg: WeightQuantConfig = Field(
        default=WeightQuantConfig(
            base=WeightQuantBaseConfig(bits=4),
            method=WeightQuantMethodConfig(type=QuantMethod.MINMAX),
            scope=WeightQuantScopePerGroupConfig(type=QuantScope.PER_GROUP)
        )
    )

    is_stage_quant: bool = Field(default=True)


@PROCESSOR_CONFIG_REGISTRY.register_by_name("w4a8_dynamic")
class W4A8DynamicProcessorConfig(BaseModel):
    disable_names: List[str] = Field(default=[])
    cfg_map: OrderedDict[str, W4A8DynamicQuantConfig] = Field(default=OrderedDict())


@PROCESSOR_REGISTRY.register_by_name("w4a8_dynamic")
class W4A8DynamicProcessor(LinearQuantProcessor):

    def __init__(self, model: nn.Module, cfg: W4A8DynamicProcessorConfig, **kwargs):
        cfg.model_validate(cfg)
        self.cfg_manager = ConfigMap[W4A8DynamicQuantConfig](cfg.cfg_map)
        self.disable_set = ConfigSet[str](cfg.disable_names)
        super().__init__(model, self.cfg_manager, self.disable_set)

    
    def support_distributed(self) -> bool:
        return True
    

    def is_data_free(self) -> bool:
        return True