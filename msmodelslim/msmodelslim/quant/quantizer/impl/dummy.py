#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

from typing import Optional

import torch
from pydantic import validate_call

import msmodelslim.quant.ir as qir
from msmodelslim.core import fake_quantize, quantize, dequantize, calculate_qparam
from msmodelslim.core.QAL import QABCRegistry, QDType, QStorage, QParam, QScope, QScheme
from msmodelslim.quant.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.exception import SpecError, SchemaValidateError
from msmodelslim.utils.logging import logger_setter
from ..base import AutoActQuantizer, AutoWeightQuantizer, QConfig


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.fp16_placeholder_sym, "dummy"),
        (qir.fp16_placeholder_asym, "dummy"),
    ],
    abc_type=AutoActQuantizer
)
@logger_setter()
class ActPlaceholderDummy(AutoActQuantizer):

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.q_param = QParam(
        scheme=QScheme(
            dtype=QDType(self.config.dtype),
            scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        ))
        return x

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            return QParam(scheme=self.config.to_scheme())
        return self.q_param

