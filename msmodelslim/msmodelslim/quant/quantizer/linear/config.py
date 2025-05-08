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

from msmodelslim.quant.quantizer.activation.base import ActivationQuantConfig
from msmodelslim.quant.quantizer.base.const import WeightQuantMethod, WeightQuantScope


class WeightQuantConfig(BaseModel):
    bits: int = 8
    method: WeightQuantMethod = WeightQuantMethod.MINMAX
    scope: WeightQuantScope = WeightQuantScope.PER_CHANNEL
    group_size: int = -1
    symmetric: bool = True
    signed: bool = True


class LinearQuantConfig(BaseModel):
    a_cfg: ActivationQuantConfig
    w_cfg: WeightQuantConfig
