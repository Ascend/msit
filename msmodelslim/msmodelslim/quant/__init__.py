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

__all__ = [
    "quant_model",
    "SessionConfig",
    "FloatQuantConfig",
    "W8A8ProcessorConfig",
    "W8A8QuantConfig",
    "W8A8DynamicProcessorConfig",
    "W8A8DynamicQuantConfig",
    "SaverProcessorConfig",
    "M1ProcessorConfig"
]

from msmodelslim.quant.processor.anti_outlier.m1 import M1ProcessorConfig
from msmodelslim.quant.processor.quant.base import FloatQuantConfig
from msmodelslim.quant.processor.quant.w8a8 import W8A8ProcessorConfig, W8A8QuantConfig
from msmodelslim.quant.processor.quant.w8a8_dynamic import W8A8DynamicProcessorConfig, W8A8DynamicQuantConfig
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig
from msmodelslim.quant.session.session import quant_model, SessionConfig

