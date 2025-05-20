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

from enum import Enum


class ActivationQuantMethod(Enum):
    MINMAX = "minmax"
    HISTOGRAM = "histogram"
    CLIP_MINMAX = "clip_minmax"


class ActivationQuantScope(Enum):
    PER_TENSOR = "per_tensor"
    PER_TOKEN = "per_token"
    PER_HEAD = "per_head"


class WeightQuantMethod(Enum):
    MINMAX = "minmax"
    GPTQ = "gptq"
    HQQ = "hqq"


class WeightQuantScope(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"
