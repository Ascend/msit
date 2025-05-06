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

from typing import List

from torch import nn

from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import BatchProcessRequest


class BackwardProcessorMerger(BaseProcessor):
    def __init__(self, model: nn.Module, processors: List[BaseProcessor]):
        super().__init__(model)
        self.processors = processors

    def process(self, request: BatchProcessRequest) -> None:
        raise NotImplementedError()
