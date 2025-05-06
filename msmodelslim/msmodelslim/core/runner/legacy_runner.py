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

from typing import Any, List, Optional, Union, Tuple, Dict, Generator

import torch
from torch import nn

from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.base.runner import BaseRunner
from msmodelslim.core.runner.generated_schedule import ProcessUnit, generated_schedule


class LegacyProcessUnit(ProcessUnit):
    def __init__(self, processor: BaseProcessor, input_datas: Optional[List[Any]] = None):
        super().__init__(processor, input_datas)

    def build_generators(self) -> List[Generator[ProcessRequest, Any, None]]:
        if self.input_datas:
            dataloader = self._create_dataloader(self.input_datas, 0, 1, 1)
            return [_legacy_forward_func(self.processor.model, data) for data in dataloader]
        else:
            return [_legacy_visit_func(self.processor.model)]


class LegacyRunner(BaseRunner):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.process_unit: List[LegacyProcessUnit] = []

    def add_processor(self, processor: BaseProcessor, input_datas: Optional[List[Any]] = None, append: bool = True):
        self.process_unit.append(LegacyProcessUnit(processor, input_datas))

    def run(self):
        generated_schedule(self.process_unit)


def _legacy_forward_func(model: torch.nn.Module,
                         inputs: Union[List, Tuple, Dict],
                         ):
    yield ProcessRequest("", model, inputs if isinstance(inputs, list) or isinstance(inputs, tuple) else [inputs],
                         inputs if isinstance(inputs, dict) else {})


def _legacy_visit_func(model: torch.nn.Module, ):
    yield ProcessRequest("", model, [], {})
