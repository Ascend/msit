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

from msmodelslim import logger
from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import BatchProcessRequest


class ForwardProcessorMerger(BaseProcessor):
    """
    前向量化处理器合并器，用于将多个前向量化处理器合并为一个处理器，用于减少模型推理的次数。
    """

    def __init__(self, model: nn.Module, processors: List[BaseProcessor]):
        super().__init__(model)
        self.processors = processors
        self.processor_names = [processor.__class__.__name__ for processor in processors]

    def process(self, request: BatchProcessRequest) -> None:

        """
        合并多个前向量化处理器，用于减少模型推理的次数。
        """
        logger.info(f"Start forward merger for {request.name} with {self.processor_names}")
        for processor in self.processors:
            processor.preprocess(request)
        self._run_forward_if_need(request)
        for processor in self.processors:
            processor.postprocess(request)
        logger.info(f"End forward merger for {request.name} with {self.processor_names}")

    def pre_run(self) -> None:
        for processor in self.processors:
            processor.pre_run()

    def post_run(self) -> None:
        for processor in self.processors:
            processor.post_run()

    def is_data_free(self) -> bool:
        """
        判断处理器是否需要数据。
        """
        return all(processor.is_data_free() for processor in self.processors)
