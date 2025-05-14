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
from typing import Union, List, Tuple, Dict, Optional, Any, Callable, Generator

import torch.nn
from torch import nn, distributed as dist

from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import ProcessRequest, BatchProcessRequest
from msmodelslim.core.base.runner import BaseRunner
from msmodelslim.core.runner.generated_schedule import generated_schedule, ProcessUnit


class LayerWiseProcessUnit(ProcessUnit):
    def __init__(self, processor: BaseProcessor, input_datas: Optional[List[Any]] = None):
        super().__init__(processor, input_datas)

    def build_generators(self) -> List[Generator[ProcessRequest, Any, None]]:
        if self.input_datas:
            dataloader = self._create_dataloader(self.input_datas, 0, 1, 1)
            return [_transformers_generated_forward_func(self.processor.model, data) for data in dataloader]
        else:
            return [_generated_decoder_layer_visit_func(self.processor.model)]


class LayerProcessHook(BaseProcessor):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.process_func = None

    def process(self, request: BatchProcessRequest):
        if self.process_func:
            self.process_func(request.name, request.module)

    def set_process_func(self, func: Callable[[str, nn.Module], None]):
        self.process_func = func


class LayerWiseRunner(BaseRunner):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.process_unit: List[LayerWiseProcessUnit] = []

    def add_processor(self, processor: BaseProcessor, input_datas: Optional[List[Any]] = None, append: bool = True):
        self.process_unit.append(LayerWiseProcessUnit(processor, input_datas))

    def run(self):
        generated_schedule(self.process_unit)


class _TransformersForwardBreak(Exception):
    """内部使用的异常类，用于中断前向传播并捕获输入"""
    pass


def _generated_decoder_layer_visit_func(model: torch.nn.Module,
                                        transformer_blocks: Optional[List[Tuple[str, torch.nn.Module]]] = None,
                                        ):
    if transformer_blocks is None:
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "decoderlayer" in module.__class__.__name__.lower()
        ]

    if dist.is_initialized():
        dist.barrier()

    for name, block in transformer_blocks:
        yield ProcessRequest(name, block, [], {})


def _transformers_generated_forward_func(model: torch.nn.Module,
                                         inputs: Union[List, Tuple, Dict, Any],
                                         transformer_blocks: Optional[List[Tuple[str, torch.nn.Module]]] = None,
                                         ):
    if transformer_blocks is None:
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "decoderlayer" in module.__class__.__name__.lower()
        ]

    # 存储第一个transformer block的输入
    first_block_input = None

    def break_hook(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        nonlocal first_block_input
        first_block_input = (args, kwargs,)
        raise _TransformersForwardBreak()

    hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True)]

    # 执行一次前向传播以获取输入
    try:
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            model(*inputs)
        elif isinstance(inputs, dict):
            model(**inputs)
        else:
            model(inputs)
    except _TransformersForwardBreak:
        pass
    except Exception as e:
        raise e
    finally:
        for hook in hooks:
            hook.remove()

    if first_block_input is None:
        raise ValueError("Can't get first block input, please check the model and input")

    # 循环处理每个transformer block
    current_inputs = first_block_input

    if dist.is_initialized():
        dist.barrier()

    for name, block in transformer_blocks:
        args, kwargs = current_inputs
        outputs = yield ProcessRequest(name, block, args, kwargs)
        hidden_states = outputs[0]
        current_inputs = ((hidden_states,), current_inputs[1])
