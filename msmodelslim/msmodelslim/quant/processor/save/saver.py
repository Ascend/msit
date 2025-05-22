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
from abc import abstractmethod
from collections.abc import Callable
from typing import Optional, Dict, Type, Any

import torch.distributed as dist
from pydantic import BaseModel, Field
from torch import nn

from msmodelslim import logger
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.base import BaseSaver
from msmodelslim.quant.processor.base import SessionBaseProcessor
from msmodelslim.quant.processor.const import ProcessStage
from msmodelslim.quant.processor.quant.w8a8 import W8A8LinearFakeQuantizer
from msmodelslim.quant.processor.quant.w8a8_dynamic import W8A8DynamicLinearFakeQuantizer
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY, PROCESSOR_CONFIG_REGISTRY
from msmodelslim.utils.dist import DistHelper
from msmodelslim.utils.registry import Registry


@PROCESSOR_CONFIG_REGISTRY.register_by_name("save")
class SaverProcessorConfig(BaseModel):
    save_output_path: Optional[str] = None
    safetensors_name: Optional[str] = None
    json_name: Optional[str] = None
    save_type: Optional[str] = Field(default="safe_tensor")
    part_file_size: Optional[int] = Field(default=4)
    model_quant_type: Optional[str] = Field(default="w8a8")


class BaseSaverBackend:
    def __init__(self, model: nn.Module, save_cfg: SaverProcessorConfig):
        self.model = model
        self.save_cfg = save_cfg
        self.saver = self._create_saver()
        self.dist_helper = None
        self.processed_modules = set()
        self.process_map: Dict[Type[nn.Module], Callable[[str, nn.Module], None]] = {
            W8A8LinearFakeQuantizer: self._process_w8a8_linear_fake_quantizer,
            W8A8DynamicLinearFakeQuantizer: self._process_w8a8_dynamic_linear_fake_quantizer,
        }

    def save(self, prefix: str, module: nn.Module):

        shared_modules_slice = set()

        if dist.is_initialized():
            self.dist_helper = DistHelper(module)
            shared_modules_slice = self.dist_helper.get_shared_modules_slice(prefix=prefix)

        for name, sub_module in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix != "" else name
            self.processed_modules.add(full_name)

            if self.dist_helper is not None:
                is_local_only = self.dist_helper.is_local_only(full_name)
                is_in_shared_modules_slice = full_name in shared_modules_slice
                save_in_this_rank = is_local_only or is_in_shared_modules_slice
                if not save_in_this_rank:
                    continue

            if type(sub_module) in self.process_map:
                self.process_map[type(sub_module)](full_name, sub_module)
                continue

            self._process_module(full_name, QuantType.FLOAT, sub_module)

        self.processed_modules.add(prefix)

        self.dist_helper = None

    def pre_process(self):
        self.saver.pre_process()

    def post_process(self):

        if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
            for name, module in self.model.named_modules():
                if name not in self.processed_modules:
                    self._process_module(name, QuantType.FLOAT, module)

        self.saver.post_process()

    @abstractmethod
    def _create_saver(self) -> BaseSaver:
        raise NotImplementedError()

    def _process_w8a8_linear_fake_quantizer(self, prefix: str, module: nn.Module):
        return self._process_module(prefix, QuantType.W8A8, module)

    def _process_w8a8_dynamic_linear_fake_quantizer(self, prefix: str, module: nn.Module):
        return self._process_module(prefix, QuantType.W8A8_DYNAMIC, module)

    def _process_module(self, prefix: str, quant_type: QuantType, module: nn.Module):
        for key, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{key}" if prefix != "" else key
            self.saver.save(full_name, quant_type, param.clone().detach().cpu())


SAVER_BACKEND_REGISTRY = Registry[BaseSaverBackend]()


@PROCESSOR_REGISTRY.register_by_name("save")
class SaverProcessor(SessionBaseProcessor):

    def __init__(self, model: nn.Module, cfg: SaverProcessorConfig, **kwargs: Dict[str, Any]):
        super().__init__(model)
        self.backend = self.create_backend(model, cfg)

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True

    def stage(self) -> ProcessStage:
        return ProcessStage.SAVE_MODEL

    def create_backend(self, model: nn.Module, cfg: SaverProcessorConfig) -> BaseSaverBackend:
        _ = self
        return SAVER_BACKEND_REGISTRY.get_by_name('mindie')(model, cfg)

    def pre_run(self) -> None:
        logger.info("Pre run for SaverProcessor")
        self.backend.pre_process()

    def post_run(self) -> None:
        logger.info("Post run for SaverProcessor")
        self.backend.post_process()

    def postprocess(self, request: BatchProcessRequest) -> None:
        self.backend.save(request.name, request.module)
