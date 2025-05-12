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

from os import PathLike
from typing import Optional

import torch
from pydantic import BaseModel
from torch import nn

from msmodelslim import logger
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.factory import SaverFactory
from msmodelslim.quant.processor.base import SessionBaseProcessor
from msmodelslim.quant.processor.const import ProcessStage
from msmodelslim.quant.processor.quant.base import BaseSessionQuantConfig, FloatQuantConfig
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY
from msmodelslim.quant.quantizer.base.fake import BaseFakeQuantizer


class SaverProcessorConfig(BaseModel):
    save_output_path: Optional[PathLike] = None
    safetensors_name: Optional[str] = None
    json_name: Optional[str] = None
    save_type: Optional[str] = None
    part_file_size: Optional[int] = None


class SessionSaver:
    def __init__(self, save_cfg: SaverProcessorConfig):
        quant_cfg = QuantConfig(a_bit=8, w_bit=8)
        self.saver = SaverFactory.create(save_cfg.save_type,
                                         output_dir=save_cfg.save_output_path,
                                         cfg=quant_cfg,
                                         safetensors_name=save_cfg.safetensors_name,
                                         json_name=save_cfg.json_name,
                                         part_file_size=save_cfg.part_file_size)

    def save(self, name: str, cfg: BaseSessionQuantConfig, value: torch.Tensor) -> None:
        self.saver.save(name, cfg.quant_type(), value.cpu().clone().detach())


@PROCESSOR_REGISTRY.register_by_name("save")
class SaverProcessor(SessionBaseProcessor):

    def __init__(self, model: nn.Module, cfg: SaverProcessorConfig):
        super().__init__(model)
        self.saver = SessionSaver(cfg)

    def is_data_free(self) -> bool:
        return True

    def stage(self) -> ProcessStage:
        return ProcessStage.SAVE_MODEL

    def pre_run(self) -> None:
        logger.info("Pre process for SaverProcessor")
        self.saver.saver.pre_process()

    def post_run(self) -> None:
        logger.info("Post process for SaverProcessor")
        self.saver.saver.post_process()

    def postprocess(self, request: BatchProcessRequest) -> None:
        for name, module in request.module.named_modules():
            full_name = f"{request.name}.{name}" if request.name != "" else name
            if isinstance(module, BaseFakeQuantizer):
                if not isinstance(module.cfg, BaseSessionQuantConfig):
                    raise ValueError(f"{module.cfg} must be a subclass of BaseSessionQuantConfig")
                for key, param in module.state_dict().items():
                    self.saver.save(f"{full_name}.{key}", module.cfg, param)
            else:
                for key, param in module.named_parameters(recurse=False):
                    cfg = FloatQuantConfig()
                    self.saver.save(f"{full_name}.{key}", cfg, param)
