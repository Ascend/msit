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

from torch import nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.base import BaseSaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.factory import SaverFactory
from msmodelslim.quant.processor.save.saver import SAVER_BACKEND_REGISTRY
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig, BaseSaverBackend


@SAVER_BACKEND_REGISTRY.register_by_name("mindie")
class MindIESaverBackend(BaseSaverBackend):
    def __init__(self, model: nn.Module, save_cfg: SaverProcessorConfig):
        super().__init__(model, save_cfg)

    def _create_saver(self) -> BaseSaver:
        quant_cfg_map = {
            "w8a8": QuantConfig(a_bit=8, w_bit=8),
            "w8a8_dynamic": QuantConfig(a_bit=8, w_bit=8, is_dynamic=True),
        }
        quant_cfg = quant_cfg_map[self.save_cfg.model_quant_type]
        return SaverFactory.create(self.save_cfg.save_type,
                                   output_dir=self.save_cfg.save_output_path,
                                   cfg=quant_cfg,
                                   safetensors_name=self.save_cfg.safetensors_name,
                                   json_name=self.save_cfg.json_name,
                                   part_file_size=self.save_cfg.part_file_size)
