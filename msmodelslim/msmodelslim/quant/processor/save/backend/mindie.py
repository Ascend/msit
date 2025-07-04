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

import numpy as np
import torch
from torch import nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.base import BaseSaver
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.factory import SaverFactory
from msmodelslim.quant.processor.quant.w8a8 import W8A8LinearFakeQuantizer
from msmodelslim.quant.processor.save.saver import SAVER_BACKEND_REGISTRY
from msmodelslim.quant.processor.save.saver import SaverProcessorConfig, BaseSaverBackend


def _deq_scale_to_int64(scale):
    scale = scale.numpy()
    scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
    scale = torch.tensor(scale)
    return scale


def _deq_scale_to_int64_by_dtype(scale, is_bf16):
    if is_bf16:
        return scale
    else:
        return _deq_scale_to_int64(scale)


@SAVER_BACKEND_REGISTRY.register_by_name("mindie")
class MindIESaverBackend(BaseSaverBackend):
    def __init__(self, model: nn.Module, save_cfg: SaverProcessorConfig):
        super().__init__(model, save_cfg)
        self.model_dtype = self._init_model_dtype()

    def _init_model_dtype(self):
        if hasattr(self.model, "config") and hasattr(self.model.config, "torch_dtype"):
            return self.model.config.torch_dtype

        if hasattr(self.model, "dtype"):
            return self.model.dtype

        return next(self.model.parameters()).dtype

    def _create_saver(self) -> BaseSaver:
        quant_cfg_map = {
            "w8a8": QuantConfig(a_bit=8, w_bit=8),
            "w8a8_dynamic": QuantConfig(a_bit=8, w_bit=8, is_dynamic=True),
            "w4a8_dynamic": QuantConfig(a_bit=8, w_bit=4, is_dynamic=True, group_size=256,
                                        is_lowbit=True, open_outlier=False),
        }
        quant_cfg = quant_cfg_map[self.save_cfg.quant_type]
        return SaverFactory.create(self.save_cfg.save_type,
                                   output_dir=self.save_cfg.save_output_path,
                                   cfg=quant_cfg,
                                   safetensors_name=self.save_cfg.safetensors_name,
                                   json_name=self.save_cfg.json_name,
                                   part_file_size=self.save_cfg.part_file_size)

    def _process_w8a8_linear_fake_quantizer(self, prefix: str, module: W8A8LinearFakeQuantizer):
        self.saver.save(f"{prefix}.input_scale", QuantType.W8A8, module.input_scale.to(self.model_dtype))
        self.saver.save(f"{prefix}.input_offset", QuantType.W8A8, module.input_offset.to(self.model_dtype))
        self.saver.save(f"{prefix}.deq_scale", QuantType.W8A8,
                        _deq_scale_to_int64_by_dtype(module.deq_scale, self.model_dtype == torch.bfloat16))
        self.saver.save(f"{prefix}.quant_bias", QuantType.W8A8, module.quant_bias.to(torch.int32))
        self.saver.save(f"{prefix}.weight", QuantType.W8A8, module.weight.to(torch.int8))
        return
