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

from functools import partial
from typing import Dict, List, Callable, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel

from msmodelslim import logger
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.const import ProcessStage
from msmodelslim.quant.processor.registry import PROCESSOR_REGISTRY
from msmodelslim.quant.kia.smooth import smooth_ln_fcs
from msmodelslim.quant.processor.anti_outlier.base import BaseSmoothProcessor, SmoothAdapter, StatKey


class M1ProcessorConfig(BaseModel):
    pass


@PROCESSOR_REGISTRY.register_by_name("m1")
class M1Processor(BaseSmoothProcessor):

    def __init__(self, model: nn.Module, cfg: M1ProcessorConfig, adapter: Optional[SmoothAdapter] = None):
        super().__init__(model, adapter)
        self.cfg = cfg
        self.act_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.smooth_pair: Dict[str, List[str]] = {}

    def stage(self) -> ProcessStage:
        return ProcessStage.FORWARD_ANTI_OUTLIER

    def preprocess(self, request: BatchProcessRequest) -> None:
        self.smooth_pair = self._get_local_norm_linear_smooth_pair(request.name, request.module,
                                                                   request.datas[0] if request.datas else None)
        logger.info(f"Smooth pair for submodule {request.name} is {self.smooth_pair}")
        return super().preprocess(request)

    def _get_stats_hook(self, name: str) -> Callable:
        def stats_hook(name: str, module: nn.Linear, args: tuple, kwargs: dict) -> None:

            tensor = args[0]

            if name not in self.act_stats:
                self.act_stats[name] = {}

            self.act_stats[name][StatKey.TENSOR] = tensor.cpu()

            hidden_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, hidden_dim).detach()  # [N,C]
            coming_max = torch.max(tensor, dim=0)[0]  # [C]
            coming_min = torch.min(tensor, dim=0)[0]  # [C]

            statis_dict = self.act_stats[name]

            # collect the min-max value
            if StatKey.STAT_KEY_MAX in statis_dict:
                statis_dict[StatKey.STAT_KEY_MAX] = torch.max(statis_dict[StatKey.STAT_KEY_MAX], coming_max)  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_MAX] = coming_max

            if StatKey.STAT_KEY_MIN in statis_dict:
                statis_dict[StatKey.STAT_KEY_MIN] = torch.min(statis_dict[StatKey.STAT_KEY_MIN], coming_min)  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_MIN] = coming_min

            # channel shifting
            if StatKey.STAT_KEY_SHIFT in statis_dict:
                statis_dict[StatKey.STAT_KEY_SHIFT] = (statis_dict[StatKey.STAT_KEY_MAX] + statis_dict[
                    StatKey.STAT_KEY_MIN]) / 2  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_SHIFT] = (coming_max + coming_min) / 2

            channel_max = torch.max((tensor - statis_dict[StatKey.STAT_KEY_SHIFT]).abs().detach(), dim=0)[0]

            if StatKey.STAT_KEY_SMOOTH_SCALE in statis_dict:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = torch.max(statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE],
                                                                       channel_max)
            else:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = channel_max

        return partial(stats_hook, name)

    def _install_statis_hook(self, name: str, module: nn.Module) -> None:
        for _, linear_names in self.smooth_pair.items():
            for linear_name in linear_names:
                self.model.get_submodule(linear_name).register_forward_hook(self._get_stats_hook(linear_name))

    def _apply_smooth(self, name: str, module: nn.Module) -> None:
        for norm_name, linear_names in self.smooth_pair.items():

            if not linear_names:
                logger.warning(f"No linear modules provided for {norm_name} to apply smooth")
                continue

            logger.info(f"Apply m1 smooth from {linear_names} to {norm_name} with cfg {self.cfg}")

            norm_module = self.model.get_submodule(norm_name)
            linear_modules = [self.model.get_submodule(linear_name) for linear_name in linear_names]
            smooth_ln_fcs(norm_module, linear_modules, self.act_stats[linear_names[0]])

        self.act_stats.clear()
        self.smooth_pair.clear()
