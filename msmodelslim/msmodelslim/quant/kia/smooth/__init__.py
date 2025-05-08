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

from functools import lru_cache
from typing import Callable, Dict, List, Optional, Union

import torch
from torch import nn

from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig


@lru_cache(maxsize=None)
def get_kia_smooth_ln_fcs() -> Callable:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils import smooth_ln_fcs as kia_smooth_ln_fcs
    return kia_smooth_ln_fcs


def smooth_ln_fcs(
        layer_norm: nn.Module,
        linears: Union[nn.Module, List[nn.Module]],
        statis: Dict[str, torch.Tensor],
        cfg: Optional[AntiOutlierConfig] = None,
) -> None:
    cfg = cfg or AntiOutlierConfig(anti_method="m1")
    return get_kia_smooth_ln_fcs()(cfg, layer_norm, linears, statis, cfg.alpha)
