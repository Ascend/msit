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
import json
import os
import stat
import sys
from typing import Optional, Tuple
from unittest.mock import MagicMock

import torch


def _mock_json_safe_dump(obj, path, indent=None, extensions="json", check_user_stat=True):
    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as json_file:
        json.dump(obj, json_file, indent=indent)


def _mock_get_valid_write_path(path: str, extensions: Optional[str] = None) -> str:
    return path


def _mock_get_write_directory(path: str, *args, **kwargs) -> str:
    if not os.path.exists(path):
        os.makedirs(name=path, mode=0o750, exist_ok=True)
    return path


def _mocked_init_weight_quant_normal(weight: torch.Tensor,
                                     bits: int = 8,
                                     is_sym=True,
                                     is_signed: bool = True,
                                     intergral_zp=True,
                                     admm=None,
                                     round_opt=False,
                                     mm_tensor=True,
                                     fake_quant=True,
                                     hqq=False,
                                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回原始值，不进行量化"""

    scale = torch.ones([1]) if mm_tensor else torch.ones([weight.shape[0], 1])
    offset = torch.zeros([1]) if mm_tensor else torch.zeros([weight.shape[0], 1])
    return weight, weight, scale, offset


def _mocked_linear_quantization_params(bit: int,
                                       x_min: torch.Tensor,
                                       x_max: torch.Tensor,
                                       intergral_zero_point: bool,
                                       q_signed: bool,
                                       sym: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(1.0), torch.tensor(0.0)


def _mocked_fake_quantize(tensor: torch.Tensor,
                          scale: torch.Tensor,
                          zero_point: torch.Tensor,
                          bits: int = 8,
                          is_signed: bool = True,
                          round_opt: bool = False,
                          is_fp=False,
                          dequant=True,
                          group_size=-1,
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    return tensor, tensor


def mock_kia_library():
    sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils'] = MagicMock()
    sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs'] = MagicMock()
    sys.modules['msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.atomic_power_outlier'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.calibration'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.quant_modules'] = MagicMock()

    mock_quant_funcs_dict = {
        'init_weight_quant_normal': _mocked_init_weight_quant_normal,
        'linear_quantization_params': _mocked_linear_quantization_params,
        'fake_quantize': _mocked_fake_quantize
    }

    for func_name, func in mock_quant_funcs_dict.items():
        setattr(sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs'], func_name, func)


def mock_security_library():
    sys.modules['ascend_utils.common.security.path'] = MagicMock()
    sys.modules['ascend_utils.common.security.path'].json_safe_dump = _mock_json_safe_dump
    sys.modules['ascend_utils.common.security.path'].get_valid_write_path = _mock_get_valid_write_path
    sys.modules['ascend_utils.common.security.path'].get_valid_path = _mock_get_valid_write_path
    sys.modules['ascend_utils.common.security.path'].get_write_directory = _mock_get_write_directory
