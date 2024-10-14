# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import stat
from collections import namedtuple
from pathlib import Path

import torch
from safetensors.torch import safe_open

from msit_llm.common.log import logger
from msit_llm.common.utils import check_data_file_size
from msit_llm.common.constant import MAX_WEIGHT_DATA_SIZE

_SCENARIOS = ["torch_to_float_atb", "float_atb_to_quant_atb", "torch_to_float_python_atb"]
SCENARIOS = namedtuple("SCENARIOS", _SCENARIOS)(*_SCENARIOS)


def get_transform_scenario(source_path, to_python=False):
    if os.path.isfile(source_path) and source_path.endswith(".cpp"):  # Single cpp input
        return SCENARIOS.float_atb_to_quant_atb

    try:
        from transformers.configuration_utils import PretrainedConfig

        _ = PretrainedConfig.get_config_dict(source_path)
        return SCENARIOS.torch_to_float_python_atb if to_python else SCENARIOS.torch_to_float_atb
    except Exception:
        pass  # Not an error

    if os.path.isdir(source_path) and any([ii.endswith(".cpp") for ii in os.listdir(source_path)]):
        return SCENARIOS.float_atb_to_quant_atb
    else:
        return None


def write_file(save_path, string):
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(save_path, flags, modes), 'w') as ff:
        ff.write(string)


def load_model_dict(model_path):
    if Path(model_path).is_file() and check_data_file_size(model_path, max_size=MAX_WEIGHT_DATA_SIZE):
        state_dict = torch.load(model_path)
        return state_dict
    elif Path(model_path).is_dir():
        suffix_list = ['.bin', '.safetensors', '.pt']
        for suffix in suffix_list:
            file_list = list(Path(model_path).glob('*' + suffix))
            if not file_list:
                continue
            state_dict = {}
            for fp in file_list:
                fp = str(fp)
                if (not Path(fp).is_file() or 
                    not check_data_file_size(fp, max_size=MAX_WEIGHT_DATA_SIZE)):
                    continue  # 如果不是普通文件或者文件大小超过限制，则跳过该文件(实际上会报错并终止)
                if suffix == '.safetensors':
                    with safe_open(fp, framework='pt') as ff:
                        ss = {kk: ff.get_tensor(kk).half() for kk in ff.keys()}
                else:
                    ss = torch.load(fp)
                state_dict.update(ss)
            return state_dict
    return {}