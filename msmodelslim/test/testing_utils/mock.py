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
from typing import Optional

from unittest.mock import MagicMock


def _mock_json_safe_dump(obj, path, indent=None, extensions="json", check_user_stat=True):
    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as json_file:
        json.dump(obj, json_file, indent=indent)


def _mock_get_valid_write_path(path: str, extensions: Optional[str] = None) -> str:
    return path


def mock_kia_library():
    sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils'] = MagicMock()
    sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs'] = MagicMock()
    sys.modules['msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.atomic_power_outlier'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.calibration'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.quant_modules'] = MagicMock()


def mock_security_library():
    sys.modules['ascend_utils.common.security.path'] = MagicMock()
    sys.modules['ascend_utils.common.security.path'].json_safe_dump = _mock_json_safe_dump
    sys.modules['ascend_utils.common.security.path'].get_valid_write_path = _mock_get_valid_write_path
    sys.modules['ascend_utils.common.security.path'].get_valid_path = _mock_get_valid_write_path
