# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

from abc import ABC, abstractmethod

from msit.common.validation import (
    valid_framework,
    valid_level,
    valid_log_level,
    valid_seed,
    valid_step_or_rank,
    valid_task,
)
from msit.utils.constants import CfgConst, MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.io import load_json
from msit.utils.log import logger


class BaseConfig(ABC):
    def __init__(self, config_path, task="", step: list = None, level: list = None):
        self.config_path = config_path
        self.config = load_json(self.config_path)
        self.task_config = {}
        self.task = task
        self.step = step
        self.level = level

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name == "check_config" and callable(attr):

            def wrapper(*args, **kwargs):
                self.common_check()
                self.get_task_dict()
                result = attr(*args, **kwargs)
                return result

            return wrapper
        return attr

    @abstractmethod
    def check_config(self):
        pass

    def get_task_dict(self):
        self.task_config = self.config.get(self.config.get(CfgConst.TASK))
        if self.task_config:
            return self.task_config
        else:
            raise MsitException(
                MsgConst.REQUIRED_ARGU_MISSING, f'Missing dictionary for key "{self.config.get(CfgConst.TASK)}".'
            )

    def common_check(self):
        logger.info("Validating configuration file parameters.")
        self._update_config(self.config, CfgConst.TASK, valid_task, self.task or self.config.get(CfgConst.TASK, ""))
        self._update_config(self.config, CfgConst.FRAMEWORK, valid_framework, self.config.get(CfgConst.FRAMEWORK, ""))
        self._update_config(
            self.config, CfgConst.STEP, valid_step_or_rank, self.step or self.config.get(CfgConst.STEP, [])
        )
        self._update_config(self.config, CfgConst.RANK, valid_step_or_rank, self.config.get(CfgConst.RANK, []))
        self._update_config(
            self.config,
            CfgConst.LEVEL,
            valid_level,
            self.level or self.config.get(CfgConst.LEVEL, [CfgConst.LEVEL_API]),
        )
        self._update_config(
            self.config, CfgConst.LOG_LEVEL, valid_log_level, self.config.get(CfgConst.LOG_LEVEL, "info")
        )
        self._update_config(self.config, CfgConst.SEED, valid_seed, self.config.get(CfgConst.SEED, None))

    def _update_config(self, dic: dict, key: str, check_fun, value: str):
        dic[key] = check_fun(value)


class Dict2Class:
    def __init__(self, data: dict, depth: int = 0):
        if depth > MsgConst.MAX_RECURSION_DEPTH:
            raise MsitException(
                MsgConst.RISK_ALERT, f"Maximum recursion depth of {MsgConst.MAX_RECURSION_DEPTH} exceeded."
            )
        if data.get(CfgConst.TASK) in data:
            data_pop = data.pop(data.get(CfgConst.TASK))
            for key, value in data_pop.items():
                if key == "input" and len(value) == 2:
                    setattr(self, "input_shape", value[0])
                    setattr(self, "input_path", value[1])
                setattr(self, key, value)
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Class(value, depth + 1))
            else:
                setattr(self, key, value)

    @classmethod
    def __getattr__(cls, item):
        raise MsitException(MsgConst.ATTRIBUTE_ERROR, f"{cls.__name__} object has no attribute {item}.")
