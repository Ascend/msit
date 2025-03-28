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
from argparse import Namespace
from pathlib import Path

from msit.common.validation import valid_dump_level, valid_dump_task, valid_log_level, valid_seed, valid_step_or_rank
from msit.utils.constants import DumpConst, MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.io import load_json
from msit.utils.log import logger


class BaseConfig(ABC):
    def __init__(self, config_path):
        self.is_from_cmd = False
        self.config_path = config_path
        if not self.config_path:
            self.config_path = Path(__file__).resolve().parent.parent / "config.json"
        self.config = load_json(self.config_path)
        self.task_config = {}

    @abstractmethod
    def check_config(self):
        pass

    def common_check(self, task: str = None, step: list = None, args: Namespace = None):
        self.is_from_cmd = isinstance(args, Namespace)
        if self.is_from_cmd:
            logger.info("Configure parameters via the command line.")
        else:
            logger.info(f"Configure parameters via {self.config_path}.")
        self._update_config(self.config, DumpConst.TASK, args, valid_dump_task, task or DumpConst.STATISTICS)
        self._update_config(self.config, DumpConst.RANK, args, valid_step_or_rank, self.config.get(DumpConst.RANK, []))
        self._update_config(
            self.config, DumpConst.STEP, args, valid_step_or_rank, step or self.config.get(DumpConst.STEP, [])
        )
        self._update_config(
            self.config,
            DumpConst.LEVEL,
            args,
            valid_dump_level,
            self.config.get(DumpConst.LEVEL, [DumpConst.LEVEL_API]),
        )
        self._update_config(
            self.config, DumpConst.LOG_LEVEL, args, valid_log_level, self.config.get(DumpConst.LOG_LEVEL, "info")
        )
        self._update_config(self.config, DumpConst.SEED, args, valid_seed, self.config.get(DumpConst.SEED))

    def _update_config(self, dic, key, args, check_fun, value):
        dic[key] = getattr(args, key) if self.is_from_cmd else check_fun(value)


class Dict2Class:
    def __init__(self, data: dict, depth: int = 0):
        if depth > MsgConst.MAX_RECURSION_DEPTH:
            raise MsitException(
                MsgConst.RISK_ALERT, f"Maximum recursion depth of {MsgConst.MAX_RECURSION_DEPTH} exceeded."
            )
        if data.get(DumpConst.TASK) in data:
            data_pop = data.pop(data.get(DumpConst.TASK))
            for key, value in data_pop.items():
                setattr(self, key, value)
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Class(value, depth + 1))
            else:
                setattr(self, key, value)

    @classmethod
    def __getattr__(cls, item):
        raise MsitException(MsgConst.ATTRIBUTE_ERROR, f"{cls.__name__} object has no attribute {item}.")
