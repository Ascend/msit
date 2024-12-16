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

from functools import wraps

from msit.common.constants import MsgConst
from msit.utils.toolkits import get_pid, get_current_time, stdout_flush


class MsitLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MsitLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, level=MsgConst.INFO):
        if not hasattr(self, MsgConst.INITIALIZED):
            self.level_id = self.get_level_id(level)
            self.initialized = True

    @staticmethod
    def get_level_id(level: str):
        if level.upper() in MsgConst.LOG_LEVEL:
            return MsgConst.LOG_LEVEL.index(level.upper())
        else:
            return MsgConst.LOG_LEVEL.index(MsgConst.LOG_LEVEL[1])

    @staticmethod
    def _print_log(level, msg, end="\n"):
        full_msg = f"{get_current_time()} (PID {get_pid()}) [{level}] {msg}"
        print(full_msg, end=end)
        stdout_flush()

    def set_level(self, level: str):
        self.level_id = self.get_level_id(level)

    def filter_special_chars(func):
        @wraps(func)
        def func_level(self, msg, **kwargs):
            for char in MsgConst.SPECIAL_CHAR:
                msg = msg.replace(char, "_")
            return func(self, msg, **kwargs)
        return func_level

    @filter_special_chars
    def error(self, msg):
        if self.level_id <= MsgConst.LogLevel.ERROR.value:
            self._print_log(MsgConst.LOG_LEVEL[3], msg)

    @filter_special_chars
    def warning(self, msg):
        if self.level_id <= MsgConst.LogLevel.WARNING.value:
            self._print_log(MsgConst.LOG_LEVEL[2], msg)

    @filter_special_chars
    def info(self, msg):
        if self.level_id <= MsgConst.LogLevel.INFO.value:
            self._print_log(MsgConst.LOG_LEVEL[1], msg)

    @filter_special_chars
    def debug(self, msg):
        if self.level_id <= MsgConst.LogLevel.DEBUG.value:
            self._print_log(MsgConst.LOG_LEVEL[0], msg)


logger = MsitLogger()


def print_log_with_star(info_message: str):
    total_length = MsgConst.TOTAL_CHAR_LENGTH
    logger.info(MsgConst.STAR * total_length)
    logger.info(f"{MsgConst.STAR}{info_message.center(total_length - 2)}{MsgConst.STAR}")
    logger.info(MsgConst.STAR * total_length)
