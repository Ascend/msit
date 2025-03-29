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

import os
from functools import wraps
from sys import stdout
from time import localtime, strftime, time

_STAR = "*"
_DEBUG = "DEBUG"
_INFO = "INFO"
_WARNING = "WARNING"
_ERROR = "ERROR"
_LOG_LEVEL = [_DEBUG, _INFO, _WARNING, _ERROR]
_SPECIAL_CHAR = [
    "\n",
    "\r",
    "\u007f",
    "\b",
    "\f",
    "\t",
    "\v",
    "\u000b",
    "%08",
    "%09",
    "%0a",
    "%0b",
    "%0c",
    "%0d",
    "%7f",
    "//",
    "\\",
    "&",
]
_TOTAL_CHAR_LENGTH = 80


def get_pid():
    return os.getpid()


def get_current_timestamp(used_for_log=True, microsecond=False):
    if used_for_log:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        if microsecond:
            return round(time() * 1e6) % 10**10
        else:
            timestamp = int(time())
            return timestamp


def stdout_flush():
    stdout.flush()


def filter_special_chars(func):
    @wraps(func)
    def func_level(self, msg, **kwargs):
        for char in _SPECIAL_CHAR:
            msg = msg.replace(char, "_")
        return func(self, msg, **kwargs)

    return func_level


class MsitLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MsitLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, level=_INFO):
        if not hasattr(self, "initialized"):
            self.level_id = self.get_level_id(level)
            self.initialized = True

    @staticmethod
    def get_level_id(level: str):
        if level.upper() in _LOG_LEVEL:
            return _LOG_LEVEL.index(level.upper())
        else:
            return _LOG_LEVEL.index(_LOG_LEVEL[1])

    @staticmethod
    def _print_log(level, msg, end="\n"):
        full_msg = f"{get_current_timestamp()} (PID {get_pid()}) [{level}] {msg}"
        print(full_msg, end=end)
        stdout_flush()

    def set_level(self, level: str):
        self.level_id = self.get_level_id(level)

    @filter_special_chars
    def error(self, msg):
        if self.level_id <= _LOG_LEVEL.index(_ERROR):
            self._print_log(_LOG_LEVEL[3], msg)

    @filter_special_chars
    def warning(self, msg):
        if self.level_id <= _LOG_LEVEL.index(_WARNING):
            self._print_log(_LOG_LEVEL[2], msg)

    @filter_special_chars
    def info(self, msg):
        if self.level_id <= _LOG_LEVEL.index(_INFO):
            self._print_log(_LOG_LEVEL[1], msg)

    @filter_special_chars
    def debug(self, msg):
        if self.level_id <= _LOG_LEVEL.index(_DEBUG):
            self._print_log(_LOG_LEVEL[0], msg)


logger = MsitLogger()


def print_log_with_star(info_message: str):
    total_length = _TOTAL_CHAR_LENGTH
    logger.info(_STAR * total_length)
    logger.info(f"{_STAR}{info_message.center(total_length - 2)}{_STAR}")
    logger.info(_STAR * total_length)
