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

from msit.lib.msit_c import log

_STAR = "*"
_DEBUG = "DEBUG"
_INFO = "INFO"
_WARNING = "WARNING"
_ERROR = "ERROR"
LOG_LEVEL = [_DEBUG, _INFO, _WARNING, _ERROR]
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


class MsitLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MsitLogger, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def get_level_id(level: str):
        if level.upper() in LOG_LEVEL:
            return LOG_LEVEL.index(level.upper())
        else:
            return LOG_LEVEL.index(LOG_LEVEL[1])

    @staticmethod
    def error(msg):
        log.print_log(LOG_LEVEL.index(_ERROR), msg)

    @staticmethod
    def warning(msg):
        log.print_log(LOG_LEVEL.index(_WARNING), msg)

    @staticmethod
    def info(msg):
        log.print_log(LOG_LEVEL.index(_INFO), msg)

    @staticmethod
    def debug(msg):
        log.print_log(LOG_LEVEL.index(_DEBUG), msg)

    def set_level(self, level: str):
        level_id = self.get_level_id(level)
        log.set_log_level(level_id)


logger = MsitLogger()


def print_log_with_star(info_message: str):
    total_length = _TOTAL_CHAR_LENGTH
    logger.info(_STAR * total_length)
    logger.info(f"{_STAR}{info_message.center(total_length - 2)}{_STAR}")
    logger.info(_STAR * total_length)
