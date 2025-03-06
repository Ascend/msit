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

import logging
from collections import namedtuple

TARGETS = namedtuple("TARGETS", ["FirstTokenTime", "Throughput"])("FirstTokenTime", "Throughput")
_SUGGESTION_TYPES = ["env", "system", "config"]
SUGGESTION_TYPES = namedtuple("SUGGESTION_TYPES", _SUGGESTION_TYPES)(*_SUGGESTION_TYPES)

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}


def str_ignore_case(value):
    return value.lower().replace("_", "").replace("-", "")


def walk_dict(data, parent_key=""):
    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(value, (dict, tuple, list)):
                yield key, value, parent_key
            else:
                new_key = f"{parent_key}.{key}" if parent_key else key
                yield from walk_dict(value, new_key)
    elif isinstance(data, (tuple, list)):
        for index, item in enumerate(data):
            if not isinstance(item, (dict, tuple, list)):
                yield key, item, parent_key
            else:
                new_key = f"{parent_key}.{index}" if parent_key else index
                yield from walk_dict(item, new_key)


def set_log_level(level="info"):
    if level.lower() in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS.get(level.lower()))
    else:
        logger.warning("Set %s log level failed.", level)


def set_logger(msit_logger):
    msit_logger.propagate = False
    msit_logger.setLevel(logging.INFO)
    if not msit_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        msit_logger.addHandler(stream_handler)


logger = logging.getLogger("msservice_advisor_logger")
set_logger(logger)
