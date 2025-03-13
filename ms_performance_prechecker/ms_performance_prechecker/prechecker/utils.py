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

CHECK_TYPES = namedtuple("CHECK_TYPES", ["basic", "deepseek"])("basic", "deepseek")
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


def str_to_digit(input_str, default_value=None):
    if not input_str.replace(".", "", 1).isdigit():
        return default_value
    return float(input_str) if "." in input_str else int(input_str)


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


def get_dict_value_by_pos(dict_value, target_pos):
    cur = dict_value
    for kk in target_pos.split(":"):
        if not cur:
            cur = None
            break
        if isinstance(cur, list) and str.isdigit(kk):
            cur = cur[int(kk)]
        elif kk in cur:
            cur = cur[kk]
        else:
            cur = None
            break
    return cur


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


def get_version_info(mindie_service_path):
    if mindie_service_path is None:
        mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)

    version_path = os.path.join(mindie_service_path, "version.info")

    if not os.path.exists(version_path): 
        return {}
    
    version_info = {}
    with open(version_path) as f:
        for line in f:
            line_split = line.split(":")
            key, value = line_split[0], line_split[-1]
            version_info[key.strip()] = value.strip()

    return version_info


logger = logging.getLogger("ms_performance_prechecker_logger")
set_logger(logger)
