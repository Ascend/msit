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
from pathlib import Path
import logging
from collections import namedtuple

TARGETS = namedtuple("TARGETS", ["FirstTokenTime", "Throughput"])("FirstTokenTime", "Throughput")
_SUGGESTION_TYPES = ["env", "system", "config"]
SUGGESTION_TYPES = namedtuple("SUGGESTION_TYPES", _SUGGESTION_TYPES)(*_SUGGESTION_TYPES)
MAX_FILE_ITER_TIME = 10000
MAX_FILE_SIZE = 10
BYTES_TO_GB = 1024**3

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


def vaild_readable_directory(path):
    if not os.path.exists(path):
        raise FileExistsError(f"Path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise ValueError(f"Path '{path}' is not a directory.")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Directory '{path}' is not readable.")


def vaild_readable_file(path):
    path = Path(path)
    if not path.exists():
        raise FileExistsError(f"path '{path}' does not exist.")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"path '{path}' is not readable.")
    
    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE * BYTES_TO_GB:
        raise ValueError(f"path '{path}' cannot exceed {MAX_FILE_SIZE}GB.")
    return path


def get_directory_size(path):
    iter_time = 0
    total_size = 0
    for dirpath, _, filename in os.walk(path):
        for f in filename:
            if iter_time > MAX_FILE_ITER_TIME:
                raise ValueError(f"path '{path}' iter times cannot exceed {MAX_FILE_ITER_TIME}.")
            iter_time += 1

            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size / BYTES_TO_GB