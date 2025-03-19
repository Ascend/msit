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

import csv
import os
import json
import logging
from collections import namedtuple

CHECK_TYPES = namedtuple("CHECK_TYPES", ["basic", "deepseek"])("basic", "deepseek")
_RUN_MODES = ["precheck", "dump", "compare", "distribute_compare"]
RUN_MODES = namedtuple("RUN_MODES", _RUN_MODES)(*_RUN_MODES)
_SUGGESTION_TYPES = ["env", "system", "config"]
SUGGESTION_TYPES = namedtuple("SUGGESTION_TYPES", _SUGGESTION_TYPES)(*_SUGGESTION_TYPES)

MIES_INSTALL_PATH = "MIES_INSTALL_PATH"
MINDIE_SERVICE_DEFAULT_PATH = "/usr/local/Ascend/mindie/latest/mindie-service"

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


def same(array):
    return len(set(array)) == 1


def print_diff(diffs, names, key=""):
    print(f"- key\033[94m {key}\033[91m diffs \033[0m")
    for index, name in enumerate(names):
        print(f"    * {name}:")
        print(f"        {diffs[index]}")


def deep_compare_dict(dicts, names, parent_key="", skip_keys=None):
    if skip_keys and parent_key in skip_keys:
        return False

    has_diff = False
    types = [type(ii) for ii in dicts]
    if not same(types):
        print_diff([f"type <{t.__name__}> : {str(x)[0:30]}" for t, x in zip(types, dicts)], names, parent_key)
        return True
    all_keys = set()
    if isinstance(dicts[0], dict):
        for dict_item in dicts:
            all_keys.update(dict_item.keys())

        for key in all_keys:
            cur_has_diff = deep_compare_dict(
                [dict_item.get(key) for dict_item in dicts], names, parent_key + "." + key, skip_keys=skip_keys
            )
            has_diff = cur_has_diff or has_diff
    elif isinstance(dicts[0], list):
        lens = [len(x) for x in dicts]
        if not same(lens):
            print_diff([f"len: {x}" for x in lens], names, parent_key)
            return True
        else:
            for index in range(len(dicts[0])):
                cur_has_diff = deep_compare_dict([x[index] for x in dicts], names, parent_key + f"[{index}]")
                has_diff = cur_has_diff or has_diff
    else:
        if not same([str(x) for x in dicts]):
            print_diff([str(x) for x in dicts], names, parent_key)
            return True
    return has_diff


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
    if mindie_service_path is None or mindie_service_path == "":
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


def read_csv(file_path):
    result = {}
    with open(file_path, mode="r", newline="", encoding="utf-8") as ff:
        for row in csv.DictReader(ff):
            for kk, vv in row.items():
                result.setdefault(kk, []).append(vv)
    return result


def read_json(file_path):
    with open(file_path) as ff:
        result = json.load(ff)
    return result


def read_csv_or_json(file_path):
    logger.debug("file_path = %s", file_path)

    if not file_path or not os.path.exists(file_path):
        return None
    if file_path.endswith(".json"):
        return read_json(file_path)
    if file_path.endswith(".csv"):
        return read_csv(file_path)
    return None


def get_next_dict_item(dict_value):
    return dict([next(iter(dict_value.items()))])


def parse_mindie_server_config(mindie_service_path=None):
    logger.debug("mindie_service_config:")
    if mindie_service_path is None:
        mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)
    if not os.path.exists(mindie_service_path):
        logger.warning(f"mindie config.json: {mindie_service_path} not exists, will skip related checkers")
        return None

    mindie_service_config = read_csv_or_json(os.path.join(mindie_service_path, "conf", "config.json"))
    logger.debug(
        "mindie_service_config: %s", get_next_dict_item(mindie_service_config) if mindie_service_config else None
    )
    return mindie_service_config
