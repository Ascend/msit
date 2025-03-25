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
import sys
import json
import csv
import socket
import logging
import time
from collections import namedtuple

_RUN_MODES = ["precheck", "dump", "compare", "distribute_compare"]
RUN_MODES = namedtuple("RUN_MODES", _RUN_MODES)(*_RUN_MODES)
_CHECKER_TYPES = ["basic", "hccl", "model", "hardware", "all"]
CHECKER_TYPES = namedtuple("CHECKER_TYPES", _CHECKER_TYPES)(*_CHECKER_TYPES)

MIES_INSTALL_PATH = "MIES_INSTALL_PATH"
MINDIE_SERVICE_DEFAULT_PATH = "/usr/local/Ascend/mindie/latest/mindie-service"
RANKTABLEFILE = "RANKTABLEFILE"

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


def is_deepseek_model(model_name):
    return "deepseek" in model_name.lower().replace(" ", "").replace("-", "").replace("_", "")


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
    else:
        logger.error(f"Neither a csv or json: {file_path}")
    return None


def get_next_dict_item(dict_value):
    return dict([next(iter(dict_value.items()))]) if dict_value else None


def get_mindie_server_config(mindie_service_path=None):
    if mindie_service_path is None:
        mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)
    if not mindie_service_path.endswith(".json"):
        mindie_service_path = os.path.join(mindie_service_path, "conf", "config.json")
    return mindie_service_path


def parse_mindie_server_config(mindie_service_path=None):
    mindie_service_path = get_mindie_server_config(mindie_service_path)
    logger.debug("mindie_service_path=%s", mindie_service_path)
    if not os.path.exists(mindie_service_path):
        logger.warning(f"mindie config.json={mindie_service_path} not exists, will skip related checkers")
        return None
    mindie_service_config = read_csv_or_json(mindie_service_path)
    logger.debug(
        "mindie_service_config: %s", get_next_dict_item(mindie_service_config) if mindie_service_config else None
    )
    return mindie_service_config


def parse_ranktable_file(ranktable_file=None):
    if ranktable_file is None:
        ranktable_file = os.getenv(RANKTABLEFILE, None)
    logger.debug("ranktable_file=%s", ranktable_file)
    if not ranktable_file or not os.path.exists(ranktable_file):
        logger.warning(f"ranktable_file={ranktable_file} not exists, will skip related checkers")
        return None

    ranktable = read_csv_or_json(ranktable_file)
    logger.debug(
        "ranktable: %s", get_next_dict_item(ranktable) if ranktable else None
    )
    return ranktable


def get_model_path_from_mindie_config(mindie_service_config=None, mindie_service_path=None):
    if not mindie_service_config:
        mindie_service_config = parse_mindie_server_config(mindie_service_path)
    if not mindie_service_config:
        return None, None
    model_deploy_config = mindie_service_config.get("BackendConfig", {}).get("ModelDeployConfig", {})
    model_config = model_deploy_config.get("ModelConfig", [])
    if not model_config:
        return None, None
    model_name = model_config[0].get("modelName", None)
    model_weight_path = model_config[0].get("modelWeightPath", None)
    return model_name, model_weight_path


def get_local_to_master_ip(test_ip="8.8.8.8"):
    local_ip = "127.0.0.1"
    try:
        ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ss.connect((test_ip, 80))
        local_ip = ss.getsockname()[0]
    finally:
        ss.close()
    return local_ip


def get_interface_by_ip(local_ip):
    import psutil

    local_ip_list = local_ip if isinstance(local_ip, (list, tuple)) else [local_ip]
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address in local_ip_list:
                return interface, addr.address
    return None, None


def run_shell_command(command, fail_msg=""):
    import subprocess
    from shutil import which

    command_split = command.split()
    base_command = command_split[0]
    base_command_path = which(base_command)
    if not base_command_path:
        logger.error(f"{base_command} command not exists" + fail_msg)
        return {}

    command_split = [base_command_path] + command_split[1:]
    try:
        result = subprocess.run(command_split, capture_output=True, text=True, check=False, shell=False)
    except Exception as err:
        logger.error(f"Failed calling {base_command}" + fail_msg)
        return {}
    return result


class ProcessBarStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + "\r")  # 用 \r 代替默认的 \n
            self.flush()
        except Exception:
            self.handleError(record)


class SimpleProgressBar:
    def __init__(self, iterable, desc=None, total=None):
        self.iterable = iterable
        self.desc = desc or ""
        self.total = total if total is not None else len(iterable)
        self.current = 0
        self.start_time = time.time()
        self.logger = self._init_logger()

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        self.logger.info('\n')

    @staticmethod
    def _init_logger():
        local_logger = logging.getLogger(__name__)
        local_logger.setLevel(logging.INFO)

        handler = ProcessBarStreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        local_logger.addHandler(handler)
        return local_logger

    def update(self, n=1):
        self.current += n
        self._print_progress()

    def _print_progress(self):
        progress = self.current / self.total
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        percent = progress * 100

        # 计算剩余时间
        elapsed_time = time.time() - self.start_time
        if progress > 0:
            remaining_time = (elapsed_time / progress) * (1 - progress)
        else:
            remaining_time = 0

        self.logger.info(
            f'\r{self.desc} |{bar}| {percent:.1f}% [{self.current}/{self.total}] ETA: {remaining_time:.1f}s'
        )
