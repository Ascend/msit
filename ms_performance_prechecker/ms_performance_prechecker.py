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
import json
import csv
from collections import namedtuple
from glob import glob

from ms_performance_prechecker.prechecker.utils import str_ignore_case, CHECK_TYPES

MIES_INSTALL_PATH = "MIES_INSTALL_PATH"
MINDIE_SERVICE_DEFAULT_PATH = "/usr/local/Ascend/mindie/latest/mindie-service"


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
    print(f">>>> {file_path = }")
    if not file_path or not os.path.exists(file_path):
        return None
    if file_path.endswith(".json"):
        return read_json(file_path)
    if file_path.endswith(".csv"):
        return read_csv(file_path)
    return None


def get_next_dict_item(dict_value):
    return dict([next(iter(dict_value.items()))])


""" parse_mindie_server_config """


def parse_mindie_server_config():
    print("\n>>>> mindie_service_config:")
    mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)
    if not os.path.exists(mindie_service_path):
        print(f"[WARNING] mindie config.json: {mindie_service_path} not exists, will skip related checkers")
        return None

    mindie_service_config = read_csv_or_json(os.path.join(mindie_service_path, "conf", "config.json"))
    print(f">>>> mindie_service_config: {get_next_dict_item(mindie_service_config) if mindie_service_config else None}")
    return mindie_service_config


""" prechecker """


def prechecker(mindie_service_config, check_type):
    import ms_performance_prechecker.prechecker
    from ms_performance_prechecker.prechecker.register import REGISTRY, print_answer

    print("\n<think>")
    for name, checker in REGISTRY.items():
        print(name)
        checker(mindie_service_config, check_type)
    print("</think>")

    print_answer()


""" arg_parse """


def arg_parse(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-t", "--check_type",
        type=str_ignore_case,
        default=CHECK_TYPES.deepseek,
        choices=CHECK_TYPES,
        help="check type",
    )
    return parser.parse_known_args(argv)[0]


def main():
    import sys

    args = arg_parse(sys.argv)
    mindie_service_config = parse_mindie_server_config()
    prechecker(mindie_service_config, args.check_type)


if __name__ == "__main__":
    main()
