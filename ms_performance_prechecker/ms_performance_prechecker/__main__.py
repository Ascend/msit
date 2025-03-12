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

from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, LOG_LEVELS, SUGGESTION_TYPES
from ms_performance_prechecker.prechecker.utils import str_ignore_case, logger, set_log_level

MIES_INSTALL_PATH = "MIES_INSTALL_PATH"
MINDIE_SERVICE_DEFAULT_PATH = "/usr/local/Ascend/mindie/latest/mindie-service"

LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]


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
    logger.debug(f"{file_path = }")
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
    logger.debug("mindie_service_config:")
    mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)
    if not os.path.exists(mindie_service_path):
        logger.warning(f"mindie config.json: {mindie_service_path} not exists, will skip related checkers")
        return None

    mindie_service_config = read_csv_or_json(os.path.join(mindie_service_path, "conf", "config.json"))
    logger.debug(
        f"mindie_service_config: {get_next_dict_item(mindie_service_config) if mindie_service_config else None}"
    )
    return mindie_service_config


""" prechecker """


def run_precheck(mindie_service_config=None, check_type=CHECK_TYPES.deepseek):
    import ms_performance_prechecker.prechecker
    from ms_performance_prechecker.prechecker.register import REGISTRY, ANSWERS, CONTENTS, CONTENT_PARTS

    logger.debug("")
    logger.debug("<think>")
    for name, checker in REGISTRY.items():
        logger.debug(name)
        checker(mindie_service_config, check_type)
    logger.debug("</think>")

    if CONTENTS.get(CONTENT_PARTS.sys, None):
        sorted_contents = [ii.split(" ", 1)[-1] for ii in sorted(CONTENTS[CONTENT_PARTS.sys])]
        sys_info = "系统信息：\n\n    " + "\n    ".join(sorted_contents) + "\n"
        logger.info(sys_info)

    logger.info("")
    logger.info("<answer>")
    for suggesion_type in SUGGESTION_TYPES:
        for name, items in ANSWERS.get(suggesion_type, dict()).items():
            for action, reason in items:
                logger.info(f"[{suggesion_type}] {name}")
                logger.info(f"[action] {action}")
                logger.info(f"[reason] {reason}")
                logger.info("")
    logger.info("</answer>")


def save_env_contents(save_path):
    from ms_performance_prechecker.prechecker.register import CONTENTS, CONTENT_PARTS

    if not CONTENTS.get(CONTENT_PARTS.before, None) and not CONTENTS.get(CONTENT_PARTS.after, None):
        logger.info("None env related needs to save")
        return


    with open(save_path, "w") as ff:
        ff.write("ENABLE=${1-1}\n")
        ff.write('echo "ENABLE=$ENABLE"\n\n')
        ff.write('if [ "$ENABLE" = "1" ]; then\n    ')
        ff.write("\n    ".join(CONTENTS[CONTENT_PARTS.after]) + "\n")
        ff.write('else\n    ')
        ff.write("\n    ".join(CONTENTS[CONTENT_PARTS.before]) + "\n")
        ff.write('fi\n')

    logger.info("")
    logger.info(f"环境相关改动使能：source {save_path}; 使能后恢复：source {save_path} 0")


""" arg_parse """


def arg_parse(argv):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-t",
        "--check_type",
        type=str_ignore_case,
        default=CHECK_TYPES.deepseek,
        choices=CHECK_TYPES,
        help="check type",
    )
    parser.add_argument(
        "-s",
        "--save_env",
        default="ms_performance_prechecker_env.sh",
        help="Save env changes as a file which could be applied directly.",
    )
    parser.add_argument("-l", "--log_level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level.")
    return parser.parse_known_args(argv)[0]


def main():
    import sys

    args = arg_parse(sys.argv)
    set_log_level(args.log_level)
    mindie_service_config = parse_mindie_server_config()
    run_precheck(mindie_service_config, args.check_type)
    save_env_contents(args.save_env)


if __name__ == "__main__":
    main()
