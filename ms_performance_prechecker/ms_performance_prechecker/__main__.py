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
import datetime
import tempfile
from collections import namedtuple
from glob import glob

from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, LOG_LEVELS, RUN_MODES
from ms_performance_prechecker.prechecker.utils import MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH
from ms_performance_prechecker.prechecker.utils import logger, set_log_level
from ms_performance_prechecker.prechecker.utils import str_ignore_case, deep_compare_dict, read_csv_or_json
from ms_performance_prechecker.prechecker.utils import get_next_dict_item

LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]


DEFAULT_DUMP_PATH = os.path.join(tempfile.gettempdir(),
    f"ms_performance_prechecker_dump_{ datetime.datetime.now().strftime('%Y%m%d_%H%M%S') }.json")


def get_next_dict_item(dict_value):
    return dict([next(iter(dict_value.items()))])


def get_all_register_perchecker():
    from ms_performance_prechecker.prechecker import checkers
    return checkers


def parse_mindie_server_config(mindie_service_path=None):
    logger.debug("mindie_service_config:")
    if mindie_service_path is None:
        mindie_service_path = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)
    if not os.path.exists(mindie_service_path):
        logger.warning(f"mindie config.json: {mindie_service_path} not exists, will skip related checkers")
        return None

    mindie_service_config = read_csv_or_json(os.path.join(mindie_service_path, "conf", "config.json"))
    logger.debug(
        f"mindie_service_config: {get_next_dict_item(mindie_service_config) if mindie_service_config else None}"
    )
    return mindie_service_config


def print_contents():
    from ms_performance_prechecker.prechecker.register import CONTENTS, CONTENT_PARTS
    logger.info(f"")

    if CONTENTS.get(CONTENT_PARTS.sys, None):
        sorted_contents = [ii.split(" ", 1)[-1] for ii in sorted(CONTENTS[CONTENT_PARTS.sys])]
        sys_info = "系统信息：\n\n    " + "\n    ".join(sorted_contents) + "\n"
        logger.info(sys_info)


def run_env_dump(dump_file_path=DEFAULT_DUMP_PATH, mindie_service_path=None, **kwargs):
    percheckers = get_all_register_perchecker()
    all_envs = {}
    for perchecker in percheckers:
        name = perchecker.name()
        envs = perchecker.collect_env(dump_file_path=dump_file_path, mindie_service_path=mindie_service_path, **kwargs)
        
        all_envs[name] = envs
    
    if dump_file_path is not None:
        with open(dump_file_path, "w") as f:
            json.dump(all_envs, f, indent=2)

        logger.info(f"dump file saved to: {dump_file_path}")
    return all_envs


def run_compare(dump_file_paths=None, mindie_service_path=None, **kwargs):
    if dump_file_paths is None or len(dump_file_paths) < 2:
        logger.error("Please provide dump file path")
        return

    env_infos = []
    env_names = []
    for dump_file_path in dump_file_paths:
        with open(dump_file_path, "r") as f:
            env_infos.append(json.load(f))
            env_names.append(dump_file_path)
        
    # 递归逐层比对
    logger.info("== compare start ==")
    has_diff = deep_compare_dict(env_infos, env_names)
    if not has_diff:
        logger.info("No difference found")
    logger.info("== compare end ==")


def run_precheck(check_type=CHECK_TYPES.deepseek,
    env_save_path="ms_performance_prechecker_env.sh", mindie_service_path=None, **kwargs):
    percheckers = get_all_register_perchecker()

    for perchecker in percheckers:
        perchecker.precheck(check_type=check_type, env_save_path=env_save_path,
            mindie_service_path=mindie_service_path, **kwargs)

    print_contents()
    logger.info("本工具提供的为经验建议，实际效果与具体的环境/场景有关，建议以实测为准")


def arg_parse():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode",
        type=str_ignore_case,
        default=RUN_MODES.precheck,
        choices=RUN_MODES,
        nargs='?',
        help="run mode",
    )
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
    parser.add_argument(
        "-d",
        "--dump_file_path",
        nargs="*",
        default=[DEFAULT_DUMP_PATH],
        help="Path save envs. It could be a list of path when you want to compare envs of multiple path.",
    )
    parser.add_argument("-l", "--log_level", default="info", choices=LOG_LEVELS_LOWER, help="specify log level.")
    return parser.parse_known_args()[0]


def main():
    # args
    args = arg_parse()
    
    # init
    set_log_level(args.log_level)

    # run
    if args.mode == RUN_MODES.precheck:
        run_precheck(check_type=args.check_type, env_save_path=args.save_env, args=args)
    elif args.mode == RUN_MODES.envdump:
        dump_file_path = args.dump_file_path[0]
        _ = run_env_dump(dump_file_path, args=args)
    elif args.mode == RUN_MODES.compare:
        run_compare(args.dump_file_path, args=args)
    else:
        logger.error("Unknown mode")

if __name__ == "__main__":
    main()
