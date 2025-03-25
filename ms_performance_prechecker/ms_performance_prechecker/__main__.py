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
import time
import argparse
import tempfile
from collections import namedtuple
from glob import glob

from ms_performance_prechecker.prechecker import CHECKERS, CHECKER_INFOS_STR
from ms_performance_prechecker.prechecker.utils import str_ignore_case, logger, set_log_level
from ms_performance_prechecker.prechecker.utils import LOG_LEVELS, RUN_MODES, CHECKER_TYPES
from ms_performance_prechecker.prechecker.utils import MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH, RANKTABLEFILE
from ms_performance_prechecker.prechecker.utils import deep_compare_dict, get_next_dict_item


LOG_LEVELS_LOWER = [ii.lower() for ii in LOG_LEVELS.keys()]

DEFAULT_DUMP_PATH = os.path.join(
    tempfile.gettempdir(), f"ms_performance_prechecker_dump_{time.strftime('%Y%m%d_%H%M%S')}.json"
)
DAFAULT_ENV_SAVE_PATH = "ms_performance_prechecker_env.sh"

RANKTABLE_FILE = os.getenv(RANKTABLEFILE, None)
MINDIE_SERVICE_PATH = os.getenv(MIES_INSTALL_PATH, MINDIE_SERVICE_DEFAULT_PATH)

COMMON_ARGS = [
    dict(
        args=["-l", "--log_level"], kwargs=dict(default="info", choices=LOG_LEVELS_LOWER, help="specify log level.")
    ),
]
DUMP_COMMON_ARGS = [
    dict(
        args=["-ch", "--checkers"],
        kwargs=dict(
            nargs="+",
            default=[CHECKER_TYPES.basic],
            choices=CHECKER_TYPES,
            help=f"specify checker types. {CHECKER_INFOS_STR}",
        ),
    ),
    dict(
        args=["-service", "--service_config_path"],
        kwargs=dict(type=str, default=MINDIE_SERVICE_PATH, help="MINDIE service or config json path"),
    ),
    dict(
        args=["-ranktable", "--ranktable_file"],
        kwargs=dict(default=RANKTABLE_FILE, help="HCCL rank table file path."),
    ),
]


def get_next_dict_item(dict_value):
    return dict([next(iter(dict_value.items()))])


def get_all_register_prechecker(checkers=(CHECKER_TYPES.basic,)):   
    if CHECKER_TYPES.all in checkers:
        checkers = [CHECKER_TYPES.all]

    res = []
    for checker_type in checkers:
        res += CHECKERS.get(checker_type, [])
    return res


def print_contents():
    from ms_performance_prechecker.prechecker.register import CONTENTS, CONTENT_PARTS

    logger.info(f"")

    if CONTENTS.get(CONTENT_PARTS.sys, None):
        sorted_contents = [ii.split(" ", 1)[-1] for ii in sorted(CONTENTS[CONTENT_PARTS.sys])]
        sys_info = "系统信息：\n\n    " + "\n    ".join(sorted_contents) + "\n"
        logger.info(sys_info)


def run_env_dump(dump_file_path=DEFAULT_DUMP_PATH, service_config_path=None, checkers=(CHECKER_TYPES.basic,), **kwargs):
    precheckers = get_all_register_prechecker(checkers)
    all_envs = {}
    for prechecker in precheckers:
        name = prechecker.name()
        envs = prechecker.collect_env(dump_file_path=dump_file_path, mindie_service_path=service_config_path, **kwargs)

        all_envs[name] = envs

    if dump_file_path is not None:
        with open(dump_file_path, "w") as f:
            json.dump(all_envs, f, indent=2)

        logger.info(f"dump file saved to: {dump_file_path}")
    return all_envs


def run_compare(dump_file_paths=None, **kwargs):
    if dump_file_paths is None or len(dump_file_paths) < 2:
        logger.error("Please provide dump file path")
        return False

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
    return has_diff


def run_precheck(
    save_env="ms_performance_prechecker_env.sh", service_config_path=None, checkers=(CHECKER_TYPES.basic,), **kwargs
):
    precheckers = get_all_register_prechecker(checkers)
    for prechecker in precheckers:
        prechecker.precheck(env_save_path=save_env, mindie_service_path=service_config_path, **kwargs)

    if CHECKER_TYPES.basic in checkers or CHECKER_TYPES.all in checkers:
        print_contents()
        logger.warning("本工具提供的为经验建议，实际效果与具体的环境/场景有关，建议以实测为准")


def run_distribute_compare(
    ranktable_file=None,
    service_config_path=None,
    master_ip=None,
    master_port=None,
    local_rank=None,
    world_size=None,
    checkers=(CHECKER_TYPES.basic,),
    **kwargs,
):
    from ms_performance_prechecker.prechecker import cluster_collector

    dump_env = run_env_dump(dump_file_path=None, service_config_path=service_config_path, checkers=checkers)
    dump_env_json_str = json.dumps(dump_env)

    cluster_collector.init_global_distribute_env(
        ranktable_file=ranktable_file, service_config_path=service_config_path, master_ip=master_ip
    )
    dump_env_json_str_dict = cluster_collector.distribute_collector(
        dump_env_json_str, master_ip=master_ip, master_port=master_port, rank=local_rank, world_size=world_size
    )
    logger.debug(dump_env_json_str_dict)

    if dump_env_json_str_dict is None:
        logger.info("Not master node, skip comparing")
        return

    env_ips = []
    env_infos = []
    for ip, dump_env_json_str in dump_env_json_str_dict.items():
        env_ips.append(ip)
        env_infos.append(json.loads(dump_env_json_str))

    skip_keys = [".Env.MIES_CONTAINER_IP", ".Env.ASCEND_CUSTOM_OPP_PATH"]
    has_diff = deep_compare_dict(env_infos, env_ips, skip_keys=skip_keys)
    if not has_diff:
        logger.info("No difference found")
    logger.info("== compare end ==")


def sub_parser_precheck(subparsers):
    ranktable_file = os.getenv(RANKTABLEFILE, None)

    parser = subparsers.add_parser(
        RUN_MODES.precheck, formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="precheck configuration"
    )
    parser.add_argument(
        "-s",
        "--save_env",
        default="ms_performance_prechecker_env.sh",
        help="Save env changes as a file which could be applied directly.",
    )
    for ii in DUMP_COMMON_ARGS + COMMON_ARGS:
        parser.add_argument(*ii.get('args', []), **ii.get('kwargs', {}))
    parser.set_defaults(func=run_precheck)


def sub_parser_dump(subparsers):
    parser = subparsers.add_parser(
        RUN_MODES.dump, formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="dump configuration"
    )

    parser.add_argument(
        "-d",
        "--dump_file_path",
        nargs="?",
        default=DEFAULT_DUMP_PATH,
        help="Path save envs. It could be a list of path when you want to compare envs of multiple path.",
    )
    for ii in DUMP_COMMON_ARGS + COMMON_ARGS:
        parser.add_argument(*ii.get('args', []), **ii.get('kwargs', {}))
    parser.set_defaults(func=run_env_dump)


def sub_parser_compare(subparsers):
    parser = subparsers.add_parser(
        RUN_MODES.compare, formatter_class=argparse.ArgumentDefaultsHelpFormatter, help="compare dumped configuration"
    )
    parser.add_argument(
        "dump_file_paths",
        nargs="+",
        help="Saved configuration path. It could be a list of path when you want to compare envs of multiple path.",
    )
    for ii in COMMON_ARGS:
        parser.add_argument(*ii.get('args', []), **ii.get('kwargs', {}))
    parser.set_defaults(func=run_compare)


def sub_parser_distribute_compare(subparsers):
    # Multi-machine
    parser = subparsers.add_parser(
        RUN_MODES.distribute_compare,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="compare distribute envs",
    )
    parser.add_argument(
        "-ip",
        "--master_ip",
        default=None,
        help="master ip, required if RANKTABLEFILE not available or using different value",
    )
    parser.add_argument(
        "-port",
        "--master_port",
        default=None,
        help="master port, required if MIES_INSTALL_PATH not available or master_port already occupied by others",
    )
    parser.add_argument(
        "-rank",
        "--local_rank",
        type=int,
        default=None,
        help="local rank, required if RANKTABLEFILE not available or using different value",
    )
    parser.add_argument(
        "-size",
        "--world_size",
        type=int,
        default=None,
        help="world size, required if RANKTABLEFILE not available or using different value",
    )
    for ii in DUMP_COMMON_ARGS + COMMON_ARGS:
        parser.add_argument(*ii.get('args', []), **ii.get('kwargs', {}))
    parser.set_defaults(func=run_distribute_compare)


def main():
    # args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="sub-command help")
    sub_parser_precheck(subparsers)
    sub_parser_dump(subparsers)
    sub_parser_compare(subparsers)
    sub_parser_distribute_compare(subparsers)
    args, _ = parser.parse_known_args()

    # init
    set_log_level(getattr(args, 'log_level', 'info'))

    # run
    if hasattr(args, "func"):
        args.func(args=args, **vars(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
