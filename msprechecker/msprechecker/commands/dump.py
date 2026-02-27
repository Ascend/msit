# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import argparse
import json
import logging
from textwrap import dedent

from msguard import Rule

from msguard.security import open_s

from ..core.collector import Collector
from ..core.strategy import Config, Env, Network, Sys, Weight
from ..utils.ascend import Framework, ParserRegistry, get_framework, Framework, get_weight_dir, search_weight_dir_mindie, search_weight_dir_vllm, search_weight_dir_sglang
from ..utils.constant import LOG_LEVELS

from .base import CmdStrategy, CmdType



logger = logging.getLogger(__name__)


def setup_dump_parser(subparsers):
    dump_parser = subparsers.add_parser(
        CmdType.DUMP.value,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_get_description(),
        usage="msprechecker dump [EXTRA OPTIONS] [--output-path <PATH>]",
        epilog=_get_epilog(),
        help="Dump the current context for later comparison",
    )

    _add_dump_arguments(dump_parser)
    _add_extra_options(dump_parser)

    return dump_parser


def _get_description():
    return dedent("""\
        DUMP - Collect and save the current environment, system, and configuration context.

        This command gathers environment variables, system information, configuration files,
        and network topology, then saves them to a specified output file for later comparison.
    """)


def _get_epilog():
    return dedent("""\
        Example:
          msprechecker dump                                                                           # Default saved to current dir 'msprechecker_dumped.json'
          msprechecker dump --output-path /output/path                                                # Save snapshots to custom path: '/output/path'
          msprechecker dump --user-config-path user_config.json --mindie-env-path mindie_env.json     # Dump extra PD disaggregation configuration files
    """)


def _add_dump_arguments(dump_parser):
    dump_parser.add_argument(
        "--output-path",
        metavar="",
        default="./msprechecker_dumped.json",
        help=(
            "Path to save the dumped context (JSON format). "
            "Default: './msprechecker_dumped.json'."
        ),
    )
    dump_parser.add_argument(
        "-l",
        "--log-level",
        choices=LOG_LEVELS,
        default="info",
        help="Set the logging level.",
    )


def _add_extra_options(dump_parser):
    framework = get_framework()
    weight_dir = get_weight_dir()

    env_group = dump_parser.add_argument_group("Env Options")
    env_group.add_argument(
        "--filter",
        action="store_true",
        help="Filter and collect only Ascend-related environment variables. Default: False.",
    )

    network_group = dump_parser.add_argument_group("Network Options")
    network_group.add_argument(
        "--rank-table-path",
        "-rt",
        help="Path to the rank table file. Supports both A2 and A3 formats.",
    )

    config_group = dump_parser.add_argument_group("Config Options")
    config_group.add_argument(
        "--configs",
        "-c",
        nargs="*",
        required=framework in {Framework.VLLM, Framework.SGLANG},
        help=(
            "Configuration files to validate, specified as '<name>:<path>'.\n"
            "  - <name>: Configuration identifier that will be used as a key in the dumped file\n"
            "  - <path>: File system path to the configuration file\n"
        ),
    )

    weight_group = dump_parser.add_argument_group("Weight Options")
    weight_group.add_argument(
        "--weight-dir", metavar="", help="Directory path containing model weights."
    )
    weight_group.add_argument(
        "--chunk-size",
        metavar="",
        choices=[32, 64, 128, 256],
        type=int,
        default=32,
        help=(
            "Specify the chunk size (in KB) for calculating sha256sum of model tensors. "
            "Only tensors will be checksummed if this option is set. "
            "Supported values: 32, 64, 128, 256."
        ),
    )


class Dump(CmdStrategy):
    def execute(self, args: argparse.Namespace) -> int:
        collector = Collector([Env(), Sys()])

        if isinstance(args.configs, list):
            for config_field in args.configs:
                if ":" not in config_field:
                    logger.warning(
                        'Invalid config field format, expected "name:path": %s',
                        config_field,
                    )
                    continue
                name, path = config_field.split(":", 1)

                rule = Rule.input_file_read
                if not rule.is_satisfied_by(path):
                    logger.warning("Expected %r to be %s", path, rule)
                    continue

                collector.add_strategy(Config(name=name, config_path=path))

                if '.' not in path:
                    continue

                ext = os.path.splitext(path)[-1]
                if ext == '.json':
                    if self._framework == Framework.MINDIE:
                        search_weight_dir_mindie(path)
                    else:
                        logger.error('Weight directory search only supported for MINDIE framework with JSON configs')
                        return 1
                elif ext == '.sh':
                    if self._framework == Framework.VLLM:
                        search_weight_dir_vllm(path)
                    elif self._framework == Framework.SGLANG:
                        search_weight_dir_sglang(path)
                    else:
                        logger.warning('Weight directory search for shell configs only supported for VLLM or SGLANG frameworks')
                        return 1

        weight_dir = args.weight_dir or get_weight_dir()
        if weight_dir:
            collector.add_strategy(Weight(weight_dir=weight_dir))

        if args.rank_table_path is not None:
            rank_table_parser = ParserRegistry.get(Framework.MINDIE)()
            rank_table = rank_table_parser.parse(args.rank_table_path)
            collector.add_strategy(Network(rank_table=rank_table))

        try:
            data = collector.collect()
            with open_s(args.output_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.warning('Error ocurred while saving')
            raise

        print("All information has been saved in: %r." % args.output_path)
        print(
            "You may now use 'msprechecker compare' to compare two or more dumped files for discrepancies!"
        )
        return 0
