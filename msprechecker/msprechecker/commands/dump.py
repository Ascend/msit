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
from pathlib import Path
from textwrap import dedent
from typing import Optional, Tuple, List, Dict

from ..core import Collector
from ..strategies import Image, Network, Ascend, Weight, Configs, Config, Sys, Env
from ..utils import Utils, PreFetch, Output, Framework, LOGGER, resolve_weight_dir, WeightDirNotFoundError, \
    parse_rank_table

from . import CmdStrategy, CmdType


def setup_dump(subparsers: argparse._SubParsersAction, parents=None):
    if parents is None:
        parents = []
    desc = dedent("""\
        DUMP - Collect and save the current environment, system, and configuration context.

        This command gathers environment variables, system information, configuration files,
        and network topology, then saves them to a specified output file for later comparison.
    """)

    epilog = dedent("""\
        Example:
          msprechecker dump                                                                           # Default saved to current dir 'msprechecker_dumped.json'
          msprechecker dump --output-path /output/path                                                # Save snapshots to custom path: '/output/path'
          msprechecker dump --user-config-path user_config.json --mindie-env-path mindie_env.json     # Dump extra PD disaggregation configuration files
    """)

    dump_parser = subparsers.add_parser(
        CmdType.DUMP.value,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc,
        usage="msprechecker dump [EXTRA OPTIONS] [--output-path <PATH>]",
        epilog=epilog,
        help="Dump the current context for later comparison",
        parents=parents,
    )

    dump_parser.add_argument(
        "--output-path",
        metavar="",
        default="./msprechecker-dumped-{}.json",
        help=(
            "Path to save the dumped context (JSON format). "
            "Default: ./msprechecker-dumped-{current_time}.json."
        ),
    )
    _add_extra_options(dump_parser)

    return dump_parser


def _add_extra_options(dump_parser):
    framework = PreFetch.get_framework()
    if framework == Framework.HOST:
        LOGGER.debug("Non-Container environment detected.")
    elif framework == Framework.UNKNOWN:
        LOGGER.debug("Unknown image type, skip.")
    else:
        LOGGER.debug(f"Detected image type: {framework.value}.")

    env_group = dump_parser.add_argument_group("Env Options")
    env_group.add_argument(
        "--ascend-only",
        action="store_true",
        help="Filter and collect only Ascend-related environment variables. Default: False.",
    )

    network_group = dump_parser.add_argument_group("Network Options")
    network_group.add_argument(
        "--rank-table-path",
        help="Path to the rank table file. Supports both A2 and A3 formats.",
    )

    # Cache the detection result for the lifetime of this setup call so that
    # the framework is only probed once, not on every attribute access.
    framework = PreFetch.get_framework()
    nargs = "+" if framework in {Framework.VLLM, Framework.SGLANG} else "*"
    config_group = dump_parser.add_argument_group("Config Options")
    config_group.add_argument(
        "--configs",
        "-c",
        nargs=nargs,
        help=(
            "Configuration files to validate, specified as '<name>:<path>'.\n"
            "  - <name>: Configuration identifier that will be used as a key in the dumped file\n"
            "  - <path>: File system path to the configuration file\n"
        ),
    )

    weight_group = dump_parser.add_argument_group("Weight Options")
    weight_group.add_argument(
        "-w", "--weight-dir", metavar="", help="Directory path containing model weights."
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

    @staticmethod
    def _resolve_weight_dir(
            framework: Framework,
            paths: List[str],
            cli_weight_dir: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        """
        Try to resolve weight_dir from config files.

        Returns (resolved_weight_dir, has_conflict).
        has_conflict=True means the caller should abort.
        """
        resolved: Optional[str] = None

        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext not in {".json", ".sh"}:
                continue

            try:
                weight_dir = resolve_weight_dir(framework, config_path=Path(path))
            except WeightDirNotFoundError:
                LOGGER.debug("No weight directory found in config %r; skipping", path)
                continue

            if cli_weight_dir is not None and cli_weight_dir != weight_dir:
                LOGGER.error(
                    "Weight directory conflict: config specifies %r but CLI provides %r",
                    weight_dir,
                    cli_weight_dir,
                )
                return None, True  # conflict

            resolved = weight_dir

        return resolved, False

    @staticmethod
    def _build_collector(
            framework: Framework,
            paths: List[str],
            weight_dir: Optional[str],
            rank_table_path: Optional[str],
    ) -> Collector:
        """
        Assemble a Collector with all applicable strategies.
        """
        strategies = [Env(), Sys(), Ascend(), Image()]
        configs = Configs()
        for path in paths:
            configs.add(Config(name=path, config_path=path))
        strategies.append(configs)

        if weight_dir:
            strategies.append(Weight(weight_dir=weight_dir))

        if rank_table_path is not None:
            rank_table = parse_rank_table(Path(rank_table_path), framework)
            strategies.append(Network(rank_table=rank_table))

        return Collector(strategies)

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute the dump command.

        This method performs the following steps:
        1. Detect the framework (Ascend or MindIE).
        2. Parse the config fields and validate the paths.
        3. Resolve the weight directory from the config files.
        4. Build a collector with all applicable strategies.
        5. Collect the data and save it to the specified output path.

        Args:
            args: Command-line arguments with output_path attribute

        Returns:
            0 if successful, 1 if error occurred
        """
        framework = PreFetch.get_framework()
        if framework == Framework.HOST:
            Utils.log_error_and_exit("Running msprechecker dump in a non-container environment is not supported.")
        elif framework == Framework.UNKNOWN:
            Utils.log_error_and_exit("Unsupported image type, exit.")

        if not args.configs and framework == Framework.MINDIE:
            default_config_path = "/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json"
            LOGGER.info(f"Using default mindie config path: {default_config_path}")
            configs = [default_config_path]
        else:
            configs = args.configs

        weight_dir, has_conflict = self._resolve_weight_dir(
            framework, configs, args.weight_dir
        )
        if has_conflict:
            return 1

        effective_weight_dir = weight_dir or args.weight_dir

        collector = self._build_collector(
            framework, configs, effective_weight_dir, args.rank_table_path
        )
        abspath = os.path.abspath(args.output_path.format(Utils.get_time_stamp()))
        try:
            data = collector.collect()
            data['timestamp'] = Utils.get_time_stamp()
            with open(abspath, "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            LOGGER.error(e)
            Utils.log_error_and_exit("Error occurred while saving to {}".format(abspath))

        Output.message("All information has been saved in: {}".format(abspath))
        Output.message(
            "You may now use 'msprechecker compare' to compare two or more dumped files for discrepancies!"
        )
        return 0
