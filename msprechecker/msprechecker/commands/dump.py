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

import argparse
import json
import logging
import os
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

from ..core.collector import Collector
from ..core.strategy import (
    Ascend,
    Config,
    CPUHighPerformance,
    Env,
    Network,
    Sys,
    Weight,
)
from ..util import (
    CONTAINER_CPU_HIGH_PERF_AMBIGUITY_HINT,
    detect_framework,
    Framework,
    is_in_container,
    parse_rank_table,
    resolve_weight_dir,
    WeightDirNotFoundError,
)
from .base import CommandType, CommandStrategy


logger = logging.getLogger(__name__)


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
          msprechecker dump                              # Default saved to current dir 'msprechecker_dumped.json'
          msprechecker dump --output-path /output/path   # Save snapshots to custom path: '/output/path'
          msprechecker dump -c user_config:user_config.json \\
            mindie_env:mindie_env.json                   # Dump extra PD disaggregation configuration files
    """)

    dump_parser = subparsers.add_parser(
        CommandType.CMD_DUMP.value,
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
        default="./msprechecker_dumped.json",
        help=(
            "Path to save the dumped context (JSON format). "
            "Default: './msprechecker_dumped.json'."
        ),
    )
    _add_extra_options(dump_parser)

    return dump_parser


def _add_extra_options(dump_parser):
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
    framework = detect_framework()
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


class Dump(CommandStrategy):
    @staticmethod
    def _split_config_field(config_field: str) -> Optional[Tuple[str, str]]:
        """
        Parse a single 'name:path' string into (name, path).
        Returns None and logs a warning if the format is invalid.
        """
        if ":" not in config_field:
            logger.warning(
                'Invalid config field format, expected "name:path": %s', config_field
            )
            return None

        name, path = config_field.split(":", 1)
        name, path = name.strip(), path.strip()

        if not name:
            logger.warning("Empty config name in field: %s", config_field)
            return None

        return name, path

    @staticmethod
    def _parse_config_fields(config_fields: List[str]) -> Dict[str, str]:
        """
        Parse and validate a list of 'name:path' config fields.
        Returns a dict mapping name -> validated path.
        """
        name_to_path: Dict[str, str] = {}

        for field in config_fields:
            result = Dump._split_config_field(field)
            if result is None:
                continue

            name, path = result
            if not os.path.isfile(path):
                logger.warning("Config file %r not found", path)
                continue

            if name in name_to_path:
                logger.warning(
                    "Duplicate config name %r, overwriting previous path", name
                )

            name_to_path[name] = path

        return name_to_path

    @staticmethod
    def _resolve_weight_dir(
        framework: Framework,
        name_to_path: Dict[str, str],
        cli_weight_dir: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        """
        Try to resolve weight_dir from config files.

        Returns (resolved_weight_dir, has_conflict).
        has_conflict=True means the caller should abort.
        """
        resolved: Optional[str] = None

        for path in name_to_path.values():
            ext = os.path.splitext(path)[1].lower()
            if ext not in {".json", ".sh"}:
                continue

            try:
                weight_dir = resolve_weight_dir(framework, config_path=Path(path))
            except WeightDirNotFoundError:
                logger.debug("No weight directory found in config %r; skipping", path)
                continue

            if cli_weight_dir is not None and cli_weight_dir != weight_dir:
                logger.error(
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
        name_to_path: Dict[str, str],
        weight_dir: Optional[str],
        rank_table_path: Optional[str],
    ) -> Collector:
        """
        Assemble a Collector with all applicable strategies.
        """
        strategies = [Env(), Sys(), Ascend()]

        for name, path in name_to_path.items():
            strategies.append(Config(name=name, config_path=path))

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
        framework = detect_framework()

        name_to_path = self._parse_config_fields(args.configs) if args.configs else {}

        weight_dir, has_conflict = self._resolve_weight_dir(
            framework, name_to_path, args.weight_dir
        )
        if has_conflict:
            return 1

        effective_weight_dir = weight_dir or args.weight_dir

        if is_in_container() and not CPUHighPerformance().execute():
            logger.error(
                "dump aborted: %s",
                CONTAINER_CPU_HIGH_PERF_AMBIGUITY_HINT,
            )
            print(CONTAINER_CPU_HIGH_PERF_AMBIGUITY_HINT)
            print(
                "本次未写入落盘文件，避免产生误导性数据；请按上述说明配置后重新执行 dump。"
            )
            return 1

        collector = self._build_collector(
            framework, name_to_path, effective_weight_dir, args.rank_table_path
        )

        try:
            data = collector.collect()
            with open(args.output_path, "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception:
            logger.exception("Error occurred while saving to %r", args.output_path)
            return 1

        print(f"All information has been saved in: {args.output_path!r}")
        print(
            "You may now use 'msprechecker compare' to compare dumped files for discrepancies!"
        )
        return 0
