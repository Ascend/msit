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

import argparse
import logging
from textwrap import dedent

from .commands import (
    CmdStrategyFactory,
    CmdType,
    setup_compare_parser,
    setup_dump_parser,
    setup_precheck_parser,
)
from .utils.constant import LOG_FORMAT, LOG_LEVELS



def main() -> int:
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent("""\
            MindStudio Pre-Checker Tool - A comprehensive validation tool for inference
        """),
        usage="msprechecker [-h] [--version] {precheck,dump,compare} ...",
        epilog=dedent("""\
            Examples:
              msprechecker precheck                          # Run validations
              msprechecker dump --output-path baseline.json  # Create a snapshot of current context
              msprechecker compare old.json new.json         # Compare two snapshots

            For detailed help on each command, use: msprechecker <command> --help
        """),
    )
    subparsers = main_parser.add_subparsers(
        dest="command", title="Available Commands", metavar=""
    )

    setup_precheck_parser(subparsers)
    setup_dump_parser(subparsers)
    setup_compare_parser(subparsers)

    args = main_parser.parse_args()
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level.lower()],
        format=LOG_FORMAT,
    )

    cmd = getattr(args, "command", None)
    if not cmd:
        main_parser.print_help()
        return 1

    strategy_factory = CmdStrategyFactory()
    args.command = CmdType(cmd)
    strategy = strategy_factory.get(args.command)
    return strategy.execute(args)
