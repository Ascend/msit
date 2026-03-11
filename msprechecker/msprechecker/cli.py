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


def main() -> int:
    from .util import LOG_FORMAT, LOG_LEVELS

    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "--log-level",
        "-l",
        choices=LOG_LEVELS,
        default="info",
        help="Set the logging level.",
    )

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

    from .commands.cmate import setup_cmate
    from .commands.compare import setup_compare
    from .commands.dump import setup_dump
    from .commands.precheck import setup_precheck

    setup_precheck(subparsers, [global_parser])
    setup_dump(subparsers, [global_parser])
    setup_compare(subparsers, [global_parser])
    setup_cmate(subparsers, [global_parser])

    args = main_parser.parse_args()
    from .commands.banner import BannerPresenter

    BannerPresenter().print_banner()
    cmd = getattr(args, "command", None)
    if not cmd:
        main_parser.print_help()
        return 1

    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVELS[args.log_level])
    from .commands import CmdStrategyFactory, CmdType

    strategy_factory = CmdStrategyFactory()
    args.command = CmdType(cmd)
    strategy = strategy_factory.get(args.command)
    return strategy.execute(args)
