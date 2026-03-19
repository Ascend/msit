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
import sys
import traceback
import warnings
from textwrap import dedent

from .commands import CmdStrategyFactory, CmdType
from .commands.cmate import setup_cmate
from .commands.compare import setup_compare
from .commands.dump import setup_dump
from .commands.sync import setup_sync
from .commands.precheck import setup_precheck
from .utils import LOGGER, CustomError, LOG_LEVELS

warnings.filterwarnings("ignore")


def main() -> int:
    global_parser = argparse.ArgumentParser(add_help=False)

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
              msprechecker sync baseline.json                # Replicate snapshots from baseline.json

            For detailed help on each command, use: msprechecker <command> --help
        """),
    )
    main_parser.add_argument('-l', '--log-level', type=int, choices=range(0, 5), default=1,
                             help="Specify the print log level, "
                                  "0(debug) | 1(info) | 2(warning) | 3(error) | 4(critical), default is 1(info).")
    subparsers = main_parser.add_subparsers(
        dest="command", title="Available Commands", metavar=""
    )

    try:
        setup_precheck(subparsers, [global_parser])
        setup_dump(subparsers, [global_parser])
        setup_compare(subparsers, [global_parser])
        setup_sync(subparsers, [global_parser])
        setup_cmate(subparsers, [global_parser])
        args = main_parser.parse_args()

        LOGGER.setLevel(LOG_LEVELS[args.log_level])

        cmd = getattr(args, "command", None)
        if not cmd:
            main_parser.print_help()
            return 1
        strategy_factory = CmdStrategyFactory()
        args.command = CmdType(cmd)
        strategy = strategy_factory.get(args.command)
        return strategy.execute(args)
    except ValueError as e:
        LOGGER.error(traceback.format_exc())
        LOGGER.error(e)
        sys.exit(2)
    except CustomError as e:
        if hasattr(e, 'code'):
            sys.exit(int(e.code))
        sys.exit(0)
    except Exception as e:
        LOGGER.error(e)
        LOGGER.error(traceback.format_exc())
