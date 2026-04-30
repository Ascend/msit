# -*- coding: utf-8 -*-
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
# -------------------------------------------------------------------------

import os
import argparse
from textwrap import dedent

from .commands import (
    setup_precheck_parser,
    setup_dump,
    setup_compare_parser,
    setup_cmate_parser,
    Coordinator
)
from .utils import global_logger


def main():
    if os.geteuid() == 0:
        global_logger.warning(
            'WARNING: Running as root is not suggested.\n\n'
            'This may lead to unexpected privilege escalation and system modifications.'
        )

    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent('''\
            MindStudio Pre-Checker Tool - A comprehensive validation tool for inference
        '''),
        usage='msprechecker [-h] [--version] {precheck,dump,compare} ...',
        epilog=dedent('''\
            Examples:
              msprechecker precheck                          # Run validations
              msprechecker dump --output-path baseline.json  # Create a snapshot of current context
              msprechecker compare old.json new.json         # Compare two snapshots

            For detailed help on each command, use: msprechecker <command> --help
        ''')
    )
    subparsers = main_parser.add_subparsers(dest="command", title="Available Commands", metavar="")

    setup_precheck_parser(subparsers)
    setup_dump(subparsers)
    setup_compare_parser(subparsers)
    setup_cmate_parser(subparsers)

    coordinator = Coordinator()
    return coordinator.execute(main_parser)
