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

import argparse
import logging

from msguard import Rule, validate_args

from . import CmdStrategy


logger = logging.getLogger(__name__)


def setup_cmate(subparsers: argparse._SubParsersAction, parents=None):
    if parents is None:
        parents = []
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        parents=parents,
        help="Execute configuration validation against specified rules",
        description="Run comprehensive validation of configurations using the specified rule set.",
    )

    run_parser.add_argument(
        "rule",
        type=validate_args(Rule.input_file_read),
        help="Path to the rule definition file (CMATE format)",
    )

    run_parser.add_argument(
        "--configs",
        "-c",
        nargs="*",
        help=(
            "Configuration files to validate, specified as '<name>:<path>' or '<name>:<path>@<parse-type>'.\n"
            "  - <name>: Configuration identifier defined in the rule file\n"
            "  - <path>: File system path to the configuration file\n"
            "  - <parse-type>: (Optional) Parsing method ('json', 'yaml', 'yml')\n"
            "Note: For 'env' type targets, the <path> component is optional and will be discarded if provided."
        ),
    )

    run_parser.add_argument(
        "--contexts",
        "-C",
        nargs="*",
        help=(
            "Context variables required for rule execution, specified as '<name>:<value>'.\n"
            "  - <name>: Context identifier defined in the rule file\n"
            "  - <value>: Value to assign to the context variable\n"
            "Note: Unquoted numbers are parsed as integers. Use quotes for string values: '2'."
        ),
    )

    run_parser.add_argument(
        "-co",
        "--collect-only",
        action="store_true",
        help="Display the list of rules that would be executed without actually running them",
    )

    run_parser.add_argument(
        "--output-path", help="Path to the save msprechecker output."
    )

    run_parser.add_argument(
        "-x",
        "--fail-fast",
        action="store_true",
        dest="failfast",
        help="Stop execution immediately upon encountering the first failure or error",
    )

    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable verbose output, displaying detailed test names and individual results",
    )

    run_parser.add_argument(
        "-s",
        "--severity",
        choices=["info", "warning", "error"],
        default="info",
        help=(
            "Minimum severity level for rule execution:\n"
            "  - info: Execute all checks (default)\n"
            "  - warning: Execute only warning and error checks\n"
            "  - error: Execute only error checks"
        ),
    )

    # Inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        parents=parents,
        help="Display detailed information about rule requirements",
        description="Inspect and display the configuration targets, contexts, and requirements defined in a rule file.",
    )

    inspect_parser.add_argument(
        "rule",
        type=validate_args(Rule.input_file_read),
        help="Path to the rule definition file (CMATE format)",
    )

    inspect_parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json"],
        default="text",
        help=(
            "Output format for the inspection results:\n"
            "  - text: Human-readable text format (default)\n"
            "  - json: Structured JSON format for programmatic processing"
        ),
    )


class Run(CmdStrategy):
    @staticmethod
    def execute(args):
        from cmate.cmate import _parse_configs, _parse_contexts, run

        from .banner import BannerPresenter

        BannerPresenter().print_banner()

        ret, configs = _parse_configs(args.configs)
        if not ret:
            return 1

        ret, contexts = _parse_contexts(args.contexts)
        if not ret:
            return 1

        return run(
            args.rule,
            configs,
            contexts,
            args.failfast,
            args.verbose,
            args.collect_only,
            args.output_path,
            args.severity,
        )


class Inspect(CmdStrategy):
    @staticmethod
    def execute(args):
        from cmate import inspect

        return inspect(args.rule, args.format)
