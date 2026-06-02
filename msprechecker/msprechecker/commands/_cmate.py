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

# pylint: disable=duplicate-code

from ..utils.path_io import normalize_user_path
from ..utils.path_io import readable_file


def setup_cmate_parser(subparsers):
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Execute configuration validation against specified rules",
        description="Run comprehensive validation of configurations using the specified rule set.",
    )

    run_parser.add_argument("rule", type=readable_file, help="Path to the rule definition file (CMATE format)")

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

    run_parser.add_argument("--output-path", type=normalize_user_path, help="Path to the save msprechecker output.")

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
        help="Display detailed information about rule requirements",
        description="Inspect and display the configuration targets, contexts, and requirements defined in a rule file.",
    )

    inspect_parser.add_argument("rule", type=readable_file, help="Path to the rule definition file (CMATE format)")

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
