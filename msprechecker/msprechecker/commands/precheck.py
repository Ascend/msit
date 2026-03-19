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
from pathlib import Path
from textwrap import dedent

from ..core.checker import has_errors, Severity
from ..core.runner import PrecheckRunner
from ..core.suite import build_suite
from ..utils import detect_framework

from . import CmdStrategy, CmdType


logger = logging.getLogger(__name__)


def setup_precheck(subparsers: argparse._SubParsersAction, parents=None):
    if parents is None:
        parents = []
    desc = dedent("""\
        PRECHECK - Run a series of validations for different PD deployment scenarios.

        Mix Mode (default):
          Validate environment, system, and HCCL settings.

        Disaggregation Mode:
          Validate only user_config.json and mindie_env.json.
    """)
    epilog = dedent("""\
        Examples:
          msprechecker precheck --rank-table-path hccl_8s_64p.json
          msprechecker precheck --user-config-path /cfg/user.json --mindie-env-path /cfg/mindie.json
    """)
    parser = subparsers.add_parser(
        CmdType.PRECHECK.value,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc,
        usage="msprechecker precheck [OPTIONS]",
        epilog=epilog,
        help="Run comprehensive system validation for PD deployment scenarios",
        parents=parents,
    )
    _add_disagg_args(parser)
    _add_mix_args(parser)
    _add_network_args(parser)
    _add_model_args(parser)
    _add_stress_args(parser)
    _add_output_args(parser)
    return parser


def _add_disagg_args(p):
    g = p.add_argument_group("PD Disaggregation Options")
    g.add_argument(
        "--scene",
        metavar="",
        help="Deploy scene: pd_disaggregation, mindie, vllm, vllm,ep ...",
    )
    g.add_argument(
        "--user-config-path",
        metavar="",
        help="Path to user_config.json for Kubernetes deployments.",
    )
    g.add_argument(
        "--mindie-env-path",
        metavar="",
        help="Path to mindie_env.json for Kubernetes deployments.",
    )
    g.add_argument(
        "--config-parent-dir",
        metavar="",
        help="Parent directory for Kubernetes deployments.",
    )


def _add_mix_args(p):
    g = p.add_argument_group("PD Mixed Mode Options")
    g.add_argument(
        "--mies-config-path",
        metavar="",
        help="Path to config.json for daemon-based deployments.",
    )


def _add_network_args(p):
    g = p.add_argument_group("Network Options")
    g.add_argument(
        "--rank-table-path",
        metavar="",
        help="Path to rank table file (A2 / A3 format).",
    )


def _add_model_args(p):
    g = p.add_argument_group("Model Options")
    g.add_argument(
        "--weight-dir", metavar="", help="Directory containing model weights."
    )


def _add_stress_args(p):
    g = p.add_argument_group("Stress Test Options")
    g.add_argument(
        "--hardware",
        action="store_true",
        default=False,
        help="Enable hardware stress testing.",
    )
    g.add_argument(
        "--threshold",
        type=int,
        choices=range(0, 101),
        default=20,
        metavar="0-100",
        help="Failure threshold percentage (0-100). Default: 20.",
    )


def _add_output_args(p):
    g = p.add_argument_group("Output Options")
    g.add_argument(
        "-s",
        "--severity-level",
        choices=["info", "warning", "error"],
        default="info",
        help="Minimum severity level to report. Default: info.",
    )


class Precheck(CmdStrategy):
    @staticmethod
    def _has_config_args(args: argparse.Namespace) -> bool:
        return any(
            [
                args.user_config_path,
                args.mindie_env_path,
                args.config_parent_dir,
                args.mies_config_path,
            ]
        )

    @staticmethod
    def _delegate_to_cmate(args: argparse.Namespace) -> int:
        from ..cmate.cmate import _parse_configs, _parse_contexts, run

        configs = []
        if args.user_config_path:
            configs.append(f"user_config:{args.user_config_path}@json")
        if args.mindie_env_path:
            configs.append(f"mindie_env:{args.mindie_env_path}@json")
        if args.mies_config_path:
            configs.append(f"config:{args.mies_config_path}@json")

        scene = args.scene or ""
        preset_name = "vllm" if "vllm" in scene.lower() else "mindie"
        preset_path = Path(__file__).parent.parent / "presets" / f"{preset_name}.cmate"

        if not preset_path.exists():
            logger.error("Preset file not found: %s", preset_path)
            return 1

        contexts = ["deploy_mode:ep"] if "ep" in scene.lower() else []

        ret, parsed_configs = _parse_configs(configs)
        if not ret:
            return 1
        ret, parsed_contexts = _parse_contexts(contexts)
        if not ret:
            return 1

        return run(
            str(preset_path),
            parsed_configs,
            parsed_contexts,
            failfast=False,
            verbosity=False,
            collect_only=False,
            output_path="",
            severity="info",
        )

    def execute(self, args: argparse.Namespace) -> int:
        if self._has_config_args(args):
            logger.warning(
                "Environment / Config validation via 'precheck' is deprecated and will be removed "
                "in May 2025. Please use 'msprechecker run' instead."
            )
            return self._delegate_to_cmate(args)

        framework = detect_framework()
        min_severity = Severity[args.severity_level.upper()]

        suite = build_suite(
            framework=framework,
            scene=args.scene or "",
            rank_table_path=args.rank_table_path or "",
            hardware=args.hardware,
            threshold=args.threshold,
        )

        result = PrecheckRunner(min_severity=min_severity).run(suite)
        return has_errors(result)
