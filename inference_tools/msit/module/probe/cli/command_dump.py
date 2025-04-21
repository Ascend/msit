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

from msit.base import Command, MsitCommand
from msit.common.validation import CheckConfigPath, CheckExec, CheckFramework
from msit.module.probe.cli.command_probe import ProbeCommand
from msit.utils.constants import CfgConst, CmdConst, PathConst


@Command.register(ProbeCommand, CmdConst.DUMP)
class DumpCommand(MsitCommand):
    @staticmethod
    def add_required_arguments(parser):
        req = parser.add_argument_group("Required arguments")
        req.add_argument(
            "-e",
            "--exec",
            dest=CfgConst.EXEC,
            action=CheckExec,
            required=True,
            help=f"""<str> Supports two input types:
        1. An offline model file with {("saved_model",) + PathConst.SUFFIX_OFFLINE_MODEL} extension;
        2. An executable CLI scripts enclosed in quotes end with {PathConst.SUFFIX_ONLINE_SCRIPT}. Default: None""",
        )
        req.add_argument(
            "-cfg",
            "--config",
            dest=CfgConst.CONFIG_PATH,
            action=CheckConfigPath,
            required=True,
            help="""<str> A config JSON file for storing data dump settings. Default: None""",
        )

    @staticmethod
    def add_optional_arguments(parser):
        opt = parser.add_argument_group("Optional arguments")
        opt.add_argument(
            "-f",
            "--framework",
            dest=CfgConst.FRAMEWORK,
            action=CheckFramework,
            help=f"""<str> Required when using: {CfgConst.ALL_FRAMEWORK}. Default: None""",
        )
        opt.add_argument(
            "-x",
            "--msitx",
            dest="msitx",
            default=False,
            action="store_true",
            help="""<bool> Use MSIT extended API. Default: False""",
        )

    @classmethod
    def add_arguments(cls, parser):
        cls.add_required_arguments(parser)
        cls.add_optional_arguments(parser)
