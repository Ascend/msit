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

from msit.base.cmd import MsitCommand
from msit.core.probe.cli import DumpCommand
from msit.utils.constants import CmdConst


class ProbeCommand(MsitCommand):
    def __init__(self):
        super().__init__(prog_name=CmdConst.PROBE, help_info=CmdConst.HELP_PROBE)
        self.add_subcommand_level = CmdConst.CMD_LEVEL_3
        self.match_command_map = {CmdConst.DUMP: DumpCommand()}
        self.name_command_help = [
            [CmdConst.DUMP, CmdConst.HELP_PROBE_DUMP],
            [CmdConst.COMPARE, CmdConst.HELP_PROBE_COMPARE],
            [CmdConst.OPCHECK, CmdConst.HELP_PROBE_OPCHECK],
        ]

    def register(self, subparser):
        probe_parser = subparser.add_parser(self.prog_name, help=self.help_info, formatter_class=self.formatter_class)
        self.subparser = probe_parser.add_subparsers(dest=CmdConst.L3COMMAND)
        super().register()
