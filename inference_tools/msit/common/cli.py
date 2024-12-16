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

from argparse import ArgumentParser, RawTextHelpFormatter

from msit.base.cmd import MsitCommand
from msit.common.constants import CmdConst
from msit.core.probe.cli.command_probe import ProbeCommand


class MainCommand(MsitCommand):
    def __init__(self):
        super().__init__(prog_name=CmdConst.MSIT)
        self.parser = ArgumentParser(description=CmdConst.DESCRIPTION, formatter_class=RawTextHelpFormatter)
        self.subparsers = self.parser.add_subparsers(dest=CmdConst.COMMAND)
        self.add_subcommand_level = CmdConst.CMD_LEVEL_2

    def add_arguments(self):
        if self.input_keyword == CmdConst.PROBE:
            self.add_subcommand(ProbeCommand())
        elif self.input_keyword == CmdConst.SURGEON:
            pass
        else:
            self.subparsers.add_parser(CmdConst.PROBE, help=CmdConst.HELP_PROBE)
            self.subparsers.add_parser(CmdConst.SURGEON, help=CmdConst.HELP_SURGEON)

    def parse_args(self):
        return self.parser.parse_args()
