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

from argparse import RawTextHelpFormatter
from sys import argv

from msit.utils.constants import MsgConst
from msit.utils.exceptions import MsitException


class MsitCommand:
    def __init__(self, prog_name=None, help_info=None):
        self.prog_name = prog_name
        self.help_info = help_info
        self.add_subcommand_level = 0
        self.subparser = None
        self.formatter_class = RawTextHelpFormatter
        self.match_command_map = {}
        self.name_command_help = []

    @property
    def input_keyword(self):
        if isinstance(self.add_subcommand_level, int) and self.add_subcommand_level > 0:
            return argv[self.add_subcommand_level] if len(argv) > self.add_subcommand_level else None
        else:
            raise MsitException(MsgConst.INVALID_ARGU, "Subcommand level must be a positive integer.")

    def register(self):
        if self.input_keyword in self.match_command_map:
            self.match_command_map[self.input_keyword].register(self.subparser)
        else:
            for cmd_name, cmd_help in self.name_command_help:
                self.subparser.add_parser(cmd_name, help=cmd_help)

    def execute(self, args):
        pass
