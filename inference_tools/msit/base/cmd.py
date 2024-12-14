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

from sys import argv
from abc import ABC, abstractmethod
from argparse import RawTextHelpFormatter

from msit.common.constants import MsgConst
from msit.common.exceptions import MsitException


class MsitCommand(ABC):
    def __init__(self, prog_name=None, help_info=None):
        self.prog_name = prog_name
        self.help_info = help_info
        self.add_subcommand_level = 0
        self.subparsers = None

    @property
    def input_keyword(self):
        if isinstance(self.add_subcommand_level, int) and self.add_subcommand_level > 0:
            return argv[self.add_subcommand_level] if len(argv) > self.add_subcommand_level else None
        else:
            raise MsitException(MsgConst.INVALID_ARGU, "Subcommand level must be a positive integer.")

    def add_subcommand(self, sub_command):
        if not isinstance(sub_command, MsitCommand):
            raise MsitException(MsgConst.INVALID_DATA_TYPE, "Sub-command must be an instance of MsitCommand.")
        if not self.subparsers:
            raise MsitException(MsgConst.INVALID_ARGU, "Subparsers not initialized. Call add_subparsers() first.")
        parser = self.subparsers.add_parser(sub_command.prog_name, help=sub_command.help_info, \
                                            formatter_class=RawTextHelpFormatter)
        sub_command.add_arguments(parser)

    @abstractmethod
    def add_arguments(self, parser):
        pass

    def execute(self, args):
        pass
