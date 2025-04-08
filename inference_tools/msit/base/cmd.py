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

from abc import ABC, abstractmethod
from argparse import RawTextHelpFormatter
from sys import argv

from msit.utils.constants import CmdConst, MsgConst
from msit.utils.exceptions import MsitException


class Command:
    """
    A hierarchical command registration system that supports multi-level command structures.
    """

    _cmd_map = {}  # Internal storage: {parent_cmd: {name: command_class}}

    @classmethod
    def register(cls, parent_cmd, name):
        def decorator(command_cls):
            if parent_cmd not in cls._cmd_map:
                cls._cmd_map[parent_cmd] = {}
            cls._cmd_map[parent_cmd][name] = command_cls
            return command_cls

        return decorator

    @classmethod
    def get(cls, parent_cmd):
        return cls._cmd_map.get(parent_cmd, {})


class MsitCommand(ABC):
    def __init__(self):
        self.formatter_class = RawTextHelpFormatter

    @property
    def input_module(self):
        if isinstance(self.subcommand_level, int) and self.subcommand_level > 0:
            return argv[self.subcommand_level] if len(argv) > self.subcommand_level else None
        else:
            raise MsitException(MsgConst.INVALID_ARGU, "Subcommand level must be a positive integer.")

    @abstractmethod
    def add_arguments(self, parse):
        pass

    def build_parser(self, parent_parser, parent_cmd_class):
        if self.subcommand_level > MsgConst.MAX_RECURSION_DEPTH:
            raise MsitException(
                MsgConst.RISK_ALERT, f"Maximum recursion depth of {MsgConst.MAX_RECURSION_DEPTH} exceeded."
            )
        subcommands = Command.get(parent_cmd_class)
        if subcommands:
            self.subcommand_level += 1
            subparsers = parent_parser.add_subparsers(dest=f"L{self.subcommand_level}command")
            for name, cmd_class in subcommands.items():
                cmd_parser = subparsers.add_parser(
                    name=name, help=CmdConst.HELP_TOOL_MAP.get(name), formatter_class=self.formatter_class
                )
                cmd_class.add_arguments(cmd_parser)
                self.build_parser(cmd_parser, cmd_class)
