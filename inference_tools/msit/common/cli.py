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

from argparse import ArgumentParser
from sys import argv

from msit.base import Command, MsitCommand, Service
from msit.common.ascend import cann
from msit.utils.constants import CfgConst, CmdConst, MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.log import logger
from msit.utils.toolkits import run_subprocess, set_ld_preload

_DESCRIPTION = """
                                   _ _
                     _ __ ___  ___(_) |_ 
                    | '_ ` _ \/ __| | __|
                    | | | | | \__ \ | |_ 
                    |_| |_| |_|___/_|\__|

msit (MindStudio Inference Tools), [Powered by MindStudio].
Providing one-site debugging and optimization toolkits for inference on Ascend devices.
For any issue, refer FAQ first. Gitee repo: Ascend/msit, wiki.
"""
_L2COMMAND = "L2command"


class MainCommand(MsitCommand):
    def __init__(self):
        super().__init__()
        self.parser = ArgumentParser(prog="msit", description=_DESCRIPTION, formatter_class=self.formatter_class)
        self.subparser = self.parser.add_subparsers(dest=_L2COMMAND)
        self.second_commands = Command.get("msit")
        self.subcommand_level = 1

    def add_arguments(self, parse):
        pass

    def register(self):
        for name, cmd_class in self.second_commands.items():
            cmd_parser = self.subparser.add_parser(
                name=name, help=CmdConst.HELP_MODULE_MAP.get(name), formatter_class=self.formatter_class
            )
            if self.input_module in self.second_commands:
                cmd_class.add_arguments(cmd_parser)
                self.subcommand_level = 2
                self.build_parser(cmd_parser, cmd_class)

    def parse(self):
        return self.parser.parse_args()

    def set_env(self, framework):
        env_map = {CfgConst.FRAMEWORK_MINDIE_LLM: cann.get_atb_probe_so_path()}
        so_path = env_map.get(framework)
        if so_path:
            set_ld_preload(so_path)
        else:
            raise MsitException(MsgConst.PATH_NOT_FOUND, f".so library path for {framework} not found.")

    def execute(self, args):
        if Service.get(argv[self.subcommand_level - 1]):
            logger.info(f"Preparing to launch {argv[self.subcommand_level - 1]} service.")
            if args.framework:
                self.set_env(args.framework)
            if not args.msitx:
                Service.get(argv[self.subcommand_level - 1])(args=args).run_cli()
            else:
                run_subprocess(args.exec)
        else:
            raise MsitException(
                MsgConst.CALL_FAILED,
                f"The {argv[self.subcommand_level - 1]} utility is not registered. Please check it.",
            )
