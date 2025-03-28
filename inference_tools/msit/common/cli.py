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

from msit.base import MsitCommand, Service
from msit.core.probe.cli.command_probe import ProbeCommand
from msit.utils.constants import CmdConst, DumpConst, MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.log import logger

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
_CMD_LEVEL_2 = 1
_L2COMMAND = "L2command"


class MainCommand(MsitCommand):
    def __init__(self):
        super().__init__(prog_name="msit")
        self.parser = ArgumentParser(description=_DESCRIPTION, formatter_class=self.formatter_class)
        self.subparser = self.parser.add_subparsers(dest=_L2COMMAND)
        self.add_subcommand_level = _CMD_LEVEL_2
        self.match_command_map = {CmdConst.PROBE: ProbeCommand()}
        self.name_command_help = [[CmdConst.PROBE, CmdConst.HELP_PROBE], [CmdConst.SURGEON, CmdConst.HELP_SURGEON]]
        self.command_mapping = {"PROBE_DUMP_STATISTICS": DumpConst.STATISTICS, "PROBE_DUMP_TENSOR": DumpConst.TENSOR}

    def execute(self, args):
        if hasattr(args, DumpConst.LOG_LEVEL):
            logger.set_level(args.log_level)
        else:
            self.parser.print_help()
            return
        command_key = f"{args.L2command}_{args.L3command}_{args.task}".upper()
        service_key = self.command_mapping.get(command_key)
        if service_key:
            Service.get(service_key)(args=args).run_cli()
        else:
            MsitException(MsgConst.CALL_FAILED, f"Unsupported command key: {command_key}.")

    def parse_args(self):
        return self.parser.parse_args()
