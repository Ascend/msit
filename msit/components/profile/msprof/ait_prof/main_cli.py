# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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

import argparse
import re
from components.utils.parser import BaseCommand
from ait_prof.msprof_process import msprof_process
from ait_prof.args_adapter import MsProfArgsAdapter
from components.utils.file_open_check import FileStat


def check_output_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal("write"):
        raise argparse.ArgumentTypeError(f"output path:{path_value} is illegal. Please check.")
    return path_value


def check_application_string_legality(value):
    cmd_str = value
    regex = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{;: /.~-]")
    if regex.search(cmd_str):
        raise argparse.ArgumentTypeError(f"application string \"{cmd_str}\" is not a legal string")
    return cmd_str


class ProfileCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "--application",
            type=check_application_string_legality,
            required=True,
            help="Configure to run AI task files on the environment"
        )
        parser.add_argument(
            "-o",
            "--output",
            type=check_output_path_legality,
            default=None,
            help="The storage path for the collected profiling data,"
                " which defaults to the directory where the app is located"
        )
        parser.add_argument(
            "--model-execution",
            default="on",
            choices=["on", "off"],
            help="Control ge model execution performance data collection switch"
        )
        parser.add_argument(
            "--sys-hardware-mem",
            default="on",
            choices=["on", "off"],
            help="Control the read/write bandwidth data acquisition switch for ddr and llc"
        )
        parser.add_argument(
            "--sys-cpu-profiling",
            default="off",
            choices=["on", "off"],
            help="CPU acquisition switch"
        )
        parser.add_argument(
            "--sys-profiling",
            default="off",
            choices=["on", "off"],
            help="System CPU usage and system memory acquisition switch"
        )
        parser.add_argument(
            "--sys-pid-profiling",
            default="off",
            choices=["on", "off"],
            help="The CPU usage of the process and the memory collection switch of the process"
        )
        parser.add_argument(
            "--dvpp-profiling",
            default="on",
            choices=["on", "off"],
            help="DVPP acquisition switch"
        )
        parser.add_argument(
            "--runtime-api",
            default="on",
            choices=["on", "off"],
            help="Control runtime api performance data collection switch"
        )
        parser.add_argument(
            "--task-time",
            default="on",
            choices=["on", "off"],
            help="Control ts timeline performance data collection switch"
        )
        parser.add_argument(
            "--aicpu",
            default="on",
            choices=["on", "off"],
            help="Control aicpu performance data collection switch"
        )

    def handle(self, args):
        args_adapter = MsProfArgsAdapter(args.application, args.output, args.model_execution, args.sys_hardware_mem,
                                         args.sys_cpu_profiling, args.sys_profiling, args.sys_pid_profiling,
                                         args.dvpp_profiling, args.runtime_api, args.task_time, args.aicpu)
        return msprof_process(args_adapter)


def get_cmd_instance():
    help_info = "get profiling data of a given programma"
    cmd_instance = ProfileCommand("profile", help_info)
    return cmd_instance