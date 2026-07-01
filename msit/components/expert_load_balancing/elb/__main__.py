# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import argparse
import os

from components.utils.file_open_check import FileStat
from components.utils.file_utils import (
    check_input_dir_path,
    check_input_file_path,
    check_output_dir_path,
    check_path_no_group_others_write,
)
from components.utils.log import logger
from components.utils.parser import BaseCommand
from components.utils.security_check import check_positive_integer, check_positive_or_zero_integer


def check_output_path_legality(value):
    if not value:
        return value
    path_value = value
    check_output_dir_path(path_value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"output path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal("write", strict_permission=False):
        raise argparse.ArgumentTypeError(f"output path:{path_value} is illegal. Please check.")
    return path_value


def check_input_path_legality(value):
    if not value:
        return value
    inputs_list = value.split(',')
    for input_path in inputs_list:
        if os.path.isfile(input_path):
            check_input_file_path(input_path)
        else:
            check_input_dir_path(input_path)
        try:
            file_stat = FileStat(input_path)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.") from err
        if not file_stat.is_basically_legal('read'):
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.")
        check_path_no_group_others_write(input_path)
    return value


class ExpertLoadBalanceCommmand(BaseCommand):
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            '--info-csv-path',
            '-icp',
            dest="expert_popularity_csv_load_path",
            required=True,
            type=check_input_path_legality,
            help="Data input directory. Contains  CSV fileswhich might have been generated during prefill or decoder.",
        )

        parser.add_argument(
            '--output-dir',
            '-o',
            dest="output_dir",
            type=check_output_path_legality,
            default='./',
            help="Data output directory. E.g: '--output /xxx_path', default=./",
        )

        parser.add_argument(
            '--num-redundant-expert',
            '-nre',
            dest="num_redundancy_expert",
            type=check_positive_integer,
            required=False,
            default=64,
            help="Number of redundant experts.",
        )

        parser.add_argument(
            '--num-share-expert-devices',
            '-nsed',
            dest="share_expert_devices",
            type=check_positive_integer,
            required=False,
            default=0,
            help="Number of shared experts.",
        )

        parser.add_argument(
            '--num-nodes',
            '-n' + 'd',
            dest="num_nodes",
            type=check_positive_or_zero_integer,
            required=False,
            default=8,
            help="Number of nodes.",
        )

        parser.add_argument(
            '--num-npus',
            '-nn',
            dest="num_npus",
            type=check_positive_or_zero_integer,
            required=False,
            default=64,
            help="Number of npu.",
        )

        parser.add_argument(
            '--algorithm',
            '-al',
            dest="algorithm",
            type=str,
            required=False,
            default="3",
            choices=['0', '1', '2', '3', '4', '5'],
            help="algorithm type. 0代表计算通信负载均衡算法(C2LB), 1代表speculative moe level 1算法,"
            "2 代表生成动态场景下的C2LB算法生成初始配置文件, 3代表增强型的speculative moe level 2算法,"
            "4 代表speculative moe level 1混置算法, 5 代表speculative moe level 2 混置算法。",
        )

        parser.add_argument(
            '--device-type',
            '-dt',
            dest="device_type",
            type=str,
            required=True,
            choices=['a2', 'a3'],
            help="device type. a2 代表适用于Atlas 800I A2推理服务器, a3 代表适用于Atlas 800I A3推理服务器。",
        )

    def handle(self, args, **kwargs) -> None:
        if os.name != "nt" and os.getuid() == 0:
            logger.warning(
                "Security Warning: Do not run this tool as root. "
                "Running with privileges may compromise system security. "
                "Use a regular account."
            )

        try:
            from elb.eplb_runner import load_balancing
        except ImportError as err:
            raise ImportError("Failed to import load_balancing module") from err

        logger.info("===================load balancing algorithm start====================")
        load_balancing(args)
        logger.info("===================load balancing algorithm end====================")


def get_cmd_instance():
    help_info = "Large Language Model(llm) Debugger Tools."
    expert_load_balancing_cmd_instance = ExpertLoadBalanceCommmand("expert-load-balancing", help_info)

    return expert_load_balancing_cmd_instance
