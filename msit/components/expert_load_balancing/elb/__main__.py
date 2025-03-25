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
import os
import subprocess

from components.utils.parser import BaseCommand
from components.utils.log import logger
from elb.utils import check_path_legality, get_algorithm_path


class ExpertLoadBalanceCommmand(BaseCommand):
    
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            '--info-csv-path',
            '-isp',
            dest="expert_popularity_csv_load_path",
            required=True,
            type=check_path_legality,
            help=".")

        parser.add_argument(
            '--output-dir',
            '-o',
            dest="output_dir",
            type=check_path_legality,
            default='./',
            help="Data output directory. E.g: '--output /xx/xxxx/xx', default=./output")

        parser.add_argument(
            '--num-redundant-expert',
            '-nre',
            dest="num_redundancy_expert",
            type=int,
            required=False,
            default=64,
            help="Number of redundant experts.")

        parser.add_argument(
            '--num-nodes',
            '-nd',
            dest="num_nodes",
            type=int,
            required=False,
            default=8,
            help="Number of nodes.")
        
        parser.add_argument(
            '--num-npus',
            '-nn',
            dest="num_npus",
            type=int,
            required=False,
            default=64,
            help="Number of npu.")

        parser.add_argument(
            '--algorithm',
            '-al',
            dest="algorithm",
            type=str,
            required=False,
            default="1",
            choices=['0', '1'],
            help="algorithm type.")

    def handle(self, args, **kwargs) -> None:
        logger.info("===================load balancing algorithm start====================")
        get_algorithm_path()
        from elb.load_balancing import load_balancing
        load_balancing(args)
        logger.info("===================load balancing algorithm end====================")


def get_cmd_instance():
    help_info = "Large Language Model(llm) Debugger Tools."
    expert_load_balancing_cmd_instance = ExpertLoadBalanceCommmand("expert-load-balancing", help_info)

    return expert_load_balancing_cmd_instance