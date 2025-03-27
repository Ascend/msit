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
import os
import sys
import pwd
import subprocess

from components.utils.parser import BaseCommand
from components.utils.log import logger
from components.debug.compare.msquickcmp.common.args_check import check_output_path_legality, check_input_path_legality


class ExpertLoadBalanceCommmand(BaseCommand):
    
    def add_arguments(self, parser, **kwargs) -> None:
        parser.add_argument(
            '--info-csv-path',
            '-isp',
            dest="expert_popularity_csv_load_path",
            required=True,
            type=check_input_path_legality,
            help="Data input directory. Contains  CSV files"
            "which might have been generated during prefill or decoder.")

        parser.add_argument(
            '--output-dir',
            '-o',
            dest="output_dir",
            type=check_output_path_legality,
            default='./',
            help="Data output directory. E.g: '--output /xxx_path', default=./")

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
            help="algorithm type. 0代表计算通信负载均衡算法(C2LB), 1代表speculative moe interface algorithm。")

    def handle(self, args, **kwargs) -> None:
        logger.info("===================load balancing algorithm start====================")
        cann_path = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
        if not os.path.exists(cann_path):
            logger.error("CANN toolkit path does not exist. Please check your environment variables.")
            raise FileNotFoundError(f"CANN toolkit path does not exist: {cann_path}")
        
        algorithm_path = os.path.join(cann_path, "tools", "operator_cmp", "load_balancing")
        if not os.path.exists(algorithm_path):
            logger.error("Algorithm path does not exist. Please verify the installation.")
            raise FileNotFoundError(f"Algorithm path does not exist: {algorithm_path}")

        # 检查路径属主和权限
        stat_info = os.stat(algorithm_path)
        owner_uid = stat_info.st_uid
        permissions = stat_info.st_mode & 0o777  # 获取权限位

        current_uid = os.getuid()
        current_user = pwd.getpwuid(current_uid).pw_name
        path_owner = pwd.getpwuid(owner_uid).pw_name

        if owner_uid != current_uid:
            logger.warning(
                "Algorithm path is owned by another user"
                "instead of the current user. This may cause permission issues."
            )
        elif permissions > 0o750:
            logger.warning(
                "Algorithm path has unsafe permissions. "
                "Recommended permissions are <= 750."
            )

        sys.path.append(algorithm_path)
        try:
            from elb.load_balancing import load_balancing
        except ImportError as e:
            raise Exception("Failed to import load_balancing module") from e

        load_balancing(args)
        logger.info("===================load balancing algorithm end====================")


def get_cmd_instance():
    help_info = "Large Language Model(llm) Debugger Tools."
    expert_load_balancing_cmd_instance = ExpertLoadBalanceCommmand("expert-load-balancing", help_info)

    return expert_load_balancing_cmd_instance