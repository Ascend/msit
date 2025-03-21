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

import torch
import torch_npu
import yaml

from ms_performance_prechecker.prechecker.register import RrecheckerBase
from ms_performance_prechecker.prechecker.utils import logger
from ms_performance_prechecker.prechecker.hardware_capacity.time_analyze import TimeAnalyze


class CPUChecker(RrecheckerBase):
    CHECK_TYPE = "cpu"

    @classmethod
    def cpu_matmul(cls, cpu_id):
        # 读取矩阵参数
        env_check_dir = os.path.dirname(__file__)
        yaml_file = os.path.join(env_check_dir, 'matmul_shape.yaml')
        with open(yaml_file, 'r') as f:
            shape_dict = yaml.safe_load(f)
        batch_size = shape_dict["cpu_check"]['batch_size']
        seq_len = shape_dict["cpu_check"]['seq_len']
        hidden_size = shape_dict["cpu_check"]['hidden_size']
        intermediate_size = shape_dict["cpu_check"]['intermediate_size']

        # 执行多次矩阵运算：mat_c + mat_a × mat_b
        for _ in range(10):
            mat_a = torch.randn(batch_size, seq_len, hidden_size).to(f'cpu:{cpu_id}')
            mat_b = torch.randn(batch_size, hidden_size, intermediate_size).to(f'cpu:{cpu_id}')
            mat_c = torch.randn(seq_len, intermediate_size).to(f'cpu:{cpu_id}')
            torch.addbmm(mat_c, mat_a, mat_b)

    def collect_env(self, **kwargs):
        cpu_ids = os.cpu_count()
        torch.set_num_threads(cpu_ids)

        output = {}
        for cpu_id in range(cpu_ids):
            # 创建事件对象，用于记录运算的开始和结束时间
            start_event = torch_npu.npu.Event(enable_timing=True)
            end_event = torch_npu.npu.Event(enable_timing=True)

            start_event.record()
            self.cpu_matmul(cpu_id)
            end_event.record()

            cpu_time = start_event.elapsed_time(end_event)
            output[cpu_id] = cpu_time
        return output

    def do_precheck(self, envs: dict, **kwargs):
        time_all = envs

        cpu_analyze = TimeAnalyze(time_all)
        slow_cpu, slow_time, max_ratio, is_problem = cpu_analyze.time_analyze()

        if is_problem:
            logger.info(f"The CPU: {slow_cpu} may have performance issues, its calculation time is {slow_time}ms, "
                        f"the relative difference from the average calculation time is {round(max_ratio * 100)}%. "
                        f"It is recommended to check and optimize the CPU performance on the server.")
        else:
            logger.info(f"CPUs are working well, no performance issues found.")


cpu_checker = CPUChecker()
