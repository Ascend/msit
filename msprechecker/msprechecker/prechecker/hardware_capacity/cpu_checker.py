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
import time

import yaml
from msguard.security import open_s

from msprechecker.prechecker.register import PrecheckerBase, show_check_result, CheckResult
from msprechecker.prechecker.utils import logger, SimpleProgressBar
from msprechecker.prechecker.hardware_capacity.time_analyze import TimeAnalyze


class CPUChecker(PrecheckerBase):
    CHECK_TYPE = "cpu"

    @classmethod
    def cpu_matmul(cls, cpu_id):
        import torch
        # 读取矩阵参数
        env_check_dir = os.path.dirname(__file__)
        yaml_file = os.path.join(env_check_dir, "matmul_shape.yaml")
        with open_s(yaml_file, "r") as f:
            shape_dict = yaml.safe_load(f)
        batch_size = shape_dict["cpu_check"]["batch_size"]
        seq_len = shape_dict["cpu_check"]["seq_len"]
        hidden_size = shape_dict["cpu_check"]["hidden_size"]
        intermediate_size = shape_dict["cpu_check"]["intermediate_size"]

        # 执行多次矩阵运算：mat_c + mat_a × mat_b
        for _ in range(10):
            mat_a = torch.randn(batch_size, seq_len, hidden_size).to(f"cpu:{cpu_id}")
            mat_b = torch.randn(batch_size, hidden_size, intermediate_size).to(f"cpu:{cpu_id}")
            mat_c = torch.randn(seq_len, intermediate_size).to(f"cpu:{cpu_id}")
            torch.addbmm(mat_c, mat_a, mat_b)

    def collect_env(self, **kwargs):
        import torch

        cpu_ids = os.cpu_count()
        torch.set_num_threads(cpu_ids)

        output = {}
        for cpu_id in SimpleProgressBar(range(cpu_ids)):
            # 创建事件对象，用于记录运算的开始和结束时间
            
            start_time = time.time()
            self.cpu_matmul(cpu_id)
            end_time = time.time()
            cpu_time = (end_time - start_time) * 1000

            output[cpu_id] = cpu_time
        return output

    def do_precheck(self, envs: dict, **kwargs):
        time_all = envs

        cpu_analyze = TimeAnalyze(time_all)
        slow_cpu, slow_time, max_ratio, is_problem = cpu_analyze.time_analyze()

        if is_problem:
            action = f"检查cpu {slow_cpu} 状态"
            reason = f"cpu计算时长 {slow_time}ms 大于平均时长的 {round(max_ratio * 100)}%"
            show_check_result("hardware", "cpu_checker", CheckResult.ERROR, action=action, reason=reason)
        else:
            show_check_result("hardware", "cpu_checker", CheckResult.OK)


cpu_checker = CPUChecker()
