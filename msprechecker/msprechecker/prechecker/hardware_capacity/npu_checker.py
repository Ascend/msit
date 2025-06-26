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

import yaml
from msguard.security import open_s

from msprechecker.prechecker.register import PrecheckerBase, show_check_result, CheckResult
from msprechecker.prechecker.utils import logger, SimpleProgressBar
from msprechecker.prechecker.hardware_capacity.time_analyze import TimeAnalyze


class NPUChecker(PrecheckerBase):

    @classmethod
    def npu_matmul(cls, npu_id):
        import torch

        # 读取矩阵参数
        env_check_dir = os.path.dirname(__file__)
        yaml_file = os.path.join(env_check_dir, "matmul_shape.yaml")
        with open_s(yaml_file, "r") as f:
            shape_dict = yaml.safe_load(f)
        batch_size = shape_dict["npu_check"]["batch_size"]
        seq_len = shape_dict["npu_check"]["seq_len"]
        hidden_size = shape_dict["npu_check"]["hidden_size"]
        intermediate_size = shape_dict["npu_check"]["intermediate_size"]

        # 执行多次矩阵运算：mat_c + mat_a × mat_b
        for _ in range(10):
            mat_a = torch.randn(batch_size, seq_len, hidden_size).to(f"npu:{npu_id}")
            mat_b = torch.randn(batch_size, hidden_size, intermediate_size).to(f"npu:{npu_id}")
            mat_c = torch.randn(seq_len, intermediate_size).to(f"npu:{npu_id}")
            torch.addbmm(mat_c, mat_a, mat_b)

    def collect_env(self, **kwargs):
        try:
            import torch_npu
        except ImportError:
            logger.warning("torch_npu not available, skipping NPUChecker")
            torch_npu = None

        output = {}
        if torch_npu is None:
            return output
        device_ids = torch_npu.npu.device_count()
        for device_id in SimpleProgressBar(range(device_ids)):
            # 创建事件对象，用于记录运算的开始和结束时间
            start_event = torch_npu.npu.Event(enable_timing=True)
            end_event = torch_npu.npu.Event(enable_timing=True)

            start_event.record()
            self.npu_matmul(device_id)
            end_event.record()

            # 同步当前流，确保全部运算均已完成
            torch_npu.npu.current_stream().synchronize()
            npu_time = start_event.elapsed_time(end_event)
            output[device_id] = npu_time

        return output

    def do_precheck(self, envs: str, **kwargs):
        if not envs:
            logger.warning("No NPUs available for hardware checking")
            return

        time_all = envs
        
        npu_analyze = TimeAnalyze(time_all)
        slow_rank, slow_time, max_ratio, is_problem = npu_analyze.time_analyze()

        if is_problem:
            action = f"检查npu {slow_rank} 状态"
            reason = f"npu计算时长 {slow_time}ms 大于平均时长的 {round(max_ratio * 100)}%"
            show_check_result("hardware", "npu_checker", CheckResult.ERROR, action=action, reason=reason)
        else:
            show_check_result("hardware", "npu_checker", CheckResult.OK)


npu_checker = NPUChecker()
