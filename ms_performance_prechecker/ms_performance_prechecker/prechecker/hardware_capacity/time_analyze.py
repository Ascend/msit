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
from ms_performance_prechecker.prechecker.utils import logger


class TimeAnalyze:
    RATIO_THRESHOLD = 0.05

    def __init__(self, run_time):
        self.run_time = run_time

    def time_analyze(self):
        if not self.run_time:
            logger.error("Running time is undefined.")
            return None

        time_list = list(self.run_time.values())

        # 耗时极值编号和数据
        slow_time = max(time_list)
        slow_rank = max(self.run_time, key=self.run_time.get)

        # 计算快慢差异
        try:
            mean_time = sum(time_list) / len(time_list)
        except ZeroDivisionError as e:
            raise RuntimeError("The input parameter is undefined.") from e

        try:
            max_ratio = (slow_time - mean_time) / mean_time
        except ZeroDivisionError as e:
            raise RuntimeError("The input parameter has value of zero.") from e

        # 判断是否存在问题
        if max_ratio > self.RATIO_THRESHOLD:
            is_problem = True
        else:
            is_problem = False

        return slow_rank, slow_time, max_ratio, is_problem
