# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from statistics import mean

from .base import BaseChecker


class StressChecker(BaseChecker):
    def __init__(self, *, error_handler=None, rule_manager=None, threshold=0.2):
        super().__init__(error_handler=error_handler, rule_manager=rule_manager)
        self.error_handler.type = "stress"
        self.threshold = threshold

    def _check(self, results):
        if not isinstance(results, dict):
            raise TypeError(
                f"Expected 'collect_data.data' to be dict. Got {type(results).__name__} instead."
            )
        if not results:
            raise ValueError(
                "No data collected and no errors occured during collection."
            )

        mean_time_cost = mean(results.values())
        if mean_time_cost == 0:
            mean_time_cost += (
                1e-6  # this should not happen since there will be no negative time cost
            )

        expected_time_cost = self.threshold * mean_time_cost + mean_time_cost
        for i, time_cost in results.items():
            score = (time_cost - mean_time_cost) / mean_time_cost
            if score > self.threshold:
                self.error_handler.add_error(
                    path=f"Core ID: {i}",
                    actual=f"Time Cost: {time_cost:.3f}ms",
                    expected=f"Time Cost <= {expected_time_cost:.3f}ms",
                    reason=f"该核计算时长大于平均时长的 {score:.2%}",
                    severity="high",
                )
