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

from .base import BaseChecker


class PingChecker(BaseChecker):
    def _check(self, results):
        success_pattern = "3 received, 0% packet loss"

        for host, ping_result in results.items():
            if success_pattern not in ping_result:
                self.error_handler.add_error(
                    path=f"当前机器 -x-> {host}",
                    expected=success_pattern,
                    actual=ping_result,
                    reason=f"当前机器 ping 主机 {host} 失败，请检查 rank table",
                    severity="high",
                )
