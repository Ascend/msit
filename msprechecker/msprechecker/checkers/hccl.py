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


class LinkChecker(BaseChecker):
    def _check(self, results):
        """
        check if results are full of 'link status: UP'
        """
        if not results:
            return

        success_pattern = "link status: UP"
        for device_id, result in enumerate(results):
            if success_pattern not in result:
                self.error_handler.add_error(
                    path=f"Device ID: {device_id}",
                    actual=result,
                    expected="link status: UP",
                    reason='当前机器的网卡 LINK 状态应该要全为 "UP"',
                    severity="high",
                )


class VnicChecker(BaseChecker):
    def _check(self, results):
        """
        check if results are full of 'vnic link status: UP' and 'ipaddr: xxx'
        """
        if not results:
            return

        for device_id, result in enumerate(results):
            fields = result.strip().split("\n")
            if not len(fields) == 3:
                self.error_handler.add_error(
                    path=f"Device ID: {device_id}",
                    actual=result,
                    reason="要有 link status, ipaddr 和 netmask",
                    severity="high",
                )
                continue

            if fields[0] != "vnic link status: UP" or not fields[1].startswith(
                "vnic ipaddr:"
            ):
                self.error_handler.add_error(
                    path=f"Device ID: {device_id}",
                    actual=result,
                    reason="如果是 A3 机器，板内的 VNIC IP 需要配置，且连接状态为 UP",
                    severity="high",
                )


class TlsChecker(BaseChecker):
    def _check(self, results):
        """
        check if results are full of 'dev_id:x, tls switch[0]'
        """
        if not results:
            return

        tls_switch_pattern = "tls switch["
        for device_id, result in enumerate(results):
            if tls_switch_pattern not in result:
                continue  # no certificates regarding as passed

            tls_switch_idx = result.index(tls_switch_pattern)
            switch_state = result[tls_switch_idx + len(tls_switch_pattern)]
            if switch_state != "0":
                self.error_handler.add_error(
                    path=f"Device ID: {device_id}",
                    expected="0",
                    actual=result,
                    reason="当前机器的 TLS 证书需要被使能，注意需要在 root 用户下执行该项检测",
                    severity="medium",
                )


class HCCLChecker(BaseChecker):
    def _check(self, results):
        """
        check if results are full of [[{'3 received, 0.00% packet loss'}, ...]]'
        """
        if not results:
            return

        for cmd, (ret, out) in results.items():
            if ret:
                self.error_handler.add_error(path=cmd, reason=out, severity="high")
            elif "3 received" not in out:
                self.error_handler.add_error(path=cmd, reason=out, severity="high")
