# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
import json
import unittest
import tempfile
from unittest.mock import patch

from msprechecker.prechecker.hccl_checker import (
    HcclIfnameChecker,
    HcclLinkChecker,
    HcclTlsSwitchChecker,
    HcclPingChecker,
    CheckResult,
    ACTION_WHEN_NO_DATA_COLLECTED,
    REASON_WHEN_NO_DATA_COLLECTED
)


class TestHcclIfnameChecker(unittest.TestCase):
    def setUp(self):
        self.checker = HcclIfnameChecker()

    @patch('msprechecker.prechecker.hccl_checker.run_hccl_command')
    def test_collect_env_success(self, mock_run_hccl):
        mock_run_hccl.return_value = [
            ["Ifname: eth0", "other info"],
            ["Ifname: eth1", "other info"]
        ]
        result = self.checker.collect_env()
        self.assertEqual(result, ["eth0", "eth1"])

    @patch('msprechecker.prechecker.hccl_checker.run_hccl_command')
    def test_collect_env_failure(self, mock_run_hccl):
        mock_run_hccl.return_value = []
        result = self.checker.collect_env()
        self.assertEqual(result, [])

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_no_ifnames(self, mock_show):
        self.checker.do_precheck([])
        mock_show.assert_called_once_with(
            "hccl",
            "lldp Ifname",
            CheckResult.UNFINISH,
            action=ACTION_WHEN_NO_DATA_COLLECTED,
            reason=REASON_WHEN_NO_DATA_COLLECTED
        )

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_empty_ifnames(self, mock_show):
        self.checker.do_precheck(["", "eth0"])
        mock_show.assert_called_once_with(
            "hccl",
            "lldp Ifname",
            CheckResult.ERROR,
            action="检查服务器上 NPU 对应的交换机连接，如果是光纤直连，忽略该条",
            reason="HCCL Ifname 存在空值 ['', 'eth0']",
        )

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_valid_ifnames(self, mock_show):
        self.checker.do_precheck(["eth0", "eth1"])
        mock_show.assert_called_once_with(
            "hccl", "lldp Ifname: ['eth0', 'eth1']", CheckResult.OK
        )


class TestHcclLinkChecker(unittest.TestCase):
    def setUp(self):
        self.checker = HcclLinkChecker()

    @patch('msprechecker.prechecker.hccl_checker.run_hccl_command')
    def test_collect_env_success(self, mock_run_hccl):
        mock_run_hccl.return_value = [
            ["link status: UP", "other info"],
            ["link status: DOWN", "other info"]
        ]
        result = self.checker.collect_env()
        self.assertEqual(result, ["UP", "DOWN"])

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_no_link_status(self, mock_show):
        self.checker.do_precheck([])
        mock_show.assert_called_once_with(
            "hccl",
            "lldp Ifname",
            CheckResult.UNFINISH,
            action=ACTION_WHEN_NO_DATA_COLLECTED,
            reason=REASON_WHEN_NO_DATA_COLLECTED
        )

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_down_link(self, mock_show):
        self.checker.do_precheck(["UP", "DOWN"])
        mock_show.assert_called_once_with(
            "hccl",
            "link",
            CheckResult.ERROR,
            action="检查服务器上 NPU 连接情况",
            reason="HCCL link 存在 down 值 ['UP', 'DOWN']",
        )

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_all_up(self, mock_show):
        self.checker.do_precheck(["UP", "UP"])
        mock_show.assert_called_once_with(
            "hccl", "link: ['UP', 'UP']", CheckResult.OK
        )


class TestHcclTlsSwitchChecker(unittest.TestCase):
    def setUp(self):
        self.checker = HcclTlsSwitchChecker()

    @patch('msprechecker.prechecker.hccl_checker.run_hccl_command')
    def test_collect_env_success(self, mock_run_hccl):
        mock_run_hccl.return_value = [
            ["tls switch[0]", "other info"],
            ["tls switch[1]", "other info"]
        ]
        result = self.checker.collect_env()
        self.assertEqual(result, ["0", "1"])

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_no_tls_switch(self, mock_show):
        self.checker.do_precheck([])
        mock_show.assert_called_once_with(
            "hccl",
            "tls switch",
            CheckResult.UNFINISH,
            action=ACTION_WHEN_NO_DATA_COLLECTED,
            reason=REASON_WHEN_NO_DATA_COLLECTED
        )

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_tls_enabled(self, mock_show):
        self.checker.do_precheck(["0", "1"])
        mock_show.assert_called_once_with(
            "hccl",
            "tls switch",
            CheckResult.ERROR,
            action="检查服务器上 HCCL tls 状态，推荐关闭：for i in {0..7}; do hccn_tool -i $i -tls -s enable 0; done",
            reason="HCCL tls 打开可能影响多机连接",
        )

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_tls_disabled(self, mock_show):
        self.checker.do_precheck(["0", "0"])
        mock_show.assert_called_once_with(
            "hccl", "tls_switch: ['0', '0']", CheckResult.OK
        )


class TestHcclPingChecker(unittest.TestCase):
    def setUp(self):
        self.checker = HcclPingChecker()

    def create_test_ranktable(self, content):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "ranktable.json")
            with open(file_path, 'w') as f:
                f.write(content)
            return file_path

    @patch('msprechecker.prechecker.hccl_checker.parse_ranktable_file')
    @patch('msprechecker.prechecker.hccl_checker.get_interface_by_ip')
    @patch('msprechecker.prechecker.hccl_checker.run_hccl_command')
    def test_collect_env_success(self, mock_run_hccl, mock_get_interface, mock_parse):
        ranktable_content = """{
            "server_list": [
                {"server_id": "192.168.1.1", "device": [{"device_ip": "192.168.1.100"}]},
                {"server_id": "192.168.1.2", "device": [{"device_ip": "192.168.1.200"}]}
            ]
        }"""
        ranktable_file = self.create_test_ranktable(ranktable_content)

        mock_parse.return_value = json.loads(ranktable_content)
        mock_get_interface.return_value = ("eth0", "192.168.1.1")
        mock_run_hccl.return_value = [["0.00% packet loss"]]

        result = self.checker.collect_env(ranktable_file=ranktable_file)
        self.assertIsNotNone(result)
        self.assertIn("192.168.1.2", result)

    @patch('msprechecker.prechecker.hccl_checker.parse_ranktable_file')
    def test_collect_env_no_ranktable(self, mock_parse):
        mock_parse.return_value = None
        result = self.checker.collect_env()
        self.assertIsNone(result)

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_no_results(self, mock_show):
        self.checker.do_precheck(None)
        mock_show.assert_called_once_with(
            "hccl",
            "ping",
            CheckResult.UNFINISH,
            action=ACTION_WHEN_NO_DATA_COLLECTED,
            reason=REASON_WHEN_NO_DATA_COLLECTED
        )

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_success(self, mock_show):
        self.checker.local_ip = "192.168.1.1"
        test_results = {
            "192.168.1.2": {"192.168.1.200": [True, True]},
            "192.168.1.1": {"192.168.1.100": [True, True]}
        }
        self.checker.do_precheck(test_results)
        mock_show.assert_called_once_with(
            "hccl", "ping server 192.168.1.2 all pass", CheckResult.OK
        )

    @patch('msprechecker.prechecker.hccl_checker.show_check_result')
    def test_do_precheck_failure(self, mock_show):
        self.checker.local_ip = "192.168.1.1"
        test_results = {
            "192.168.1.2": {"192.168.1.200": [False, False]},
            "192.168.1.1": {"192.168.1.100": [True, True]}
        }
        self.checker.do_precheck(test_results)
        mock_show.assert_called_once_with(
            "hccl",
            "ping",
            CheckResult.ERROR,
            action="检查本机到服务器 192.168.1.2 0 卡的连接状态",
            reason="本机到对端卡的 ping 结果存在失败 [False, False]",
        )
