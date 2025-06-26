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
import tempfile
import unittest
from unittest.mock import patch, mock_open

from msguard.security import open_s

from msprechecker.prechecker.register import CheckResult
from msprechecker.prechecker.system_checker import (
    SystemInfoCollect,
    KernelReleaseChecker,
    DriverVersionChecker,
    VirtualMachineChecker,
    TransparentHugepageChecker,
    CpuHighPerformanceChecker,
    SystemChecker,
    TRANSPARENT_HUGEPAGE_PATH,
)


class TestSystemInfoCollect(unittest.TestCase):
    @patch('msprechecker.prechecker.system_checker.run_shell_command')
    @patch('os.sysconf')
    @patch('os.getenv')
    def test_collect_env(self, mock_getenv, mock_sysconf, mock_run_shell):
        with tempfile.TemporaryDirectory() as temp_dir:
            toolkit_path = os.path.join(temp_dir, "toolkit")
            os.mkdir(toolkit_path, 0o750)
            toolkit_version_path = os.path.join(toolkit_path, "version.cfg")
            with open_s(toolkit_version_path, 'w') as f:
                f.write("Version=1.2.3\n")

            mindie_path = os.path.join(temp_dir, "mindie")
            os.mkdir(mindie_path, 0o750)
            mindie_version_path = os.path.join(mindie_path, "version.info")
            with open_s(mindie_version_path, 'w') as f:
                f.write("Ascend-mindie-service: 3.4.5\n")

            mock_run_shell.return_value.stdout = "CPU(s): 8\nModel name: Test CPU"
            mock_sysconf.return_value = 4096

            def getenv_side_effect(var):
                if var == "ASCEND_TOOLKIT_HOME":
                    return toolkit_path
                return mindie_path
            mock_getenv.side_effect = getenv_side_effect

            # Test
            collector = SystemInfoCollect()
            result = collector.collect_env()

            # Assertions
            self.assertEqual(result['cpu_model_name'], "Test CPU")
            self.assertEqual(result['cpu_num'], "8")
            self.assertEqual(result['page_size'], 4096)
            self.assertEqual(result['ascend_toolkit_version'], "1.2.3")
            self.assertEqual(result['mindie_version'], "3.4.5")


class TestKernelReleaseChecker(unittest.TestCase):
    @patch('platform.release')
    def test_collect_env(self, mock_release):
        mock_release.return_value = "5.10.0"

        checker = KernelReleaseChecker()
        result = checker.collect_env()

        self.assertEqual(result, (5, 10))

    def test_do_precheck(self):
        checker = KernelReleaseChecker()

        # Test older version
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck((4, 15))
            expected_reason = "建议升级到 5.10 以上，cpu下发加快，减少 host bound"
            mock_show.assert_called_with(
                domain="system",
                checker="内核版本",
                result=CheckResult.ERROR,
                action="升级到 5.10 以上",
                reason=expected_reason
            )

        # Test equal version
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck((5, 10))
            mock_show.assert_called_with("system", "内核版本", CheckResult.OK)

        # Test newer version
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck((6, 1))
            mock_show.assert_called_with("system", "内核版本", CheckResult.OK)


class TestDriverVersionChecker(unittest.TestCase):
    @patch('os.access')
    @patch('os.stat')
    def test_collect_env(self, mock_stat, mock_access):
        mock_stat.return_value = os.stat_result(
            [33188, 238815, 2096, 1, 0, 0, 20491, 1750903227, 1750903226, 1750903226]
        )
        mock_access.return_value = True
        version_file_content = "Version=24.1.0\n"

        checker = DriverVersionChecker()
        with patch('msprechecker.prechecker.system_checker.open_s', mock_open(read_data=version_file_content)):
            result = checker.collect_env()

        self.assertEqual(result['major_version'], 24)
        self.assertEqual(result['minor_version'], 1)
        self.assertEqual(result['mini_version'], 0)

    def test_do_precheck(self):
        checker = DriverVersionChecker()

        # Test older version
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck({'major_version': 23, 'minor_version': 2, 'mini_version': 0})
            expected_reason = "建议升级到最新的版本的驱动，性能会有提升"
            mock_show.assert_called_with(
                domain="system",
                checker="驱动版本",
                result=CheckResult.ERROR,
                action="升级到 24.1.0 以上",
                reason=expected_reason
            )

        # Test equal version
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck({'major_version': 24, 'minor_version': 1, 'mini_version': 0})
            mock_show.assert_called_with("system", "驱动版本", CheckResult.OK)

        # Test newer version
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck({'major_version': 25, 'minor_version': 0, 'mini_version': 0})
            mock_show.assert_called_with("system", "驱动版本", CheckResult.OK)


class TestVirtualMachineChecker(unittest.TestCase):
    @patch('os.access', return_value=True)
    @patch('os.path.exists')
    def test_collect_env(self, mock_exists, _):
        mock_exists.return_value = True

        # Test VM case
        cpuinfo_content = "hypervisor: KVM\n"
        checker = VirtualMachineChecker()
        with patch('msprechecker.prechecker.system_checker.open_s', mock_open(read_data=cpuinfo_content)):
            result = checker.collect_env()
        self.assertTrue(result)

        # Test physical machine case
        cpuinfo_content = "processor: 0\n"
        checker = VirtualMachineChecker()
        with patch('msprechecker.prechecker.system_checker.open_s', mock_open(read_data=cpuinfo_content)):
            result = checker.collect_env()
        self.assertFalse(result)

    def test_do_precheck(self):
        checker = VirtualMachineChecker()

        # Test VM case
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck(True)
            action = (
                "确定分配的 cpu 是完全体，如 VMware 中 启用 CPU/MMU Virtualization（ESXi 高级设置）、"
                "禁用 CPU 限制（cpuid.coresPerSocket 配置为物理核心数）；KVM 中 配置 host-passthrough "
                "模式（暴露完整 CPU 指令集）、启用多队列 virtio-net（减少网络延迟）"
            )
            reason = (
                "虚拟机和物理机的 cpu 核数、频率有差异会导致性能下降，"
                "如果是虚拟机环境，建议检查 cpu 情况"
            )
            mock_show.assert_called_with(
                "system",
                "可能是虚拟机",
                CheckResult.ERROR,
                action=action,
                reason=reason
            )

        # Test physical machine case
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck(False)
            self.assertFalse(mock_show.called)


class TestTransparentHugepageChecker(unittest.TestCase):
    @patch('os.access', return_value=True)
    @patch('os.path.exists')
    def test_collect_env(self, mock_exists, _):
        mock_exists.return_value = True
        
        # Test enabled case
        hugepage_content = "[always] madvise never\n"
        checker = TransparentHugepageChecker()
        with patch('msprechecker.prechecker.system_checker.open_s', mock_open(read_data=hugepage_content)):
            result = checker.collect_env()
        self.assertTrue(result)
        
        # Test disabled case
        hugepage_content = "always madvise [never]\n"
        checker = TransparentHugepageChecker()
        with patch('msprechecker.prechecker.system_checker.open_s', mock_open(read_data=hugepage_content)):
            result = checker.collect_env()
        self.assertFalse(result)
    
    def test_do_precheck(self):
        checker = TransparentHugepageChecker()
        checker.additional_msg = "；恢复配置使用：echo never > /path"
        
        # Test disabled case
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck(False)
            action = f"设置为 always：echo always > {TRANSPARENT_HUGEPAGE_PATH}" + checker.additional_msg
            mock_show.assert_called_with(
                "system",
                "透明大页",
                CheckResult.ERROR,
                action=action,
                reason="开启透明大页，吞吐率结果会更稳定"
            )
        
        # Test enabled case
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck(True)
            self.assertFalse(mock_show.called)


class TestCpuHighPerformanceChecker(unittest.TestCase):
    @patch.object(CpuHighPerformanceChecker, '_normal_check')
    @patch.object(CpuHighPerformanceChecker, '_kunpeng_check')
    def test_collect_env(self, mock_kunpeng, mock_normal):
        mock_normal.return_value = (4, 2)
        mock_kunpeng.return_value = False
        
        checker = CpuHighPerformanceChecker()
        result = checker.collect_env()
        
        self.assertEqual(result['cpu_count'], 4)
        self.assertEqual(result['performance_count'], 2)
        self.assertFalse(result['is_high_performance_on'])
    
    def test_do_precheck(self):
        checker = CpuHighPerformanceChecker()
        
        # Test high performance case
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck({
                'cpu_count': 4,
                'performance_count': 4,
                'is_high_performance_on': True
            })
            mock_show.assert_called_with("system", "CPU高性能模式", CheckResult.OK)
        
        # Test not high performance case
        with patch('msprechecker.prechecker.system_checker.show_check_result') as mock_show:
            checker.do_precheck({
                'cpu_count': 4,
                'performance_count': 2,
                'is_high_performance_on': False
            })
            action = (
                "常规 CPU 高性能模式开启校验失败，如果您确保已经开启了高性能模式，请忽略。\n"
                "        开启 CPU 高性能模式：cpupower -c all frequency-set -g performance；\n"
                "        如果没有 cpupower 命令可以通过 EulerOS/CentOS: yum install kernel-tools "
                "或 Ubuntu：apt install cpufrequtils 安装；\n"
                "        如果失败可能需要在 BIOS 中开启\n"
                "        如果需要回退，可以使用命令：cpupower -c all frequency-set -g powersave"
            )
            reason = (
                "使 CPU 运行在最大频率下，可以提升CPU性能，但是会提高能耗"
            )
            mock_show.assert_called_with(
                "system",
                "CPU高性能模式",
                CheckResult.WARN,
                action=action,
                reason=reason
            )


class TestSystemChecker(unittest.TestCase):
    def test_init_sub_checkers(self):
        checker = SystemChecker()
        sub_checkers = checker.init_sub_checkers()
        
        self.assertEqual(len(sub_checkers), 6)
        self.assertIsInstance(sub_checkers[0], SystemInfoCollect)
        self.assertIsInstance(sub_checkers[1], KernelReleaseChecker)
        self.assertIsInstance(sub_checkers[2], DriverVersionChecker)
        self.assertIsInstance(sub_checkers[3], VirtualMachineChecker)
        self.assertIsInstance(sub_checkers[4], TransparentHugepageChecker)
        self.assertIsInstance(sub_checkers[5], CpuHighPerformanceChecker)
