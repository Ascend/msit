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

# pylint: disable=duplicate-code

import unittest
from unittest.mock import mock_open
from unittest.mock import patch

from msprechecker.utils import is_in_container
from msprechecker.utils import singleton


class TestIsInContainer(unittest.TestCase):
    @patch("os.path.exists")
    def test_is_in_container_when_dockerenv_exists(self, mock_exists):
        """测试当.dockerenv文件存在时返回True"""
        mock_exists.return_value = True
        self.assertTrue(is_in_container())

    @patch("os.path.exists")
    @patch("msprechecker.utils.helper.Path.open", new_callable=mock_open, read_data="systemd (1, #threads: 1)\n")
    def test_is_in_container_when_dockerenv_not_exists_and_systemd(self, mock_open_file, mock_exists):
        """测试当.dockerenv不存在且第一个进程是systemd时返回False"""
        mock_exists.return_value = False
        self.assertFalse(is_in_container())

    @patch("os.path.exists")
    @patch("msprechecker.utils.helper.Path.open", new_callable=mock_open, read_data="not_systemd (1, #threads: 1)\n")
    def test_is_in_container_when_dockerenv_not_exists_and_not_systemd(self, mock_open_file, mock_exists):
        """测试当.dockerenv不存在且第一个进程不是systemd时返回True"""
        mock_exists.return_value = False
        self.assertTrue(is_in_container())

    @patch("os.path.exists")
    @patch("msprechecker.utils.helper.Path.open", side_effect=Exception("File not found"))
    def test_is_in_container_when_sched_file_read_fails(self, mock_open_file, mock_exists):
        """测试当读取/proc/1/sched失败时返回True"""
        mock_exists.return_value = False
        self.assertTrue(is_in_container())


class TestSingletonDecorator(unittest.TestCase):
    def test_singleton_decorator_creates_only_one_instance(self):
        """测试单例装饰器只创建一个实例"""

        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value

        instance1 = TestClass(1)
        instance2 = TestClass(2)

        self.assertIs(instance1, instance2)
        self.assertEqual(instance1.value, 1)  # 第二次初始化参数被忽略
        self.assertEqual(instance2.value, 1)
