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

import os
import unittest
from unittest.mock import patch

from msprechecker.collectors import EnvCollector


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.normal_collector = EnvCollector()
        self.filtered_collector = EnvCollector(filter_env=True)

    def test_normal_collect_should_return_all_os_environ(self):
        test_data = {"abc": "123", "ASCEND": "123"}
        with patch.dict(os.environ, test_data, clear=True):
            collect_result = self.normal_collector.collect()
            self.assertEqual(collect_result.data, test_data)

    def test_filtered_collect_should_return_ascend_related(self):
        test_data = {"abc": "123", "ASCEND": "123"}
        with patch.dict(os.environ, test_data, clear=True):
            collect_result = self.filtered_collector.collect()
            self.assertEqual(collect_result.data, {"ASCEND": "123"})

    @patch("os.environ")
    def test_when_occur_collect_error_should_added_in_error_handler(self, mock_environ):
        mock_environ.items.side_effect = RuntimeError
        collect_result = self.normal_collector.collect()
        self.assertEqual(len(collect_result.error_handler.errors), 1)
