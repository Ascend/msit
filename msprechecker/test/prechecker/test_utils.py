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

import unittest
import os
import json
import shutil
import tempfile
import socket
import logging
from unittest.mock import patch, mock_open, MagicMock

from msprechecker.prechecker.utils import (
    str_to_digit,
    is_deepseek_model,
    same,
    get_dict_value_by_pos,
    set_log_level,
    set_logger,
    read_csv,
    read_json,
    read_csv_or_json,
    get_next_dict_item,
    get_version_info,
    get_mindie_server_config,
    run_shell_command,
    get_global_env_info,
    get_npu_info,
    get_interface_by_ip,
    SimpleProgressBar
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("msprechecker_logger")
        self.logger.handlers = []

    def test_str_to_digit(self):
        self.assertEqual(str_to_digit("123"), 123)
        self.assertEqual(str_to_digit("123.45"), 123.45)
        self.assertEqual(str_to_digit("abc"), None)
        self.assertEqual(str_to_digit("123abc"), None)

    def test_is_deepseek_model(self):
        self.assertTrue(is_deepseek_model("DeepSeek-Model"))
        self.assertTrue(is_deepseek_model("deepseek_model"))
        self.assertFalse(is_deepseek_model("other_model"))

    def test_same(self):
        self.assertTrue(same([1, 1, 1]))
        self.assertFalse(same([1, 2, 1]))

    def test_get_dict_value_by_pos(self):
        test_dict = {"a": {"b": [1, 2, {"c": 3}]}}
        self.assertEqual(get_dict_value_by_pos(test_dict, "a:b:2:c"), 3)
        self.assertEqual(get_dict_value_by_pos(test_dict, "a:b:0"), 1)
        self.assertEqual(get_dict_value_by_pos(test_dict, "invalid:path", "default"), "default")

    def test_set_log_level(self):
        set_log_level("debug")
        self.assertEqual(self.logger.level, logging.DEBUG)
        set_log_level("info")
        self.assertEqual(self.logger.level, logging.INFO)

    def test_set_logger(self):
        test_logger = logging.getLogger("test_logger")
        set_logger(test_logger)
        self.assertEqual(len(test_logger.handlers), 1)
        self.assertFalse(test_logger.propagate)


class TestFileOperations(unittest.TestCase):
    def test_read_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, "test.csv")
            with open(csv_file, "w") as f:
                f.write("key1,key2\nvalue1,value2\nvalue3,value4")

            result = read_csv(csv_file)
            self.assertEqual(result, {"key1": ["value1", "value3"], "key2": ["value2", "value4"]})

    def test_read_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, "test.json")
            with open(json_file, "w") as f:
                json.dump({"key": "value"}, f)

            result = read_json(json_file)
            self.assertEqual(result, {"key": "value"})

    def test_read_csv_or_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test CSV
            csv_file = os.path.join(temp_dir, "test.csv")
            with open(csv_file, "w") as f:
                f.write("key1,key2\nvalue1,value2\nvalue3,value4")
            self.assertEqual(
                read_csv_or_json(csv_file),
                {"key1": ["value1", "value3"], "key2": ["value2", "value4"]}
            )

            # Test JSON
            json_file = os.path.join(temp_dir, "test.json")
            with open(json_file, "w") as f:
                json.dump({"key": "value"}, f)
            self.assertEqual(read_csv_or_json(json_file), {"key": "value"})

            # Test non-existent file
            self.assertIsNone(read_csv_or_json("nonexistent.file"))

    def test_get_next_dict_item(self):
        self.assertEqual(get_next_dict_item({"a": 1, "b": 2}), {"a": 1})
        self.assertIsNone(get_next_dict_item(None))
        self.assertEqual(get_next_dict_item({}), None)


class TestSystemInfoCollect(unittest.TestCase):
    @patch('os.path.exists')
    @patch('msprechecker.prechecker.utils.open_s', new_callable=mock_open, read_data="key: value\nversion: 1.0")
    def test_get_version_info(self, mock_file, mock_exists):
        mock_exists.return_value = True
        result = get_version_info("/test/path")
        self.assertEqual(result, {"key": "value", "version": "1.0"})

        mock_exists.return_value = False
        result = get_version_info("/test/path")
        self.assertEqual(result, {})

    @patch('os.getenv')
    @patch('os.path.exists')
    def test_get_mindie_server_config(self, mock_exists, mock_getenv):
        mock_getenv.return_value = "/custom/path"
        mock_exists.return_value = True
        result = get_mindie_server_config()
        self.assertEqual(result, "/custom/path/conf/config.json")

        result = get_mindie_server_config("/explicit/path")
        self.assertEqual(result, "/explicit/path/conf/config.json")

    @patch('subprocess.run')
    def test_run_shell_command(self, mock_run):
        mock_run.return_value = MagicMock(stdout="output", stderr="", returncode=0)
        result = run_shell_command("test command")
        self.assertEqual(result.stdout, "output")
        
        mock_run.side_effect = Exception("Error")
        result = run_shell_command("failing command")
        self.assertEqual(result, {})

    @patch('os.environ', {'ASCEND_TEST': 'value', 'OTHER_VAR': 'ignore'})
    def test_get_global_env_info(self):
        result = get_global_env_info()
        self.assertEqual(result, {'ASCEND_TEST': 'value'})

    @patch('shutil.which', return_value=True)
    @patch('subprocess.run')
    def test_get_npu_info(self, mock_run, _):
        mock_run.return_value = MagicMock(stdout="Device d802 found in accelerators", stderr="", returncode=0)
        self.assertEqual(get_npu_info(), "d802")
        self.assertEqual(get_npu_info(to_inner_type=True), "A2")
        
        mock_run.return_value = MagicMock(stdout="No device found", stderr="", returncode=0)
        self.assertIsNone(get_npu_info())


class TestNetworkFunctions(unittest.TestCase):
    @patch("msprechecker.prechecker.utils.psutil")
    def test_get_interface_by_ip(self, psutil_mock):
        psutil_mock.net_if_addrs.return_value = {
            'eth0': [MagicMock(family=socket.AF_INET, address='192.168.1.1')],
            'lo': [MagicMock(family=socket.AF_INET, address='127.0.0.1')]
        }

        interface, ip = get_interface_by_ip('192.168.1.1')
        self.assertEqual(interface, 'eth0')
        self.assertEqual(ip, '192.168.1.1')


class TestVersion(unittest.TestCase):
    def test_version_comparison(self):
        # Test proper use of assertGreater/assertLess instead of assertTrue
        version1 = "1.0.0"
        version2 = "2.0.0"
        self.assertLess(version1, version2)
        self.assertGreater(version2, version1)


class TestSimpleProgressBar(unittest.TestCase):
    def setUp(self):
        self.iterable = range(10)
        self.progress_bar = SimpleProgressBar(self.iterable, desc="Test")

    def test_iteration(self):
        count = 0
        for _ in self.progress_bar:
            count += 1
        self.assertEqual(count, 10)

    def test_update(self):
        self.progress_bar.update(5)
        self.assertEqual(self.progress_bar.current, 5)
        self.progress_bar.update(3)
        self.assertEqual(self.progress_bar.current, 8)

    def test_progress_calculation(self):
        self.progress_bar.update(5)
        progress = self.progress_bar.current / self.progress_bar.total
        self.assertGreaterEqual(progress, 0.5)
        self.assertLessEqual(progress, 1.0)

    def test_remaining_time_calculation(self):
        # Test with no progress
        self.progress_bar._print_progress()
        self.assertEqual(self.progress_bar.current, 0)

        # Test with some progress
        self.progress_bar.update(5)
        self.progress_bar._print_progress()
        self.assertGreater(self.progress_bar.current, 0)

    def test_total_auto_calculation(self):
        pb = SimpleProgressBar(range(5))
        self.assertEqual(pb.total, 5)

    def test_empty_iterable(self):
        pb = SimpleProgressBar([], desc="Empty")
        count = 0
        self.assertEqual(count, 0)

    def test_custom_total(self):
        pb = SimpleProgressBar(range(5), total=100)
        self.assertEqual(pb.total, 100)

    def test_no_description(self):
        pb = SimpleProgressBar(range(5))
        self.assertEqual(pb.desc, "")

    def test_complete_progress(self):
        pb = SimpleProgressBar(range(5), desc="Complete")
        for _ in pb:
            pass
        self.assertEqual(pb.current, pb.total)
