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
from unittest.mock import mock_open, patch
from msprechecker.core.utils import read_file_lines


class TestFileUtils(unittest.TestCase):
    def test_read_file_lines_file_not_exists(self):
        self.assertIsNone(read_file_lines("/nonexistent/file"))

    def test_read_file_lines_empty_path(self):
        self.assertIsNone(read_file_lines(""))
        self.assertIsNone(read_file_lines(None))

    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_read_file_lines_exception(self, _):
        with patch("os.path.isfile", return_value=True):
            self.assertIsNone(read_file_lines("/dummy/path"))
