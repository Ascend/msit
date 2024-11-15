# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import unittest
import tempfile
from components.utils.security_check import ms_makedirs
from components.utils.check import PathChecker


class TestMakedirs(unittest.TestCase):
    
    def setUp(self) -> None:
        self.dp = tempfile.TemporaryDirectory()
        self.dp_invalid = tempfile.TemporaryDirectory()
        os.chmod(self.dp_invalid, chmod=0o777)

    def test_makedirs_valid(self) -> None:
        target_dir = os.path.join(self.dp, "d1")
        ms_makedirs(target_dir)
        assert os.path.exists(target_dir)
    
        target_dir = os.path.join(self.dp, "d2/d3/d4")
        ms_makedirs(target_dir)
        assert os.path.exists(target_dir)

    def test_makedirs_invalid(self) -> None:
        target_dir = os.path.join(self.dp_invalid, "d1")
        if not PathChecker().is_safe_parent_dir().check(os.path.join(self.dp_invalid, "d1")):
            with self.assertRaises(OSError):
                ms_makedirs(target_dir)

        target_dir = os.path.join(self.dp_invalid, "d2/d3/d4")
        if not PathChecker().is_safe_parent_dir().check(os.path.join(self.dp_invalid, "d2")):
            with self.assertRaises(OSError):
                ms_makedirs(target_dir)

    def tearDown(self) -> None:
        self.dp.cleanup()



