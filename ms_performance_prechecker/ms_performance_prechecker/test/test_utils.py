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
import errno
import unittest
from unittest.mock import patch

from ms_performance_prechecker.prechecker.utils import (
    str_ignore_case,
    str_to_digit,
    is_deepseek_model,
    same,
    is_port_in_use,
    get_local_to_master_ip,
)


class TestUtils(unittest.TestCase):
    def test_str_ignore_case(self):
        self.assertEqual(str_ignore_case("Test_String"), "teststring")
        self.assertEqual(str_ignore_case("Another-Test"), "anothertest")

    def test_str_to_digit(self):
        self.assertEqual(str_to_digit("123"), 123)
        self.assertEqual(str_to_digit("123.45"), 123.45)
        self.assertEqual(str_to_digit("abc", default_value=0), 0)

    def test_is_deepseek_model(self):
        self.assertTrue(is_deepseek_model("DeepSeek_Model"))
        self.assertFalse(is_deepseek_model("OtherModel"))

    def test_same(self):
        self.assertTrue(same([1, 1, 1]))
        self.assertFalse(same([1, 2, 1]))

    @patch("socket.socket")
    def test_is_port_in_use(self, mock_socket):
        self.assertFalse(is_port_in_use(8080))
        
        mock_socket.return_value.__enter__().bind.side_effect = OSError(errno.EADDRINUSE, "Address in use")
        self.assertTrue(is_port_in_use(8080))

    @patch("socket.socket")
    def test_get_local_to_master_ip(self, mock_socket):
        mock_socket.return_value.getsockname.return_value = ("192.168.1.1", 0)
        self.assertEqual(get_local_to_master_ip(), "192.168.1.1")
