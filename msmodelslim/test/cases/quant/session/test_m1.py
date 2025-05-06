#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from test.testing_utils.mock import mock_kia_library, mock_security_library

mock_kia_library()
mock_security_library()

from msmodelslim.quant.processor.anti_outlier.m1 import M1ProcessorConfig
from msmodelslim.quant.processor.anti_outlier.m1 import smooth_ln_fcs
from msmodelslim.quant.session.session import quant_model, SessionConfig


class SimpleModel(nn.Module):
    """简单的测试模型，包含LayerNorm和Linear层"""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x


class TestM1Processor(unittest.TestCase):
    """测试M1处理器通过quant_model函数的功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.batch_size = 4
        self.hidden_size = 64
        self.model = SimpleModel(self.hidden_size)
        self.calib_data = [torch.randn(self.batch_size, self.hidden_size) for _ in range(2)]

    @patch('msmodelslim.quant.processor.anti_outlier.m1.smooth_ln_fcs', wraps=smooth_ln_fcs)
    def test_quant_model_with_m1(self, mock_smooth_ln_fcs):
        """测试使用quant_model函数应用M1处理器"""
        # 创建会话配置
        session_cfg = SessionConfig(
            processor_cfg_map={"m1": M1ProcessorConfig()},
            calib_data=self.calib_data
        )

        # 应用量化处理
        quant_model(self.model, session_cfg)

        # 验证模型是否被正确处理
        # 这里我们主要验证模型结构是否保持不变，以及是否可以正常前向传播
        test_input = torch.randn(self.batch_size, self.hidden_size)
        output = self.model(test_input)

        # 验证输出形状是否正确
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

        # 验证smooth_ln_fcs是否被调用
        self.assertTrue(hasattr(mock_smooth_ln_fcs, 'call_args'))
        smooth_args = mock_smooth_ln_fcs.call_args[0]
        self.assertEqual(smooth_args[0], self.model.ln)
        self.assertEqual(smooth_args[1], [self.model.fc1])


if __name__ == "__main__":
    unittest.main()
