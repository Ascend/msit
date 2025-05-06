#  -*- coding: utf-8 -*-
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
from typing import Tuple, Optional
from unittest.mock import patch

import torch
import torch.nn as nn

from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.runner.layer_wise_runner import LayerWiseProcessUnit


class TestDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        return (x, None)


class SimpleModel(nn.Module):
    """简单的测试模型，包含一个线性层和一个TransformerDecoder层"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.decoder = TestDecoderLayer()

    def forward(self, x):
        x = self.linear(x)
        x = self.decoder(x, x)
        return x


class TestProcessor(BaseProcessor):
    """测试用的处理器，用于测试LayerWiseProcessUnit的功能"""

    def __init__(self, model):
        super().__init__(model)
        self.processed_modules = []

    def process(self, request):
        self.processed_modules.append(request.name)
        super().process(request)


class TestLayerWiseProcessUnit(unittest.TestCase):
    """测试LayerWiseProcessUnit的功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.model = SimpleModel()
        self.processor = TestProcessor(self.model)

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    def test_build_generators_with_input_data(self, mock_get_world_size, mock_get_rank):
        """测试使用输入数据构建生成器的功能"""
        # 设置模拟函数
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        # 创建输入数据
        input_data = [torch.randn(5, 10)]

        # 创建处理单元
        process_unit = LayerWiseProcessUnit(self.processor, input_data)

        # 验证生成器
        self.assertEqual(len(process_unit.generators), 1)

        # 获取第一个生成器的第一个请求
        request = next(process_unit.generators[0])

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)

    def test_build_generators_without_input_data(self):
        """测试不使用输入数据构建生成器的功能"""
        # 创建处理单元
        process_unit = LayerWiseProcessUnit(self.processor)

        # 验证生成器
        self.assertEqual(len(process_unit.generators), 1)

        # 获取第一个生成器的第一个请求
        request = next(process_unit.generators[0])

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)


if __name__ == '__main__':
    unittest.main()
