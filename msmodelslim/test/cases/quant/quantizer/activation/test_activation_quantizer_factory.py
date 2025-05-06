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

from msmodelslim.quant.quantizer.activation.base import ActivationQuantConfig, ActivationQuantizer
from msmodelslim.quant.quantizer.activation.factory import ActivationQuantizerFactory
from msmodelslim.quant.quantizer.activation.minmax import STATISTIC_STRATEGY_REGISTRY
from msmodelslim.quant.quantizer.activation.observer import OBSERVER_REGISTRY
from msmodelslim.quant.quantizer.base.const import ActivationQuantMethod, ActivationQuantScope


class TestActivationQuantizerFactory(unittest.TestCase):
    """测试ActivationQuantizerFactory的单元测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建基本的量化配置
        self.config = ActivationQuantConfig(
            bits=8,
            method=ActivationQuantMethod.MINMAX,
            scope=ActivationQuantScope.PER_TENSOR,
            symmetric=False,
            signed=True
        )

    def test_create_quantizer_basic(self):
        """测试基本的量化器创建功能"""
        # 创建量化器
        quantizer = ActivationQuantizerFactory.create(self.config)

        # 验证量化器类型
        self.assertIsInstance(quantizer, ActivationQuantizer)

        # 验证量化器配置
        self.assertEqual(quantizer.config.bits, 8)
        self.assertEqual(quantizer.config.method, ActivationQuantMethod.MINMAX)
        self.assertEqual(quantizer.config.scope, ActivationQuantScope.PER_TENSOR)
        self.assertEqual(quantizer.config.symmetric, False)
        self.assertEqual(quantizer.config.signed, True)

    def test_create_quantizer_with_different_scopes(self):
        """测试不同作用域的量化器创建"""
        # 测试PER_TENSOR作用域
        config_per_tensor = ActivationQuantConfig(
            bits=8,
            method=ActivationQuantMethod.MINMAX,
            scope=ActivationQuantScope.PER_TENSOR
        )
        quantizer_per_tensor = ActivationQuantizerFactory.create(config_per_tensor)
        self.assertIsInstance(quantizer_per_tensor.observer,
                              OBSERVER_REGISTRY.get_by_name(ActivationQuantScope.PER_TENSOR.value))

        # 测试PER_TOKEN作用域
        config_per_token = ActivationQuantConfig(
            bits=8,
            method=ActivationQuantMethod.MINMAX,
            scope=ActivationQuantScope.PER_TOKEN
        )
        quantizer_per_token = ActivationQuantizerFactory.create(config_per_token)
        self.assertIsInstance(quantizer_per_token.observer,
                              OBSERVER_REGISTRY.get_by_name(ActivationQuantScope.PER_TOKEN.value))

    def test_create_quantizer_with_different_methods(self):
        """测试不同统计方法的量化器创建"""
        # 测试MINMAX方法
        config_minmax = ActivationQuantConfig(
            bits=8,
            method=ActivationQuantMethod.MINMAX,
            scope=ActivationQuantScope.PER_TENSOR
        )
        quantizer_minmax = ActivationQuantizerFactory.create(config_minmax)
        self.assertIsInstance(quantizer_minmax.statistics,
                              STATISTIC_STRATEGY_REGISTRY.get_by_name(ActivationQuantMethod.MINMAX.value))


if __name__ == '__main__':
    unittest.main()
