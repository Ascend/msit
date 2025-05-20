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

import torch

from msmodelslim.quant.quantizer.activation.minmax import MinMaxStatistic
from msmodelslim.quant.quantizer.activation.observer import PerTensorObserver


class TestPerTensorObserver(unittest.TestCase):
    def setUp(self):
        self.strategy = MinMaxStatistic()
        self.observer = PerTensorObserver()
        self.observer.set_strategy(self.strategy)

    def test_get_reduce_dims(self):
        x = torch.randn(2, 3, 4)
        reduce_dims = self.observer._get_reduce_dims(x)
        self.assertEqual(reduce_dims, [0, 1, 2])

    def test_update(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.observer.update(x)
        stats = self.observer.get_stats()
        self.assertEqual(stats[0].item(), 1.0)
        self.assertEqual(stats[1].item(), 4.0)

    def test_multiple_updates(self):
        # 第一轮更新
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.observer.update(x1)
        stats = self.observer.get_stats()
        self.assertEqual(stats[0].item(), 1.0)
        self.assertEqual(stats[1].item(), 4.0)

        # 第二轮更新 - 包含更小的最小值
        x2 = torch.tensor([[0.5, 5.0], [2.0, 3.0]])
        self.observer.update(x2)
        stats = self.observer.get_stats()
        # PerTensor应该累积历史最小/最大值
        self.assertEqual(stats[0].item(), 0.5)  # 更新后的最小值
        self.assertEqual(stats[1].item(), 5.0)  # 更新后的最大值

        # 第三轮更新 - 包含更大的最大值
        x3 = torch.tensor([[2.0, 6.0], [3.0, 4.0]])
        self.observer.update(x3)
        stats = self.observer.get_stats()
        # PerTensor应该继续累积历史最小/最大值
        self.assertEqual(stats[0].item(), 0.5)  # 保持历史最小值
        self.assertEqual(stats[1].item(), 6.0)  # 更新后的最大值

        # 第四轮更新 - 所有值都在历史范围内
        x4 = torch.tensor([[1.5, 3.0], [2.5, 3.5]])
        self.observer.update(x4)
        stats = self.observer.get_stats()
        # PerTensor应该保持历史最小/最大值不变
        self.assertEqual(stats[0].item(), 0.5)  # 保持历史最小值
        self.assertEqual(stats[1].item(), 6.0)  # 保持历史最大值

    def test_set_strategy(self):
        """测试设置新的统计策略"""
        # 创建新的统计策略
        new_strategy = MinMaxStatistic()
        
        # 设置新的统计策略
        self.observer.set_strategy(new_strategy)
        
        # 验证策略已被更新
        self.assertIs(self.observer.strategy, new_strategy)
        
        # 使用新策略进行更新
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.observer.update(x)
        
        # 验证新策略的统计结果
        stats = self.observer.get_stats()
        self.assertEqual(stats[0].item(), 1.0)
        self.assertEqual(stats[1].item(), 4.0)
        
        # 验证新策略是独立的（不影响原策略）
        self.assertNotEqual(self.observer.strategy, self.strategy)
        
        # 使用原策略进行更新
        self.observer.set_strategy(self.strategy)
        x = torch.tensor([[0.5, 5.0], [2.0, 3.0]])
        self.observer.update(x)
        
        # 验证原策略的统计结果
        stats = self.observer.get_stats()
        self.assertEqual(stats[0].item(), 0.5)
        self.assertEqual(stats[1].item(), 5.0)


if __name__ == '__main__':
    unittest.main()
