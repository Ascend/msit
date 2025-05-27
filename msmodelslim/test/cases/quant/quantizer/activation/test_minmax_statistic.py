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

from msmodelslim.quant.quantizer.activation.minmax import MinMaxStrategy, ActMinMaxConfig


class TestMinMaxStatistic(unittest.TestCase):
    def setUp(self):
        self.config = ActMinMaxConfig()
        self.statistic = MinMaxStrategy(config=self.config)

    def test_update(self):
        # 测试单个张量的统计
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.statistic.update(x)
        min_val, max_val = self.statistic.get_min_max()
        self.assertEqual(min_val.item(), 1.0)
        self.assertEqual(max_val.item(), 4.0)

        # 测试多个张量的累积统计
        x2 = torch.tensor([[0.5, 5.0], [2.0, 3.0]])
        self.statistic.update(x2)
        min_val, max_val = self.statistic.get_min_max()
        self.assertEqual(min_val.item(), 0.5)  # 累积后的最小值
        self.assertEqual(max_val.item(), 5.0)  # 累积后的最大值

    def test_reset(self):
        # 先更新一些统计数据
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.statistic.update(x)

        # 清除统计数据
        self.statistic.reset()

        # 验证统计数据已被清除
        with self.assertRaises(RuntimeError):
            _, _ = self.statistic.get_min_max()

    def test_get_stats_without_update(self):
        # 尝试在没有更新统计数据的情况下获取统计数据
        with self.assertRaises(RuntimeError):
            _, _ = self.statistic.get_min_max()


if __name__ == '__main__':
    unittest.main()
