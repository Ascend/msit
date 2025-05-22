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


class TestMinMaxStatistic(unittest.TestCase):
    def setUp(self):
        self.statistic = MinMaxStatistic()

    def test_update_stats(self):
        # 测试单个张量的统计
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.statistic.update_stats(x, reduce_dims=[0, 1])
        stats = self.statistic.get_stats()
        self.assertEqual(stats[0].item(), 1.0)  # 最小值
        self.assertEqual(stats[1].item(), 4.0)  # 最大值

        # 测试多个张量的累积统计
        x2 = torch.tensor([[0.5, 5.0], [2.0, 3.0]])
        self.statistic.update_stats(x2, reduce_dims=[0, 1])
        stats = self.statistic.get_stats()
        self.assertEqual(stats[0].item(), 0.5)  # 累积后的最小值
        self.assertEqual(stats[1].item(), 5.0)  # 累积后的最大值

    def test_different_reduce_dims(self):
        # 创建一个3维张量
        x = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ])

        # 测试只缩减第一个维度
        self.statistic = MinMaxStatistic()  # 重置统计对象
        self.statistic.update_stats(x, reduce_dims=[0], keep_dims=True)
        stats = self.statistic.get_stats()
        # 形状为[2, 1, 2, 2]，其中第一个维度是min/max，第二个维度是keepdim后的维度
        self.assertEqual(stats.shape, (2, 1, 2, 2))
        # 检查第一个位置的最小值和最大值
        self.assertEqual(stats[0, 0, 0, 0].item(), 1.0)  # 第一个批次的最小值
        self.assertEqual(stats[1, 0, 0, 0].item(), 5.0)  # 第一个批次的最大值

        # 测试只缩减第二个维度
        self.statistic = MinMaxStatistic()  # 重置统计对象
        self.statistic.update_stats(x, reduce_dims=[1], keep_dims=True)
        stats = self.statistic.get_stats()
        # 形状为[2, 2, 1, 2]，其中第一个维度是min/max，第三个维度是keepdim后的维度
        self.assertEqual(stats.shape, (2, 2, 1, 2))
        # 检查第一个位置的最小值和最大值
        self.assertEqual(stats[0, 0, 0, 0].item(), 1.0)  # 第一行的最小值
        self.assertEqual(stats[1, 0, 0, 0].item(), 3.0)  # 第一行的最大值

        # 测试只缩减第三个维度
        self.statistic = MinMaxStatistic()  # 重置统计对象
        self.statistic.update_stats(x, reduce_dims=[2], keep_dims=True)
        stats = self.statistic.get_stats()
        # 形状为[2, 2, 2, 1]，其中第一个维度是min/max，第四个维度是keepdim后的维度
        self.assertEqual(stats.shape, (2, 2, 2, 1))
        # 检查第一个位置的最小值和最大值
        self.assertEqual(stats[0, 0, 0, 0].item(), 1.0)  # 第一列的最小值
        self.assertEqual(stats[1, 0, 0, 0].item(), 2.0)  # 第一列的最大值

        # 测试缩减多个维度
        self.statistic = MinMaxStatistic()  # 重置统计对象
        self.statistic.update_stats(x, reduce_dims=[0, 2], keep_dims=True)
        stats = self.statistic.get_stats()
        # 形状为[2, 1, 2, 1]，其中第一个维度是min/max，第二和第四个维度是keepdim后的维度
        self.assertEqual(stats.shape, (2, 1, 2, 1))
        # 检查第一个位置的最小值和最大值
        self.assertEqual(stats[0, 0, 0, 0].item(), 1.0)  # 第一维度和第三维度缩减后的最小值
        self.assertEqual(stats[1, 0, 0, 0].item(), 6.0)  # 第一维度和第三维度缩减后的最大值

        # 测试不缩减任何维度
        self.statistic = MinMaxStatistic()  # 重置统计对象
        self.statistic.update_stats(x, reduce_dims=[], keep_dims=True)
        stats = self.statistic.get_stats()
        # 形状为[2, 1, 1, 1]，其中第一个维度是min/max，后面是keepdim后的维度
        self.assertEqual(stats.shape, (2, 1, 1, 1))
        # 检查第一个位置的最小值和最大值
        self.assertEqual(stats[0, 0, 0, 0].item(), 1.0)  # 全局最小值
        self.assertEqual(stats[1, 0, 0, 0].item(), 8.0)  # 全局最大值
        
    def test_keep_dim(self):
        # 创建一个3维张量
        x = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ])
        
        # 测试keep_dim=True（默认值）
        self.statistic = MinMaxStatistic()  # 重置统计对象
        self.statistic.update_stats(x, reduce_dims=[0], keep_dims=True)
        stats = self.statistic.get_stats()
        # 形状为[2, 1, 2, 2]，其中第一个维度是min/max，第二个维度是keepdim后的维度
        self.assertEqual(stats.shape, (2, 1, 2, 2))
        # 检查第一个位置的最小值和最大值
        self.assertEqual(stats[0, 0, 0, 0].item(), 1.0)  # 第一个批次的最小值
        self.assertEqual(stats[1, 0, 0, 0].item(), 5.0)  # 第一个批次的最大值
        
        # 测试keep_dim=False
        self.statistic = MinMaxStatistic()  # 重置统计对象
        self.statistic.update_stats(x, reduce_dims=[0], keep_dims=False)
        stats = self.statistic.get_stats()
        # 形状为[2, 2, 2]，其中第一个维度是min/max，后面是移除被缩减维度后的形状
        self.assertEqual(stats.shape, (2, 2, 2))
        # 检查第一个位置的最小值和最大值
        self.assertEqual(stats[0, 0, 0].item(), 1.0)  # 第一个批次的最小值
        self.assertEqual(stats[1, 0, 0].item(), 5.0)  # 第一个批次的最大值
        
        # 测试多个维度的keep_dim=True
        self.statistic = MinMaxStatistic()  # 重置统计对象
        self.statistic.update_stats(x, reduce_dims=[0, 2], keep_dims=True)
        stats = self.statistic.get_stats()
        # 形状为[2, 1, 2, 1]，其中第一个维度是min/max，第二和第四个维度是keepdim后的维度
        self.assertEqual(stats.shape, (2, 1, 2, 1))
        # 检查第一个位置的最小值和最大值
        self.assertEqual(stats[0, 0, 0, 0].item(), 1.0)  # 第一维度和第三维度缩减后的最小值
        self.assertEqual(stats[1, 0, 0, 0].item(), 6.0)  # 第一维度和第三维度缩减后的最大值
        
        # 测试多个维度的keep_dim=False
        self.statistic = MinMaxStatistic()  # 重置统计对象
        self.statistic.update_stats(x, reduce_dims=[0, 2], keep_dims=False)
        stats = self.statistic.get_stats()
        # 形状为[2, 2]，其中第一个维度是min/max，后面是移除被缩减维度后的形状
        self.assertEqual(stats.shape, (2, 2))
        # 检查第一个位置的最小值和最大值
        self.assertEqual(stats[0, 0].item(), 1.0)  # 第一维度和第三维度缩减后的最小值
        self.assertEqual(stats[1, 0].item(), 6.0)  # 第一维度和第三维度缩减后的最大值


if __name__ == '__main__':
    unittest.main()
