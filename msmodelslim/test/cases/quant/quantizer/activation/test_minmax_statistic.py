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


import os
import unittest

import torch
import torch.distributed as dist

from msmodelslim.quant.quantizer.activation.minmax import MinMaxStrategy, ActMinMaxConfig
from testing_utils.dist import distributed_test


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def run_distributed_test(rank, world_size):
    setup_distributed(rank, world_size)

    # 创建统计器
    config = ActMinMaxConfig()
    statistic = MinMaxStrategy(config=config)

    # 每个进程处理不同的数据
    if rank == 0:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    else:
        x = torch.tensor([[0.5, 5.0], [2.0, 3.0]])

    # 更新统计数据
    statistic.update(x)

    # 同步所有进程
    dist.barrier()

    # 获取全局统计结果
    min_val, max_val = statistic.get_min_max()

    # 验证结果
    assert min_val.item() == 0.5  # 全局最小值
    assert max_val.item() == 5.0  # 全局最大值

    cleanup_distributed()


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

    @distributed_test(world_size=2)
    def test_distributed_update(self, rank, world_size):
        # 创建统计器
        config = ActMinMaxConfig()
        statistic = MinMaxStrategy(config=config)

        # 每个进程处理不同的数据
        if rank == 0:
            x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        else:
            x = torch.tensor([[0.5, 5.0], [2.0, 3.0]])

        # 更新统计数据
        statistic.update(x)

        # 同步所有进程
        torch.distributed.barrier()

        # 获取全局统计结果
        min_val, max_val = statistic.get_min_max()

        # 验证结果
        assert min_val.item() == 0.5  # 全局最小值
        assert max_val.item() == 5.0  # 全局最大值


if __name__ == '__main__':
    unittest.main()
