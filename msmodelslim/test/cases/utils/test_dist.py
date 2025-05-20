#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
import sys
import unittest

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from msmodelslim.utils.dist import DistHelper


def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def run_distributed_test(rank, world_size):
    """运行分布式测试"""
    setup_distributed(rank, world_size)

    # 创建一个简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)  # 共享模块
            self.layer2 = nn.Linear(20, 30)  # 共享模块
            if dist.get_rank() == 0:
                self.local_layer = nn.Linear(30, 40)  # rank 0的local_only模块
            else:
                self.other_layer = nn.Linear(30, 50)  # rank 1的local_only模块

    model = TestModel()
    dist_helper = DistHelper(model)

    # 执行测试
    assert len(list(dist_helper.local_modules())) > 0
    assert len(list(dist_helper.shared_modules())) > 0
    assert dist_helper.is_local('layer1')
    assert dist_helper.is_local('layer2')
    assert dist_helper.is_shared('layer1')
    assert dist_helper.is_shared('layer2')

    # 测试local_only模块
    if rank == 0:
        assert dist_helper.is_local_only('local_layer')
        assert not dist_helper.is_local_only('other_layer')
        assert not dist_helper.is_shared('local_layer')
    else:
        assert dist_helper.is_local_only('other_layer')
        assert not dist_helper.is_local_only('local_layer')
        assert not dist_helper.is_shared('other_layer')

    # 验证local_only模块列表
    local_only_modules = list(dist_helper.local_only_modules())
    assert len(local_only_modules) == 1
    if rank == 0:
        assert isinstance(local_only_modules[0], nn.Linear) and local_only_modules[0].out_features == 40
    else:
        assert isinstance(local_only_modules[0], nn.Linear) and local_only_modules[0].out_features == 50

    rank = dist_helper.get_rank()
    assert isinstance(rank, int)
    assert 0 <= rank < dist.get_world_size()

    # 获取当前rank的shared_slice
    shared_slice = dist_helper.get_shared_modules_slice()
    assert isinstance(shared_slice, list)
    assert len(shared_slice) > 0

    # 收集所有rank的shared_slice
    all_slices = [None] * world_size
    dist.all_gather_object(all_slices, shared_slice)

    # 在rank 0上验证所有slices的并集是否等于shared_modules
    if rank == 0:
        all_shared_modules = set.union(*all_slices)
        shared_modules_set = set(dist_helper._shared_modules)

        # 验证并集是否相等
        assert all_shared_modules == shared_modules_set, \
            f"所有rank的shared_slice并集 {all_shared_modules} 不等于完整的shared_modules {shared_modules_set}"

    cleanup_distributed()


class TestDistHelper(unittest.TestCase):
    @pytest.mark.skipif(sys.platform == "win32", reason="分布式测试在Windows上不支持")
    def test_distributed(self):
        """分布式测试入口"""
        world_size = 2
        mp.spawn(
            run_distributed_test,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )


if __name__ == '__main__':
    unittest.main()
