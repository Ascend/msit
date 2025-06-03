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

import functools
import os
import sys

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_distributed(rank, world_size):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def distributed_test(world_size=2):
    """
    分布式测试装饰器
    
    Args:
        world_size (int): 分布式进程数量，默认为2
        
    Returns:
        decorator: 装饰器函数
    """

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            @pytest.mark.skipif(sys.platform == "win32", reason="分布式测试在Windows上不支持")
            def run_distributed_test(rank, world_size):
                setup_distributed(rank, world_size)
                try:
                    test_func(rank, world_size, *args, **kwargs)
                finally:
                    cleanup_distributed()

            mp.spawn(
                run_distributed_test,
                args=(world_size,),
                nprocs=world_size,
                join=True
            )

        return wrapper

    return decorator
