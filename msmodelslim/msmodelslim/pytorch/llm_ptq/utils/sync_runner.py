# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import abc
import threading

from functools import partial
from typing import Dict

import torch

from msmodelslim import logger


class SyncRunner:

    def __init__(self, model: torch.nn.Module,
                 sync_count: int,
                 target_module=torch.nn.Linear):

        """
        基于多线程对多个前向过程在特定模块进行同步。通过注册torch hook并在hook中插入barrier操作实现。

        Args:
            model: 运行前向的模型
            sync_count: 同步线程数，与校准级数量保持一直
            target_module: 需要同步的模块
        """

        self.model = model
        self.sync_barriers: Dict[str, threading.Barrier] = {}
        self.sync_lock = threading.Lock()
        self.sync_hooks = []
        self.sync_count = sync_count
        self.target_type = target_module
        self.is_enable = True

    def setup(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Module) and isinstance(module, self.target_type):
                self.sync_barriers[name] = threading.Barrier(self.sync_count)

                def do_sync(module, args, kwargs, output, name):

                    if not self.is_enable:
                        return None

                    self.pre_sync(module, args, kwargs, output, name)

                    self.sync_lock.release()
                    logger.debug(f"Thread {threading.current_thread().name} layer {name} wait for barrier")
                    self.sync_barriers[name].wait()
                    self.sync_lock.acquire()

                    return self.post_sync(module, args, kwargs, output, name)

                self.sync_hooks.append(module.register_forward_hook(partial(do_sync, name=name), with_kwargs=True))

    def run(self, dataset_calib):
        @torch.no_grad()
        def run_forward_task(model, input):
            self.sync_lock.acquire()

            if isinstance(input, tuple) or isinstance(input, list):
                model(*input)
            elif isinstance(input, dict):
                model(**input)

            self.sync_lock.release()

        threads = []

        for i, one_data in zip(range(len(dataset_calib)), dataset_calib):
            threads.append(threading.Thread(target=run_forward_task,
                                            args=(self.model, one_data),
                                            name=f"Calib-Thread-{i}"))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def clear(self):

        """
        移除该对象注册的所有hook。
        """

        self.reset_barrier()

        for hook in self.sync_hooks:
            hook.remove()

    def reset_barrier(self):

        """
        重置所有的Barrier。
        """

        _ = [barrier.reset() for _, barrier in self.sync_barriers.items()]

    def no_sync_hook(self):

        """
        禁用同步hook，如果要在pre_sync和post_sync中调用module的forward，需要禁用同步hook防止递归。
        """

        class NoSyncHook:

            def __init__(self, runner: SyncRunner):
                self.runner = runner
                self.old_enable = runner.is_enable

            def __enter__(self):
                self.old_enable = self.runner.is_enable
                self.runner.is_enable = False

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.runner.is_enable = self.old_enable

        return NoSyncHook(self)

    @abc.abstractmethod
    def pre_sync(self, module, args, kwargs, output, name):

        """
        Barrier.wait()之前的动作，你可以在该函数返回新的output。
        注意，如果要在该函数中调用module的forward过程，需要使用with self.no_sync_hook()避免递归。
        """

        pass

    @abc.abstractmethod
    def post_sync(self, module, args, kwargs, output, name):

        """
        Barrier.wait()之后的动作，你可以在该函数返回新的output。
        注意，如果要在该函数中调用module的forward过程，需要使用with self.no_sync_hook()避免递归。
        """

        pass
