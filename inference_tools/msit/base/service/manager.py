# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

from abc import abstractmethod

from msit.base.component.manager import BaseComponent, Scheduler
from msit.base.component.primordium import OfflineModelActuatorComp
from msit.common.dirs import DirPool
from msit.utils.toolkits import register


class ServiceProbe(object):
    def __init__(self):
        self.is_from_cmd = True
        self.comps = []
        self.current_iter = 0

    @abstractmethod
    def construct(self):
        pass

    def start(self, *args, **kwargs):
        DirPool.make_step_dir(self.current_iter)
        DirPool.make_rank_dir()
        self.construct()
        for attr in self.__dict__.values():
            if isinstance(attr, BaseComponent):
                self.comps.append(attr)
                self._ignore_actuator(attr)
        self.comps.sort(key=lambda x: x.priority)
        scheduler = Scheduler()
        scheduler.add(self.comps)
        scheduler.remove(self.comps)
        self.comps.clear()

    def step(self, *args, **kwargs):
        self.current_iter += 1

    def stop(self, *args, **kwargs):
        pass

    def _ignore_actuator(self, attr):
        if not self.is_from_cmd and isinstance(attr, OfflineModelActuatorComp):
            self.comps.remove(attr)


class Service:
    _services_map = {}

    @classmethod
    def register(cls, name):
        return register(name, cls._services_map)

    @classmethod
    def get(cls, name):
        return cls._services_map.get(name)
