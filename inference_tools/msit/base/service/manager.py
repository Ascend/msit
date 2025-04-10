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

from abc import ABC, abstractmethod

from msit.base import BaseComponent, Scheduler
from msit.common.validation import valid_task
from msit.utils.constants import CfgConst, CmdConst
from msit.utils.io import load_json
from msit.utils.toolkits import register

_TASK_SERVICE_MAP = {CfgConst.TASK_STAT: CmdConst.DUMP, CfgConst.TASK_TENSOR: CmdConst.DUMP}


class Service:
    _services_map = {}

    def __init__(self, config_path, **kwargs):
        config = load_json(config_path)
        task = valid_task(config.get(CfgConst.TASK))
        serv_name = _TASK_SERVICE_MAP.get(task)
        self.service_class = self.get(serv_name)
        self.service_instance = self.service_class(config_path, **kwargs)

    def __getattr__(self, name):
        return getattr(self.service_instance, name)

    @classmethod
    def register(cls, name):
        return register(name, cls._services_map)

    @classmethod
    def get(cls, name):
        return cls._services_map.get(name)


class BaseService(ABC):
    def __init__(self):
        self.comps = []

    @abstractmethod
    def construct(self):
        pass

    def start(self, *args, **kwargs):
        """
        Service startup workflow:
        1. Configure services (init_start).
        2. Build components (construct).
        3. Filter/prioritize components (ignore_actuator), then schedule execution.
        4. Schedule execution and cleanup.
        5. Post-processing (finalize_start).
        """
        self.init_start()
        self.construct()
        for attr in self.__dict__.values():
            if isinstance(attr, BaseComponent):
                self.comps.append(attr)
                self.ignore_actuator(attr)
        self.comps.sort(key=lambda x: x.priority)
        scheduler = Scheduler()
        scheduler.add(self.comps)
        scheduler.remove(self.comps)
        self.comps.clear()
        self.finalize_start()

    def init_start(self):
        pass

    def ignore_actuator(self, attr):
        pass

    def finalize_start(self):
        pass
