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

from functools import wraps
from abc import ABC, abstractmethod

from msit.common.constants import MsgConst
from msit.common.exceptions import MsitException


class BaseComponent(object):
    """
    Methods that need to be implemented:
        activate: Called when service.start() is invoked.
        deactivate: Called when service.stop() is invoked.
    """
    def __init__(self):
        self.activative = False

    @property
    def is_activative(self):
        return self.activative

    def activate(self, *args, **kwargs):
        pass

    def deactivate(self, *args, **kwargs):
        pass

    def _activate(self):
        if self.activative:
            return 
        self.activate()
        self.activative = True

    def _deactivate(self):
        if not self.activative:
            return 
        self.deactivate()
        self.activative = False


class ProducerComp(BaseComponent, ABC):
    """
    A ProducerComp can generate data.
        If the data is passively generated (e.g., when a consumer applies the data), implement "load_data".
        If the data is actively generated (e.g., when an interest event occurs), 
            call "publish" to send it to subscribers.
    """
    def __init__(self):
        super(ProducerComp, self).__init__()
        self.output_buffer = None
        self.subscribers = set()

    @property
    def _is_ready(self):
        return self.output_buffer is not None

    @abstractmethod
    def load_data(self):
        pass

    def publish(self, data, msg_id=0):
        """
        Wrap the data and pack it into the output buffer.
        """
        self.output_buffer = [self, msg_id, data]
        Scheduler().schedule([self])

    def _on_subscribe(self, comp):
        if not isinstance(comp, ConsumerComp):
            raise MsitException(MsgConst.INVALID_DATA_TYPE, "Only ConsumerComp can subscribe to ProducerComp.")
        self.subscribers.add(comp)

    def _retrieve(self):
        ret = self.output_buffer
        self.output_buffer = None
        return ret

    def _load_data(self):
        if self.output_buffer is not None:
            return 
        data = self.load_data()
        if data:
            self.publish(data)

    def _get_subscribers(self):
        return self.subscribers


class ConsumerComp(BaseComponent, ABC):
    """
    A ConsumerComp can consume data.
    Call "subscribe" to subscribe data from a ProducerComp.
    Implement "consume" to process data.
    """
    def __init__(self):
        super(ConsumerComp, self).__init__()
        self.dependencies = {}

    def subscribe(self, comp):
        if not isinstance(comp, ProducerComp):
            raise MsitException(MsgConst.INVALID_DATA_TYPE, "Only ProducerComp can subscribe to ConsumerComp.")
        if self.is_activative:
            raise MsitException(MsgConst.INVALID_DATA_TYPE, f"The component {comp} has been activated.")
        comp._on_subscribe(self)
        if comp not in self.dependencies:
            self.dependencies[comp] = None

    @abstractmethod
    def consume(self, packages):
        pass

    def _on_receive(self, package):
        self.dependencies[package[0]] = package

    def _get_empty_dependencies(self):
        dependencies_list = []
        for k, v in self.dependencies.items():
            if v is None:
                dependencies_list.append(k)
        return dependencies_list

    def _consume(self):
        """
        Encapsulate the data in "dependencies" and invoke it using "consume".
        """
        if self._get_empty_dependencies():
            return 
        packages = []
        for key in self.dependencies:
            packages.append(self.dependencies[key])
            self.dependencies[key] = None
        self.consume(packages)


class Component:
    _component_type_map = {}

    @classmethod
    def register(cls, name):
        @wraps(name)
        def wrapper(comp_type):
            cls._component_type_map[name] = comp_type
            return comp_type
        return wrapper

    @classmethod
    def get(cls, name):
        return cls._component_type_map.get(name)


class Scheduler(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Scheduler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, MsgConst.INITIALIZED):
            self.comp_ref = {}
            self.comps_to_schedule = set()
            self.in_scheduling = False
            self.initialized = True

    def add(self, components):
        for comp in components:
            if comp in self.comp_ref:
                self.comp_ref[comp] += 1
            else:
                self.comp_ref[comp] = 1
                comp._activate()
        self.schedule(components)

    def remove(self, components):
        for comp in components:
            if comp not in self.comp_ref:
                continue
            if self.comp_ref[comp] > 1:
                self.comp_ref[comp] -= 1
            else:
                comp._deactivate()
                del self.comp_ref[comp]

    def schedule(self, comps_to_schedule=None):
        if not comps_to_schedule:
            comps_to_schedule = set(self.comp_ref.keys())
        if self.in_scheduling:
            self.comps_to_schedule = self.comps_to_schedule.union(set(comps_to_schedule))
            return 
        self.in_scheduling = True
        self.comps_to_schedule = set(comps_to_schedule)
        while self.comps_to_schedule:
            comps = self.comps_to_schedule
            self.comps_to_schedule = set()
            for comp in comps:
                if isinstance(comp, ProducerComp):
                    self._schedule_producter(comp)
                if isinstance(comp, ConsumerComp):
                    self._schedule_consumer(comp)
        self.in_scheduling = False

    def _schedule_producter(self, comp: ProducerComp):
        if not comp._is_ready:
            return 
        package = comp._retrieve()
        subscribers = comp._get_subscribers()
        if not subscribers:
            return 
        for subscriber in subscribers:
            subscriber._on_receive(package)
            self.comps_to_schedule.add(subscriber)

    def _schedule_consumer(self, comp: ConsumerComp):
        dependencies = comp._get_empty_dependencies()
        if not dependencies:
            comp._consume()
            self.comps_to_schedule.add(comp)
            return 
        for dependency in dependencies:
            dependency._load_data()
            if dependency._is_ready:
                self.comps_to_schedule.add(dependency)
