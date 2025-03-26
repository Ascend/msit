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

from msit.utils.constants import MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.toolkits import register


class BaseComponent(object):
    """
    Methods that need to be implemented:
        activate: Called when service.start() is invoked.
        deactivate: Called when service.stop() is invoked.
    """

    def __init__(self, priority=100):
        self.activated = False
        self.priority = priority

    @property
    def is_activated(self):
        return self.activated

    def activate(self, *args, **kwargs):
        pass

    def deactivate(self, *args, **kwargs):
        pass

    def do_activate(self):
        if self.activated:
            return
        self.activate()
        self.activated = True

    def do_deactivate(self):
        if not self.activated:
            return
        self.deactivate()
        self.activated = False


class ProducerComp(BaseComponent, ABC):
    """
    A ProducerComp can generate data.
        If the data is passively generated (e.g., when a consumer applies the data), implement "load_data".
        If the data is actively generated (e.g., when an interest event occurs),
            call "publish" to send it to subscribers.
    """

    def __init__(self, priority):
        super(ProducerComp, self).__init__(priority)
        self.output_buffer = None
        self.subscribers = set()

    @property
    def is_ready(self):
        return self.output_buffer is not None

    @abstractmethod
    def load_data(self):
        pass

    def publish(self, data, msg_id=0):
        """
        Wrap the data and pack it into the output buffer.
        """
        self.output_buffer = [self, data, msg_id]
        Scheduler().schedule([self])

    def on_subscribe(self, comp):
        if not isinstance(comp, ConsumerComp):
            raise MsitException(MsgConst.INVALID_DATA_TYPE, "Only ConsumerComp can subscribe to ProducerComp.")
        self.subscribers.add(comp)

    def retrieve(self):
        ret = self.output_buffer
        self.output_buffer = None
        return ret

    def do_load_data(self):
        if self.output_buffer is not None:
            return
        data = self.load_data()
        if data:
            self.publish(data)

    def get_subscribers(self):
        return self.subscribers


class ConsumerComp(BaseComponent, ABC):
    """
    A ConsumerComp can consume data.
    Call "subscribe" to subscribe data from a ProducerComp.
    Implement "consume" to process data.
    """

    def __init__(self, priority):
        super(ConsumerComp, self).__init__(priority)
        self.dependencies = {}

    def subscribe(self, comp):
        if not isinstance(comp, ProducerComp):
            raise MsitException(MsgConst.INVALID_DATA_TYPE, "Only ProducerComp can subscribe to ConsumerComp.")
        if self.is_activated:
            raise MsitException(MsgConst.INVALID_DATA_TYPE, f"Component {comp} must be subscribed before activation.")
        if self.is_cycle(comp):
            raise MsitException(MsgConst.RISK_ALERT, "Cycle dependency detected! Subscription denied.")
        comp.on_subscribe(self)
        if comp not in self.dependencies:
            self.dependencies[comp] = None

    @abstractmethod
    def consume(self, packages):
        pass

    def is_cycle(self, comp, visited=None, stack=None):
        if visited is None:
            visited = set()
        if stack is None:
            stack = set()
        if comp in stack:
            return True
        if comp in visited:
            return False
        visited.add(comp)
        stack.add(comp)
        if isinstance(comp, ConsumerComp):
            for producer in comp.dependencies:
                if self.is_cycle(producer, visited, stack):
                    return True
        stack.remove(comp)
        return False

    def on_receive(self, package):
        try:
            self.dependencies[package[0]] = package
        except Exception as e:
            raise MsitException(
                MsgConst.PARSING_FAILED,
                "The first element in the data (self.output_buffer) published by the producer must be itself.",
            ) from e

    def get_empty_dependencies(self):
        dependencies_list = []
        for k, v in self.dependencies.items():
            if v is None:
                dependencies_list.append(k)
        return dependencies_list

    def do_consume(self):
        """
        Encapsulate the data in "dependencies" and invoke it using "consume".
        """
        if self.get_empty_dependencies():
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
        return register(name, cls._component_type_map)

    @classmethod
    def get(cls, name):
        return cls._component_type_map.get(name)


class Scheduler(object):
    def __init__(self):
        self.comp_ref = {}
        self.comps_to_schedule = set()
        self.in_scheduling = False

    def add(self, components):
        for comp in components:
            if comp in self.comp_ref:
                self.comp_ref[comp] += 1
            else:
                self.comp_ref[comp] = 1
                comp.do_activate()
        self.schedule(components)

    def remove(self, components):
        for comp in components:
            if comp not in self.comp_ref:
                continue
            if self.comp_ref[comp] > 1:
                self.comp_ref[comp] -= 1
            else:
                comp.do_deactivate()
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
        if not comp.is_ready:
            return
        package = comp.retrieve()
        subscribers = comp.get_subscribers()
        if not subscribers:
            return
        for subscriber in subscribers:
            subscriber.on_receive(package)
            self.comps_to_schedule.add(subscriber)

    def _schedule_consumer(self, comp: ConsumerComp):
        dependencies = comp.get_empty_dependencies()
        if not dependencies:
            comp.do_consume()
            self.comps_to_schedule.add(comp)
            return
        for dependency in dependencies:
            dependency.do_load_data()
            if dependency.is_ready:
                self.comps_to_schedule.add(dependency)
