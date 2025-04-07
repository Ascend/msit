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

from msit.base import BaseComponent, Component, ConsumerComp, ProducerComp
from msit.core.probe.dump import AtbModelConfiguration


class AtbActuatorComp(BaseComponent):
    def __init__(self, priority, dump_path, **kwargs):
        super().__init__(priority)
        self.actuator = AtbModelConfiguration(
            dump_path,
            task=kwargs.get("task"),
            dump_level=kwargs.get("dump_level"),
            step=kwargs.get("step"),
            rank=kwargs.get("rank"),
            seed=kwargs.get("seed"),
            log_level=kwargs.get("log_level"),
            dump_extra=kwargs.get("dump_extra"),
            dump_time=kwargs.get("dump_time"),
            op_id=kwargs.get("op_id"),
            op_name=kwargs.get("op_name"),
            exec=kwargs.get("exec"),
        )

    def activate(self, *args, **kwargs):
        self.actuator.set_env_vars()
        self.actuator.execute_dump()
