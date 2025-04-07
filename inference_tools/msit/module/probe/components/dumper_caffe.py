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

from msit.base import Component, ConsumerComp, OfflineModelActuatorComp, ProducerComp
from msit.core.probe.base import BaseDumper
from msit.core.probe.dump import CaffeModelActuator, CaffeModelDataWriter
from msit.utils.constants import CompConst
from msit.utils.hijack import POST_HOOK, hijacker


@Component.register(CompConst.CAFFE_ACTUATOR_COMP)
class CaffeActuatorComp(OfflineModelActuatorComp):
    def __init__(self, priority, model_path, input_shape, input_path, **kwargs):
        super().__init__(priority)
        self.actuator = CaffeModelActuator(model_path, input_shape, input_path, **kwargs)

    def activate(self, *args, **kwargs):
        self.actuator.load_model()
        inputs_tensor_info = self.actuator.get_input_tensor_info()
        input_map = self.actuator.get_inputs_data(inputs_tensor_info)
        _ = self.actuator.infer(input_map)


@Component.register(CompConst.CAFFE_DUMPER_COMP)
class CaffeDumperComp(ProducerComp, BaseDumper):
    def __init__(self, priority):
        ProducerComp.__init__(self, priority)
        BaseDumper.__init__(self)
        self.caffe_net = None

    def activate(self, *args, **kwargs):
        self.register_hook()

    def deactivate(self, *args, **kwargs):
        self.release_hook()

    def register_hook(self):
        self.handler.append(hijacker(stub=self._capture_caffe_net, module="caffe", function="Net", action=POST_HOOK))

    def load_data(self):
        if self.caffe_net:
            ret = self.caffe_net
            self.caffe_net = None
            return ret
        return None

    def _capture_caffe_net(self, ret, *args, **kwargs):
        self.caffe_net = ret
        return ret


@Component.register(CompConst.CAFFE_WRITER_COMP)
class CaffeWriterComp(ConsumerComp):
    def __init__(self, priority, task, dump_mode):
        super().__init__(priority)
        self.writer = CaffeModelDataWriter(task, dump_mode)

    def consume(self, packages):
        caffe_net = packages[0][1]
        input_map, output_map = self.writer.get_input_output_map(caffe_net)
        self.writer.summ_dump_data(input_map, output_map)
