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

from msit.common.constants import CompConst
from msit.core.probe.dump.onnx_model import OnnxDataDumper, OnnxDataWriter
from msit.base.component.manager import ProducerComp, ConsumerComp, Component


@Component.register(CompConst.ONNXREADER)
class OnnxReader(ProducerComp, OnnxDataDumper):
    def __init__(self, args):
        ProducerComp.__init__(self)
        OnnxDataDumper.__init__(self, args)

    def activate(self):
        self.is_loaded = False

    def load_data(self):
        if not self.is_loaded:
            inputs_tensor_info = self.get_input_tensor_info()
            input_map = self.get_inputs_data(inputs_tensor_info)
            self.recapture_input_data()
            self.is_loaded = True
            return input_map
        else:
            return None


@Component.register(CompConst.ONNXACTUATOR)
class OnnxActuator(ProducerComp, ConsumerComp, OnnxDataDumper):
    def __init__(self, args):
        ProducerComp.__init__(self)
        ConsumerComp.__init__(self)
        OnnxDataDumper.__init__(self, args)

    def load_data(self):
        pass

    def consume(self, packargs):
        new_model_path = self.export_new_model()
        dump_data = self.run_model(new_model_path, packargs[0][2])
        self.publish(dump_data)


@Component.register(CompConst.ONNXWRITER)
class OnnxWriter(ConsumerComp, OnnxDataWriter):
    def __init__(self, args):
        ConsumerComp.__init__(self)
        OnnxDataWriter.__init__(self, args)

    def consume(self, packargs):
        self.summ_output_data(packargs[0][2])
