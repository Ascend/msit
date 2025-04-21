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
from msit.module.probe.base import BaseDumper
from msit.module.probe.components.dumper_offline_model import OfflineModelActuatorComp
from msit.module.probe.dump import FrozenGraphActuatorCPU, FrozenGraphActuatorNPU, FrozenGraphDataWriter
from msit.utils.constants import CompConst, DumpConst
from msit.utils.env import evars
from msit.utils.hijack import POST_HOOK, PRE_HOOK, hijacker


@Component.register(CompConst.FROZEN_GRAPH_ACTUATOR_COMP_CPU)
class FrozenGraphActuatorCompCPU(OfflineModelActuatorComp):
    def __init__(self, priority, model_path, input_shape, input_path, **kwargs):
        super().__init__(priority)
        self.actuator = FrozenGraphActuatorCPU(model_path, input_shape, input_path, **kwargs)

    def activate(self, *args, **kwargs):
        self.actuator.load_model()
        inputs_tensor_info = self.actuator.get_input_tensor_info()
        input_map = self.actuator.get_inputs_data(inputs_tensor_info)
        _ = self.actuator.infer(input_map)


@Component.register(CompConst.FROZEN_GRAPH_DUMPER_COMP_CPU)
class FrozenGraphDumperCompCPU(ProducerComp, BaseDumper):
    def __init__(self, priority):
        ProducerComp.__init__(self, priority)
        BaseDumper.__init__(self)
        self.graph_def = None
        self.infer_output = []
        self.tf_ops = []

    def activate(self, *args, **kwargs):
        self.register_hook()

    def deactivate(self, *args, **kwargs):
        self.release_hook()

    def register_hook(self):
        self.handler.append(
            hijacker(
                stub=self._capture_graph_def,
                module="tensorflow.python.framework.importer",
                function="_import_graph_def_internal",
                action=PRE_HOOK,
                priority=20,
            )
        )
        self.handler.append(
            hijacker(
                stub=self._capture_tf_ops,
                module="tensorflow.python.client.session",
                cls="Session",
                function="run",
                action=PRE_HOOK,
                priority=20,
            )
        )
        self.handler.append(
            hijacker(
                stub=self._capture_output,
                module="tensorflow.python.client.session",
                cls="Session",
                function="run",
                action=POST_HOOK,
                priority=25,
            )
        )

    def load_data(self):
        ret = self.tf_ops, self.infer_output, self.graph_def
        if all(ret):
            self.tf_ops, self.infer_output, self.graph_def = [], [], None
            return ret
        return None

    def _capture_graph_def(self, *args, **kwargs):
        self.graph_def = args[0]
        return args, kwargs

    def _capture_tf_ops(self, *args, **kwargs):
        self.tf_ops = args[1]
        return args, kwargs

    def _capture_output(self, output, *args, **kwargs):
        self.infer_output = output
        return output


@Component.register(CompConst.FROZEN_GRAPH_WRITER_COMP_CPU)
class FrozenGraphWriterCompCPU(ConsumerComp):
    def __init__(self, priority, task, data_mode):
        super().__init__(priority)
        self.writer = FrozenGraphDataWriter(task, data_mode)

    def consume(self, packages):
        tf_ops, infer_output, graph_def = packages[0][1]
        input_map, output_map = self.writer.get_input_output_map(tf_ops, infer_output)
        self.writer.summ_dump_data(tf_ops, input_map, output_map, graph_def)


@Component.register(CompConst.FROZEN_GRAPH_ACTUATOR_COMP_NPU)
class FrozenGraphActuatorCompNPU(OfflineModelActuatorComp):
    def __init__(self, priority, model_path, input_shape, input_path, **kwargs):
        super().__init__(priority)
        self.actuator = FrozenGraphActuatorNPU(model_path, input_shape, input_path, **kwargs)

    def activate(self, *args, **kwargs):
        self.actuator.load_model()
        inputs_tensor_info = self.actuator.get_input_tensor_info()
        input_map = self.actuator.get_inputs_data(inputs_tensor_info)
        _ = self.actuator.infer(input_map)
        self.actuator.convert_txt2json()


@Component.register(CompConst.FROZEN_GRAPH_SET_GE_COMP_NPU)
class FrozenGraphSetGECompNPU(BaseComponent):
    def __init__(self, priority, work_path, dump_ge_graph, dump_graph_level, dump_graph_path):
        super().__init__(priority)
        self.work_path = work_path
        self.dump_ge_graph = dump_ge_graph
        self.dump_graph_level = dump_graph_level
        self.dump_graph_path = dump_graph_path

    def activate(self, *args, **kwargs):
        evars.set(DumpConst.ENVVAR_ASCEND_WORK_PATH, self.work_path)
        evars.set(DumpConst.ENVVAR_DUMP_GE_GRAPH, self.dump_ge_graph)
        evars.set(DumpConst.ENVVAR_DUMP_GRAPH_LEVEL, self.dump_graph_level)
        evars.set(DumpConst.ENVVAR_DUMP_GRAPH_PATH, self.dump_graph_path)
