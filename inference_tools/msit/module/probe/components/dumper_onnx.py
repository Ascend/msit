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

from msit.base import Component, ConsumerComp, ProducerComp
from msit.module.probe.base import BaseDumper
from msit.module.probe.components.dumper_offline_model import OfflineModelActuatorComp
from msit.module.probe.dump import OnnxModelActuator, OnnxModelDataWriter
from msit.utils.constants import CompConst, MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.hijack import POST_HOOK, PRE_HOOK, hijacker


@Component.register(CompConst.ONNX_ACTUATOR_COMP)
class OnnxActuatorComp(OfflineModelActuatorComp):
    def __init__(self, priority, model_path, input_shape, input_path, **kwargs):
        super().__init__(priority)
        self.actuator = OnnxModelActuator(model_path, input_shape, input_path, **kwargs)

    def activate(self, *args, **kwargs):
        self.actuator.load_model()
        inputs_tensor_info = self.actuator.get_input_tensor_info()
        input_map = self.actuator.get_inputs_data(inputs_tensor_info)
        uninfer_model_path = self.actuator.export_uninfer_model()
        _ = self.actuator.infer(uninfer_model_path, input_map)


@Component.register(CompConst.ONNX_DUMPER_COMP)
class OnnxDumperComp(ProducerComp, BaseDumper):
    def __init__(self, priority):
        ProducerComp.__init__(self, priority)
        BaseDumper.__init__(self)
        self.input_map = {}
        self.output_list = []
        self.origin_model = None

    def activate(self, *args, **kwargs):
        self.register_hook()

    def deactivate(self, *args, **kwargs):
        self.release_hook()

    def register_hook(self):
        self.handler_session = hijacker(
            stub=self._capture_model_session,
            module="onnxruntime",
            cls="InferenceSession",
            function="__init__",
            action=PRE_HOOK,
            priority=10,
        )
        self.handler.append(self.handler_session)
        self.handler.append(
            hijacker(
                stub=self._capture_input_map,
                module="onnxruntime",
                cls="InferenceSession",
                function="run",
                action=PRE_HOOK,
                priority=10,
            )
        )
        self.handler.append(
            hijacker(
                stub=self._capture_output_list,
                module="onnxruntime",
                cls="InferenceSession",
                function="run",
                action=POST_HOOK,
                priority=20,
            )
        )
        self.handler.append(
            hijacker(
                stub=self._capture_origin_model, module="onnx", function="load_model", action=POST_HOOK, priority=20
            )
        )

    def load_data(self):
        model_session = self._get_model_session()
        ret = self.input_map, self.output_list, self.origin_model, model_session
        if all(ret):
            self.input_map, self.output_list, self.origin_model, model_session = ({}, [], None, None)
            return ret
        return None

    def _get_model_session(self):
        try:
            return self.handler_session.call_data.get(1).get("args")[0]
        except Exception as e:
            raise MsitException(
                MsgConst.VALUE_NOT_FOUND, "The hook function failed to capture the model_session."
            ) from e

    def _capture_model_session(self, *args, **kwargs):
        return args, kwargs

    def _capture_input_map(self, *args, **kwargs):
        self.input_map = args[2]
        return args, kwargs

    def _capture_output_list(self, output, *args, **kwargs):
        self.output_list = output
        return output

    def _capture_origin_model(self, origin_model, *args, **kwargs):
        self.origin_model = origin_model
        return origin_model


@Component.register(CompConst.ONNX_WRITER_COMP)
class OnnxWriterComp(ConsumerComp):
    def __init__(self, priority, task, dump_mode):
        super().__init__(priority)
        self.writer = OnnxModelDataWriter(task, dump_mode)

    def consume(self, packages):
        input_map, output_list, origin_model, model_session = packages[0][1]
        input_map, output_map = self.writer.get_input_output_map(input_map, output_list, origin_model)
        self.writer.summ_dump_data(input_map, output_map, origin_model, model_session)
