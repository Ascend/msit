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

import numpy as np

from msit.common.dirs import DirPool
from msit.module.probe.base import OfflineModelActuator, WriterDump
from msit.utils.constants import CfgConst, DumpConst, MsgConst, PathConst
from msit.utils.dependencies import dependent
from msit.utils.exceptions import MsitException
from msit.utils.io import load_npy_from_buffer, load_onnx_model, load_onnx_session, save_onnx_model
from msit.utils.log import logger
from msit.utils.path import convert_bytes, is_file
from msit.utils.toolkits import get_valid_name

_ONNX_DTYPE = {1: np.float32, 2: np.float64}


class OnnxModelActuator(OfflineModelActuator):
    def __init__(self, model_path, input_shape, input_path, **kwargs):
        super().__init__(model_path, input_shape, input_path, **kwargs)

    @staticmethod
    def infer(uninfer_model_path, input_map):
        model_session = load_onnx_session(uninfer_model_path)
        output_name = [node.name for node in model_session.get_outputs()]
        try:
            return model_session.run(output_name, input_map)
        except Exception as e:
            raise MsitException(
                MsgConst.CALL_FAILED, "Please check if the input shape or data matches the model requirements."
            ) from e

    def load_model(self):
        self.origin_model = load_onnx_model(self.model_path)
        self.model_session = load_onnx_session(self.model_path, self.kwargs.get("onnx_fusion_switch", True))

    def get_input_tensor_info(self):
        inputs_tensor_info = []
        for input_item in self.model_session.get_inputs():
            tensor_name, tensor_type, tensor_shape = (input_item.name, input_item.type, tuple(input_item.shape))
            tensor_shape_info = self.process_tensor_shape(tensor_name, tensor_type, tensor_shape)
            inputs_tensor_info.extend(tensor_shape_info)
        logger.info(f"Model input tensor info: {inputs_tensor_info}.")
        return inputs_tensor_info

    def export_uninfer_model(self):
        uninfer_model_path = DirPool.get_uninfer_model_path(self.model_path)
        if not is_file(uninfer_model_path):
            onnx = dependent.get("onnx")
            del self.origin_model.graph.output[:]
            for node in self.origin_model.graph.node:
                for tensor_name in node.output:
                    value_info = onnx.ValueInfoProto(name=tensor_name)
                    self.origin_model.graph.output.append(value_info)
            model_size = self.origin_model.ByteSize()
            logger.info(f"The size of the modified ONNX model to be saved is {convert_bytes(model_size)}.")
            if model_size < 0 or model_size > PathConst.SIZE_2G:
                logger.warning("The modified ONNX model size has exceeded 2GB, posing a risk of numerical overflow.")
            save_onnx_model(self.origin_model, uninfer_model_path)
            logger.info(f"The modified ONNX model has been successfully saved to {uninfer_model_path}.")
        return uninfer_model_path


class OnnxModelDataWriter(WriterDump):
    def __init__(self, dump_format, dump_mode):
        super().__init__(dump_format)
        self.dump_mode = dump_mode
        self.cache_dump_json[CfgConst.LEVEL] = CfgConst.LEVEL_KERNEL
        self.cache_dump_json[CfgConst.FRAMEWORK] = CfgConst.FRAMEWORK_ONNX

    @staticmethod
    def _get_output_map(output_list, origin_model):
        output_map, res_idx = {}, 0
        for node in origin_model.graph.node:
            for node_output in node.output:
                output_map[get_valid_name(node_output)] = output_list[res_idx]
                res_idx += 1
        return output_map

    @staticmethod
    def _augment_input_map(input_map, output_map, origin_model):
        for temp in origin_model.graph.initializer:
            npy_data = load_npy_from_buffer(temp.raw_data, _ONNX_DTYPE.get(temp.data_type), temp.dims)
            input_map[get_valid_name(temp.name)] = npy_data
        input_map = {**input_map, **output_map}
        return input_map

    def get_input_output_map(self, input_map, output_list, origin_model):
        output_map = self._get_output_map(output_list, origin_model)
        input_map = self._augment_input_map(input_map, output_map, origin_model)
        return input_map, output_map

    def summ_dump_data(self, input_map, output_map, origin_model, model_session):
        self.net_output_nodes = list(item.name for item in model_session.get_outputs())
        for node in origin_model.graph.node:
            self.cache_dump_json[DumpConst.DATA].setdefault(get_valid_name(node.name), {})
            if self.dump_mode in DumpConst.INPUT_ALL:
                self.through_inputs(node.input, node.name, input_map)
            if self.dump_mode in DumpConst.OUTPUT_ALL:
                self.through_outputs(node.output, node.name, output_map)
