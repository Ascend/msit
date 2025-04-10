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

from msit.module.probe.base import OfflineModelActuator, WriterDump
from msit.utils.constants import CfgConst, DumpConst, MsgConst
from msit.utils.exceptions import MsitException
from msit.utils.io import load_caffe_model
from msit.utils.log import logger


class CaffeModelActuator(OfflineModelActuator):
    def __init__(self, model_path, input_shape, input_path, **kwargs):
        super().__init__(model_path, input_shape, input_path, **kwargs)
        self.weight_path = kwargs.get("weight_path", "")
        if not self.weight_path:
            raise MsitException(
                MsgConst.REQUIRED_ARGU_MISSING,
                "When using Caffe for inference, a weight file (.caffemodel) is required.",
            )

    def load_model(self):
        self.model = load_caffe_model(self.model_path, self.weight_path)

    def get_input_tensor_info(self):
        inputs_tensor_info = []
        input_blob_names = list(self.model.blobs.keys())[: len(self.model.inputs)]
        for input_name in input_blob_names:
            tensor_data = self.model.blobs[input_name].data
            tensor_info = {"name": input_name, "shape": tuple(tensor_data.shape), "type": str(tensor_data.dtype)}
            inputs_tensor_info.append(tensor_info)
        logger.warning(
            "Caffe model doesn't support dynamic shapes and "
            "will use the input shape defined in the model for inference."
        )
        logger.info(f"Model input tensor info: {inputs_tensor_info}.")
        return inputs_tensor_info

    def infer(self, input_map):
        try:
            for input_name, input_data in input_map.items():
                np.copyto(self.model.blobs[input_name].data, input_data)
            return self.model.forward()
        except Exception as e:
            raise MsitException(
                MsgConst.CALL_FAILED, "Please check if the input shape or data matches the model requirements."
            ) from e


class CaffeModelDataWriter(WriterDump):
    def __init__(self, task, dump_mode):
        super().__init__(task)
        self.dump_mode = dump_mode
        self.cache_dump_json[CfgConst.LEVEL] = CfgConst.LEVEL_LAYER
        self.cache_dump_json[CfgConst.FRAMEWORK] = CfgConst.FRAMEWORK_CAFFE
        self.caffe_net = None

    @staticmethod
    def _get_output_map(caffe_net):
        output_map = {}
        for layer_name, blob in caffe_net.blobs.items():
            output_map[layer_name] = blob.data
        return output_map

    def get_input_output_map(self, caffe_net):
        output_map = self._get_output_map(caffe_net)
        input_map = self._augment_input_map(caffe_net)
        return input_map, output_map

    def summ_dump_data(self, input_map, output_map):
        self.net_output_nodes = self.caffe_net.outputs
        for layer_name in self.caffe_net.blobs.keys():
            self.cache_dump_json[DumpConst.DATA].setdefault(layer_name, {})
            if any(x in self.dump_mode for x in DumpConst.INPUT_ALL):
                self.through_inputs(self.caffe_net.bottom_names.get(layer_name), layer_name, input_map)
            if any(x in self.dump_mode for x in DumpConst.OUTPUT_ALL):
                self.through_outputs(self.caffe_net.top_names.get(layer_name), layer_name, output_map)

    def _augment_input_map(self, caffe_net):
        input_map = {}
        for layer_name, param in caffe_net.params.items():
            if len(param) != 2:
                raise MsitException(
                    MsgConst.REQUIRED_ARGU_MISSING,
                    f"The current layer ({layer_name})'s input does not include weights and biases.",
                )
            input_map[f"{layer_name}_weight"] = param[0].data
            input_map[f"{layer_name}_bias"] = param[1].data
            caffe_net.bottom_names.get(layer_name).append(f"{layer_name}_weight")
            caffe_net.bottom_names.get(layer_name).append(f"{layer_name}_bias")
            self.caffe_net = caffe_net
        input_map = {**input_map, **self._get_output_map(caffe_net)}
        return input_map
