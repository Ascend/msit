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

from onnx import ValueInfoProto

from msit.common.log import logger
from msit.common.dirs import DirPool
from msit.common.constants import DumpConst
from msit.core.probe.base import DataDumper, DataWriter, DataStat
from msit.utils.toolkits import convert_bytes
from msit.utils.io import load_onnx_model, load_onnx_session, save_onnx_model, save_npy


class OnnxDataDumper(DataDumper):
    def __init__(self, args):
        super(OnnxDataDumper, self).__init__(args)
        self.origin_model = load_onnx_model(args.exec[0])
        self.model_session = load_onnx_session(args.exec[0], args.onnx_fusion_switch)
        self.dump_json = DataWriter.init_dump_json(task=self.args.task)

    def recapture_input_data(self):
        for name, input_data in self.input_map.items():
            DataWriter.update_dump_json(
                self.dump_json[DumpConst.DATA][DumpConst.INPUT_ARGS], name, DataStat.summ_npy(input_data)
            )

    def get_input_tensor_info(self):
        self.dump_json[DumpConst.DATA].setdefault(DumpConst.INPUT_ARGS, {})
        inputs_tensor_info = []
        for input_item in self.model_session.get_inputs():
            tensor_name, tensor_type, tensor_shape = input_item.name, input_item.type, tuple(input_item.shape)
            DataWriter.update_dump_json(
                self.dump_json[DumpConst.DATA][DumpConst.INPUT_ARGS], tensor_name, {DumpConst.TYPE: tensor_type}
            )
            tensor_shape_info = self._process_tensor_shape(tensor_name, tensor_type, tensor_shape)
            inputs_tensor_info.extend(tensor_shape_info)
        logger.info(f"Model input tensor info: {inputs_tensor_info}.")
        return inputs_tensor_info

    def export_new_model(self):
        del self.origin_model.graph.output[:]
        for node in self.origin_model.graph.node:
            for tensor_name in node.output:
                value_info = ValueInfoProto(name=tensor_name)
                self.origin_model.graph.output.append(value_info)
        model_size = self.origin_model.ByteSize()
        logger.info(f"The size of the modified ONNX model to be saved is {convert_bytes(model_size)}.")
        if model_size < 0 or model_size > DumpConst.MAX_PROTOBUF_2G:
            logger.warning("The modified ONNX model size has exceeded 2GB, posing a risk of numerical overflow.")
        new_model_path = DirPool.get_new_model_path(self.args.exec[0])
        save_onnx_model(self.origin_model, new_model_path)
        logger.info(f"The modified ONNX model has been successfully saved to {new_model_path}.")
        return new_model_path

    def run_model(self, new_model_path, input_map):
        new_model_session = load_onnx_session(new_model_path, self.args.onnx_fusion_switch)
        output_name = [node.name for node in new_model_session.get_outputs()]
        return new_model_session.run(output_name, input_map)


class OnnxDataWriter(DataWriter):
    def __init__(self, args):
        self.args = args
        self.origin_model = load_onnx_model(args.exec[0])
        self.model_session = load_onnx_session(args.exec[0], args.onnx_fusion_switch)

    @classmethod
    def _save_output_stat(cls, name, np_data):
        cls.update_dump_json(
            cls.cache_dump_json[DumpConst.DATA][DumpConst.OUTPUT_ARGS], cls._to_valid_name(name), \
            {**{DumpConst.TYPE: cls._to_valid_type(np_data)}, **DataStat.summ_npy(np_data)}
        )

    @classmethod
    def _save_output_data(cls, name, ind, npy_data):
        cls.cache_dump_json[DumpConst.DUMP_DATA_DIR] = DirPool.get_tensor_dir()
        rank_dir = DirPool.get_rank_dir()
        file_name = cls._generate_tensor_name(name, ind)
        cls.tensor_path = cls._generate_tensor_path(rank_dir, file_name)
        save_npy(npy_data, cls.tensor_path)

    def summ_output_data(self, dump_data):
        self.cache_dump_json[DumpConst.DATA].setdefault(DumpConst.OUTPUT_ARGS, {})
        net_output_node = [item.name for item in self.model_session.get_outputs()]
        res_idx, net_output = 0, {}
        for node in self.origin_model.graph.node:
            for j, output in enumerate(node.output):
                self._save_output_stat(node.name, dump_data[res_idx])
                if self.args.task == DumpConst.TENSOR:
                    self._save_output_data(node.name, j, dump_data[res_idx])
                if self.args.task == DumpConst.TENSOR and output in net_output_node:
                    net_output[net_output_node.index(output)] = self.tensor_path
                res_idx += 1
        for index, path in net_output.items():
            logger.info(f"net_output node is: {index}, file: {path}.")
