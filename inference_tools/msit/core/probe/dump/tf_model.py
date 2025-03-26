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

from glob import glob

from msit.common.ascend import cann
from msit.common.dirs import DirPool
from msit.core.probe.base import OfflineModelActuator, WriterDump
from msit.utils.constants import DumpConst, MsgConst, PathConst
from msit.utils.dependencies import dependent
from msit.utils.exceptions import MsitException
from msit.utils.io import load_pb_frozen_graph_model
from msit.utils.log import logger
from msit.utils.path import get_name_and_ext, join_path
from msit.utils.toolkits import get_net_output_nodes_from_graph_def, get_valid_name


class FrozenGraphActuator(OfflineModelActuator):
    def __init__(self, model_path, input_shape, input_path, **kwargs):
        super().__init__(model_path, input_shape, input_path, **kwargs)
        self.tf, self.rewriter_config = self._import_tf()
        self.sess = None
        self.graph_def = None
        self.all_node_names = []

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    @staticmethod
    def _import_tf():
        pons = dependent.get_tensorflow()
        if None not in pons:
            tf, rewriter_config, _ = pons
            tf.compat.v1.disable_eager_execution()
            return tf, rewriter_config
        return None, None

    @staticmethod
    def _get_tensor_name(name: str):
        return name.split(":")[0]

    @staticmethod
    def _tf_shape_to_list(tensor_shape):
        shape_list = []
        for dim in tensor_shape.dim:
            if dim.size == -1:
                shape_list.append(None)
            else:
                shape_list.append(dim.size)
        return shape_list

    def get_input_tensor_info(self):
        inputs_tensor_info = []
        for node in self.graph_def.node:
            if node.op == "Placeholder":
                tensor_name = node.name
                tensor_dtype = self.tf.dtypes.as_dtype(node.attr["dtype"].type)
                tensor_shape = self._tf_shape_to_list(node.attr["shape"].shape)
                inputs_tensor_info.extend(self.process_tensor_shape(tensor_name, tensor_dtype, tensor_shape))
            self.all_node_names.append(node.name)
        logger.info(f"Model input tensor info: {inputs_tensor_info}.")
        return inputs_tensor_info

    def load_model(self):
        self.graph_def = load_pb_frozen_graph_model(self.model_path)

    def infer(self, input_map: dict):
        self.sess = self._open_session()
        self._renew_all_node_names()
        tf_ops = self._get_tf_ops()
        feed_dict = self._build_feed(input_map)
        try:
            outputs = self.sess.run(tf_ops, feed_dict=feed_dict)
        except Exception as e:
            raise MsitException(
                MsgConst.CALL_FAILED, "Please check if the input shape or data matches the model requirements."
            ) from e
        return outputs

    def _open_session(self):
        return

    def _renew_all_node_names(self):
        pass

    def _get_tf_ops(self):
        tf_ops = []
        for name in self.all_node_names:
            try:
                tf_ops.append(self.sess.graph.get_tensor_by_name(name + ":0"))
            except Exception as e:
                raise MsitException(
                    MsgConst.CALL_FAILED, f'The model lacks the {name + ":0"} node. Please check your model.'
                ) from e
        return tf_ops

    def _build_feed(self, input_map: dict):
        feed_dict = {}
        for name, input_data in input_map.items():
            tensor_name = name + ":0" if ":" not in name else name
            try:
                feed_dict[self.sess.graph.get_tensor_by_name(tensor_name)] = input_data
            except Exception as e:
                raise MsitException(
                    MsgConst.CALL_FAILED, f"The model lacks the {tensor_name} node. Please check your model."
                ) from e
        return feed_dict


class FrozenGraphActuatorCPU(FrozenGraphActuator):
    def __init__(self, model_path, input_shape, input_path, **kwargs):
        super().__init__(model_path, input_shape, input_path, **kwargs)

    def _open_session(self):
        return self.tf.compat.v1.Session()

    def _renew_all_node_names(self):
        pass


class FrozenGraphActuatorNPU(FrozenGraphActuator):
    def __init__(self, model_path, input_shape, input_path, **kwargs):
        super().__init__(model_path, input_shape, input_path, **kwargs)
        self.dump_mode = kwargs.get("dump_mode", "all")
        self.fusion_switch_file = kwargs.get("fsf", "")

    def convert_txt2json(self):
        model_path = sorted(glob(join_path(DirPool.get_model_dir(), "*", "*_Build.txt")))
        if model_path:
            name, _ = get_name_and_ext(model_path[-1])
            cann.model2json(model_path[-1], join_path(DirPool.get_model_dir(), name + PathConst.SUFFIX_JSON))
        else:
            raise MsitException(MsgConst.PATH_NOT_FOUND, "No TXT format graph file found in the TensorFlow framework.")

    def _open_session(self):
        npu_device = dependent.get("npu_device")
        if npu_device:
            npu_device.compat.enable_v1()
            config_proto = self.tf.compat.v1.ConfigProto()
            custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["enable_dump"].b = True
            custom_op.parameter_map["dump_path"].s = self.tf.compat.as_bytes(DirPool.get_rank_dir())
            custom_op.parameter_map["dump_step"].s = self.tf.compat.as_bytes("0")
            custom_op.parameter_map["dump_mode"].s = self.tf.compat.as_bytes(self.dump_mode)
            if self.fusion_switch_file:
                logger.info(f"Fusion switch settings read from {self.fusion_switch_file}.")
                custom_op.parameter_map["fusion_switch_file"].s = self.tf.compat.as_bytes(self.fusion_switch_file)
            config_proto.graph_options.rewrite_options.remapping = self.rewriter_config.OFF
            return self.tf.compat.v1.Session(config=config_proto)
        else:
            raise MsitException(
                MsgConst.ATTRIBUTE_ERROR, "Please ensure that the TF plugin npu_device is properly installed."
            )

    def _renew_all_node_names(self):
        self.all_node_names = get_net_output_nodes_from_graph_def(self.graph_def)


class FrozenGraphDataWriter(WriterDump):
    def __init__(self, task, dump_mode):
        super().__init__()
        self.task = task
        self.dump_mode = dump_mode
        self.cache_dump_json[DumpConst.TASK] = task
        self.cache_dump_json[DumpConst.LEVEL] = DumpConst.LEVEL_KERNEL
        self.cache_dump_json[DumpConst.FRAMEWORK] = DumpConst.FRAMEWORK_TF

    @staticmethod
    def _get_input_map(tf_ops, output_map):
        node_names = [tensor.op.name for tensor in tf_ops]
        input_map = {}
        for idx, node_name in enumerate(node_names):
            for input_tensor in tf_ops[idx].op.inputs:
                input_data = output_map.get(get_valid_name(input_tensor.name))
                if input_data is None:
                    logger.warning(f"Input {input_tensor.name} for {node_name} not found.")
                    continue
                input_map[get_valid_name(input_tensor.name)] = input_data
        return input_map

    @staticmethod
    def _get_output_map(tf_ops, infer_output):
        output_map = {get_valid_name(tensor.name): result for tensor, result in zip(tf_ops, infer_output)}
        return output_map

    def get_input_output_map(self, tf_ops, infer_output):
        output_map = self._get_output_map(tf_ops, infer_output)
        input_map = self._get_input_map(tf_ops, output_map)
        return input_map, output_map

    def summ_dump_data(self, tf_ops, input_map, output_map, graph_def):
        self.net_output_nodes = get_net_output_nodes_from_graph_def(graph_def)
        for node in tf_ops:
            self.cache_dump_json[DumpConst.DATA].setdefault(get_valid_name(node.name), {})
            if self.dump_mode in DumpConst.INPUT_ALL:
                self.through_inputs(node.op.inputs, node.name, input_map)
            if self.dump_mode in DumpConst.OUTPUT_ALL:
                self.through_outputs(node.op.outputs, node.name, output_map)
