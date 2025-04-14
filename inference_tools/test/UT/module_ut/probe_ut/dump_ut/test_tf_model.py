import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from msit.module.probe.dump.tf_model import (
    FrozenGraphActuator,
    FrozenGraphActuatorCPU,
    FrozenGraphActuatorNPU,
    FrozenGraphDataWriter,
)
from msit.utils.exceptions import MsitException


class TestFrozenGraphActuator(unittest.TestCase):

    @patch("msit.module.probe.dump.tf_model.dependent.get_tensorflow")
    def setUp(self, mock_get_tf):
        mock_tf = MagicMock()
        mock_rewriter_config = MagicMock()
        mock_get_tf.return_value = (mock_tf, mock_rewriter_config, None)
        self.actuator = FrozenGraphActuator(
            model_path="fake_model.pb", input_shape=(1, 224, 224, 3), input_path="input.npy"
        )
        self.tf = mock_tf
        self.actuator.tf = mock_tf

    @patch("msit.module.probe.dump.tf_model.dependent.get_tensorflow")
    def test_import_tf_success(self, mock_get_tf):
        mock_tf = MagicMock()
        mock_rewriter = MagicMock()
        mock_get_tf.return_value = (mock_tf, mock_rewriter, "extra")
        tf, rewriter = FrozenGraphActuator._import_tf()
        self.assertEqual(tf, mock_tf)
        self.assertEqual(rewriter, mock_rewriter)
        mock_tf.compat.v1.disable_eager_execution.assert_called_once()

    @patch("msit.module.probe.dump.tf_model.dependent.get_tensorflow")
    def test_import_tf_none(self, mock_get_tf):
        mock_get_tf.return_value = (None, None, None)
        tf, rewriter = FrozenGraphActuator._import_tf()
        self.assertIsNone(tf)
        self.assertIsNone(rewriter)

    @patch("msit.module.probe.dump.tf_model.load_pb_frozen_graph_model")
    def test_load_model(self, mock_load_pb):
        mock_graph_def = MagicMock()
        mock_load_pb.return_value = mock_graph_def
        self.actuator.load_model()
        mock_load_pb.assert_called_once_with("fake_model.pb")
        self.assertEqual(self.actuator.graph_def, mock_graph_def)

    def test_get_tensor_name(self):
        name = FrozenGraphActuator._get_tensor_name("input:0")
        self.assertEqual(name, "input")
        name = FrozenGraphActuator._get_tensor_name("no_colon")
        self.assertEqual(name, "no_colon")

    def test_tf_shape_to_list(self):
        mock_shape = MagicMock()
        dim1 = MagicMock(size=1)
        dim2 = MagicMock(size=-1)
        dim3 = MagicMock(size=3)
        mock_shape.dim = [dim1, dim2, dim3]
        result = FrozenGraphActuator._tf_shape_to_list(mock_shape)
        self.assertEqual(result, [1, None, 3])

    def test_get_input_tensor_info(self):
        mock_dtype = MagicMock()
        mock_dtype.type = 1
        mock_tensor_shape = MagicMock()
        mock_tensor_shape.dim = [MagicMock(size=1), MagicMock(size=224)]
        node = MagicMock()
        node.name = "input_node"
        node.op = "Placeholder"
        node.attr = {"dtype": mock_dtype, "shape": MagicMock(shape=mock_tensor_shape)}
        self.actuator.graph_def = MagicMock()
        self.actuator.graph_def.node = [node]
        self.actuator.tf.dtypes.as_dtype.return_value = "float32"
        self.actuator.process_tensor_shape = MagicMock(
            return_value=[{"name": "input_node", "shape": [1, 224], "type": "float32"}]
        )
        result = self.actuator.get_input_tensor_info()
        self.assertEqual(len(result), 1)
        self.assertIn("input_node", self.actuator.all_node_names)

    def test_close_session(self):
        mock_sess = MagicMock()
        self.actuator.sess = mock_sess
        self.actuator.close()
        mock_sess.close.assert_called_once()
        self.assertIsNone(self.actuator.sess)

    def test_close_session_no_attr(self):
        self.actuator.sess = None
        try:
            self.actuator.close()
        except Exception as e:
            self.fail(f"close() raised an exception unexpectedly: {e}")

    def test_get_tf_ops_success(self):
        self.actuator.all_node_names = ["input"]
        mock_graph = MagicMock()
        tensor = MagicMock()
        mock_graph.get_tensor_by_name.return_value = tensor

        self.actuator.sess = MagicMock()
        self.actuator.sess.graph = mock_graph

        ops = self.actuator._get_tf_ops()
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0], tensor)

    def test_get_tf_ops_failure(self):
        self.actuator.all_node_names = ["bad_node"]
        self.actuator.sess = MagicMock()
        self.actuator.sess.graph.get_tensor_by_name.side_effect = Exception("fail")

        with self.assertRaises(MsitException):
            self.actuator._get_tf_ops()

    def test_build_feed_success(self):
        tensor = MagicMock()
        input_map = {"input": np.ones((1, 224, 224, 3))}

        self.actuator.sess = MagicMock()
        self.actuator.sess.graph.get_tensor_by_name.return_value = tensor

        feed_dict = self.actuator._build_feed(input_map)
        self.assertEqual(feed_dict[tensor].shape, (1, 224, 224, 3))

    def test_build_feed_failure(self):
        input_map = {"bad_input": np.zeros((1,))}

        self.actuator.sess = MagicMock()
        self.actuator.sess.graph.get_tensor_by_name.side_effect = Exception("fail")

        with self.assertRaises(MsitException):
            self.actuator._build_feed(input_map)

    def test_infer_success(self):
        mock_sess = MagicMock()
        mock_sess.run.return_value = ["result"]
        self.actuator._open_session = MagicMock(return_value=mock_sess)
        self.actuator._renew_all_node_names = MagicMock()
        self.actuator._get_tf_ops = MagicMock(return_value=["fake_op"])
        self.actuator._build_feed = MagicMock(return_value={"input": "fake_data"})
        self.actuator.close = MagicMock()
        result = self.actuator.infer({"input": "data"})
        self.assertEqual(result, ["result"])
        mock_sess.run.assert_called_once()

    def test_infer_failure(self):
        mock_sess = MagicMock()
        mock_sess.run.side_effect = RuntimeError("bad inference")
        self.actuator._open_session = MagicMock(return_value=mock_sess)
        self.actuator._renew_all_node_names = MagicMock()
        self.actuator._get_tf_ops = MagicMock(return_value=["fake_op"])
        self.actuator._build_feed = MagicMock(return_value={"input": "fake_data"})
        self.actuator.close = MagicMock()
        with self.assertRaises(MsitException) as context:
            self.actuator.infer({"input": "data"})
        self.assertIn("input shape or data", str(context.exception))


class TestFrozenGraphActuatorCPU(unittest.TestCase):
    @patch("msit.module.probe.dump.tf_model.FrozenGraphActuator._import_tf")
    def test_open_session(self, mock_import_tf):
        mock_tf = MagicMock()
        mock_tf.compat.v1.Session.return_value = "mock_session"
        mock_import_tf.return_value = (mock_tf, MagicMock())
        actuator = FrozenGraphActuatorCPU("model", {}, "input")
        session = actuator._open_session()
        self.assertEqual(session, "mock_session")


class TestFrozenGraphActuatorNPU(unittest.TestCase):
    @patch("msit.module.probe.dump.tf_model.DirPool.get_rank_dir", return_value="/mock/rank_dir")
    @patch("msit.module.probe.dump.tf_model.FrozenGraphActuator._import_tf")
    @patch("msit.module.probe.dump.tf_model.dependent.get")
    def test_open_session_npu(self, mock_dependent_get, mock_import_tf, mock_rank_dir):
        mock_tf = MagicMock()
        mock_import_tf.return_value = (mock_tf, MagicMock())
        mock_device = MagicMock()
        mock_device.compat.enable_v1 = MagicMock()
        mock_dependent_get.return_value = mock_device
        mock_tf.compat.v1.ConfigProto.return_value = MagicMock()
        mock_tf.compat.v1.Session.return_value = "npu_session"

        actuator = FrozenGraphActuatorNPU("model", {}, "input")
        session = actuator._open_session()
        self.assertEqual(session, "npu_session")

    @patch("msit.module.probe.dump.tf_model.cann.model2json")
    @patch("msit.module.probe.dump.tf_model.get_name_and_ext")
    @patch("msit.module.probe.dump.tf_model.glob")
    @patch("msit.module.probe.dump.tf_model.DirPool.get_model_dir")
    def test_convert_txt2json(self, mock_get_model_dir, mock_glob, mock_get_name_ext, mock_model2json):
        mock_get_model_dir.return_value = "/mock/dir"
        mock_glob.return_value = ["/mock/dir/mock_Build.txt"]
        mock_get_name_ext.return_value = ("mock", ".txt")
        actuator = FrozenGraphActuatorNPU("model", {}, "input")
        actuator.convert_txt2json()
        mock_model2json.assert_called_once()


class TestFrozenGraphDataWriter(unittest.TestCase):
    def setUp(self):
        self.task = MagicMock()
        self.dump_mode = ["all"]
        self.writer = FrozenGraphDataWriter(self.task, self.dump_mode)
        self.writer.cache_dump_json = {"data": {}}
        self.writer.through_inputs = MagicMock()
        self.writer.through_outputs = MagicMock()

    @patch("msit.module.probe.dump.tf_model.get_valid_name")
    def test_get_output_map(self, mock_get_valid_name):
        tf_ops = [MagicMock(name="Tensor1"), MagicMock(name="Tensor2")]
        infer_output = ["out1", "out2"]
        mock_get_valid_name.side_effect = lambda x: f"valid_{x}"

        output_map = self.writer._get_output_map(tf_ops, infer_output)
        self.assertEqual(output_map, {f"valid_{tf_ops[0].name}": "out1", f"valid_{tf_ops[1].name}": "out2"})

    @patch("msit.module.probe.dump.tf_model.get_valid_name")
    @patch("msit.module.probe.dump.tf_model.logger")
    def test_get_input_map(self, mock_logger, mock_get_valid_name):
        # Mock input tensors
        tensor_input = MagicMock()
        tensor_input.name = "input_tensor:0"

        tf_op = MagicMock()
        tf_op.op.name = "nodeA"
        tf_op.op.inputs = [tensor_input]

        output_map = {"input_tensor": "data123"}
        mock_get_valid_name.side_effect = lambda x: x.split(":")[0]

        input_map = self.writer._get_input_map([tf_op], output_map)
        self.assertEqual(input_map, {"input_tensor": "data123"})

    @patch("msit.module.probe.dump.tf_model.FrozenGraphDataWriter._get_output_map")
    @patch("msit.module.probe.dump.tf_model.FrozenGraphDataWriter._get_input_map")
    def test_get_input_output_map(self, mock_get_input_map, mock_get_output_map):
        tf_ops = ["op1"]
        infer_output = ["output1"]
        mock_get_output_map.return_value = {"x": "y"}
        mock_get_input_map.return_value = {"a": "b"}
        input_map, output_map = self.writer.get_input_output_map(tf_ops, infer_output)
        self.assertEqual(input_map, {"a": "b"})
        self.assertEqual(output_map, {"x": "y"})

    @patch("msit.module.probe.base.dump_writer.save_json")
    @patch("msit.module.probe.dump.onnx_model.DirPool.get_model_dir", return_value="/mock/model/dir")
    @patch("msit.module.probe.dump.tf_model.get_net_output_nodes_from_graph_def")
    @patch("msit.module.probe.dump.tf_model.get_valid_name")
    def test_summ_dump_data(self, mock_get_valid_name, mock_get_net_output_nodes, mock_model_dir, mock_save_json):
        node_mock = MagicMock()
        node_mock.name = "nodeA"
        node_mock.op.inputs = ["input1"]
        node_mock.op.outputs = ["output1"]
        mock_get_valid_name.side_effect = lambda x: f"valid_{x}"
        mock_get_net_output_nodes.return_value = ["output_node"]
        self.writer.dump_mode = ["input", "output"]
        self.writer.cache_dump_json = {"data": {}}
        self.writer.summ_dump_data([node_mock], {"input1": "data"}, {"output1": "out"}, MagicMock())
        self.writer.through_inputs.assert_called_once()
        self.writer.through_outputs.assert_called_once()
        self.assertIn("valid_nodeA", self.writer.cache_dump_json["data"])
