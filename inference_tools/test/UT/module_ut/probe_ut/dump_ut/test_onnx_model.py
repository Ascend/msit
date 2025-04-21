import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from msit.module.probe.dump import OnnxModelActuator, OnnxModelDataWriter
from msit.utils.exceptions import MsitException


class TestOnnxModelActuator(unittest.TestCase):
    @patch("msit.module.probe.dump.onnx_model.load_onnx_session")
    def test_infer_success(self, mock_load_session):
        mock_output_node = MagicMock()
        mock_output_node.name = "output1"
        mock_session = MagicMock()
        mock_session.get_outputs.return_value = [mock_output_node]
        mock_session.run.return_value = ["dummy_output"]
        mock_load_session.return_value = mock_session
        result = OnnxModelActuator.infer("dummy_path", {"input1": np.array([1])})
        mock_session.run.assert_called_once_with(["output1"], {"input1": np.array([1])})
        self.assertEqual(result, ["dummy_output"])

    @patch("msit.module.probe.dump.onnx_model.load_onnx_session")
    def test_infer_failure(self, mock_load_session):
        mock_session = MagicMock()
        mock_session.get_outputs.return_value = [MagicMock(name="output1")]
        mock_session.run.side_effect = Exception("Runtime error")
        mock_load_session.return_value = mock_session
        with self.assertRaises(MsitException) as context:
            OnnxModelActuator.infer("dummy_path", {})
        self.assertIn("Please check if the input shape", str(context.exception))

    @patch("msit.module.probe.dump.onnx_model.load_onnx_model")
    @patch("msit.module.probe.dump.onnx_model.load_onnx_session")
    def test_load_model(self, mock_load_session, mock_load_model):
        actuator = OnnxModelActuator("model_path", None, None)
        actuator.load_model()

        mock_load_model.assert_called_once_with("model_path")
        mock_load_session.assert_called_once_with("model_path", True)
        self.assertEqual(actuator.model_session, mock_load_session.return_value)

    def test_get_input_tensor_info(self):
        actuator = OnnxModelActuator("model_path", None, None)
        mock_input = MagicMock()
        mock_input.name = "input1"
        mock_input.type = "tensor(float32)"
        mock_input.shape = [1, 3, 224, 224]
        actuator.model_session = MagicMock()
        actuator.model_session.get_inputs.return_value = [mock_input]

        result = actuator.get_input_tensor_info()
        self.assertIn("input1", str(result))

    @patch("msit.module.probe.dump.onnx_model.DirPool.get_uninfer_model_path")
    @patch("msit.module.probe.dump.onnx_model.is_file", return_value=False)
    @patch("msit.module.probe.dump.onnx_model.dependent.get")
    @patch("msit.module.probe.dump.onnx_model.convert_bytes", return_value="10MB")
    @patch("msit.module.probe.dump.onnx_model.save_onnx_model")
    @patch("msit.module.probe.dump.onnx_model.logger")
    def test_export_uninfer_model(
        self,
        mock_logger,
        mock_save_model,
        mock_convert_bytes,
        mock_dependent_get,
        mock_is_file,
        mock_get_uninfer_model_path,
    ):
        mock_get_uninfer_model_path.return_value = "/fake/path/model_uninfer.onnx"
        mock_onnx = MagicMock()
        mock_value_info_proto = MagicMock()
        mock_onnx.ValueInfoProto.return_value = mock_value_info_proto
        mock_dependent_get.return_value = mock_onnx
        fake_graph = MagicMock()
        fake_graph.node = [MagicMock(output=["out1", "out2"])]
        fake_graph.output = []
        actuator = OnnxModelActuator("model.onnx", (1, 3, 224, 224), "input.npy")
        actuator.origin_model = MagicMock()
        actuator.origin_model.graph = fake_graph
        actuator.origin_model.ByteSize.return_value = 1024 * 1024 * 10
        result_path = actuator.export_uninfer_model()
        self.assertEqual(result_path, "/fake/path/model_uninfer.onnx")
        self.assertEqual(len(fake_graph.output), 2)
        mock_save_model.assert_called_once_with(actuator.origin_model, "/fake/path/model_uninfer.onnx")
        mock_logger.info.assert_any_call("The size of the modified ONNX model to be saved is 10MB.")
        mock_logger.info.assert_any_call(
            "The modified ONNX model has been successfully saved to /fake/path/model_uninfer.onnx."
        )


class TestOnnxModelDataWriter(unittest.TestCase):
    def setUp(self):
        self.task_mock = MagicMock()
        self.dump_mode = ["all"]
        self.writer = OnnxModelDataWriter(self.task_mock, self.dump_mode)

    @patch("msit.module.probe.dump.onnx_model.get_valid_name")
    def test_get_output_map(self, mock_get_valid_name):
        mock_model = MagicMock()
        mock_model.graph.node = [MagicMock(output=["out1"]), MagicMock(output=["out2"])]
        mock_get_valid_name.side_effect = lambda name: f"valid_{name}"
        output_list = ["data1", "data2"]
        output_map = self.writer._get_output_map(output_list, mock_model)
        expected = {"valid_out1": "data1", "valid_out2": "data2"}
        self.assertEqual(output_map, expected)

    @patch("msit.module.probe.dump.onnx_model.load_npy_from_buffer")
    @patch("msit.module.probe.dump.onnx_model.get_valid_name")
    def test_augment_input_map(self, mock_get_valid_name, mock_load_npy):
        mock_model = MagicMock()
        initializer = MagicMock()
        initializer.name = "init1"
        initializer.raw_data = b"data"
        initializer.data_type = 1
        initializer.dims = [1]
        mock_model.graph.initializer = [initializer]
        mock_get_valid_name.return_value = "valid_init1"
        mock_load_npy.return_value = np.array([123])
        input_map = {"input1": np.array([1])}
        output_map = {"output1": np.array([2])}
        result = self.writer._augment_input_map(input_map, output_map, mock_model)
        expected_keys = {"input1", "valid_init1", "output1"}
        self.assertTrue(expected_keys.issubset(result.keys()))

    @patch.object(OnnxModelDataWriter, "_augment_input_map")
    @patch.object(OnnxModelDataWriter, "_get_output_map")
    def test_get_input_output_map(self, mock_get_output_map, mock_augment_input_map):
        input_map = {"a": 1}
        output_list = ["out"]
        origin_model = MagicMock()
        mock_get_output_map.return_value = {"out1": 100}
        mock_augment_input_map.return_value = {"a": 1, "out1": 100}
        result_input, result_output = self.writer.get_input_output_map(input_map, output_list, origin_model)
        self.assertEqual(result_output, {"out1": 100})
        self.assertEqual(result_input, {"a": 1, "out1": 100})

    @patch("msit.module.probe.base.dump_writer.save_json")
    @patch("msit.module.probe.dump.onnx_model.DirPool.get_model_dir", return_value="/mock/model/dir")
    @patch("msit.module.probe.dump.onnx_model.get_valid_name")
    def test_summ_dump_data(self, mock_get_valid_name, mock_model_dir, mock_save_json):
        mock_model_session = MagicMock()
        mock_output_info = MagicMock()
        mock_output_info.name = "output_node_1"
        mock_model_session.get_outputs.return_value = [mock_output_info]
        mock_origin_model = MagicMock()
        node1 = MagicMock()
        node1.name = "node1"
        node1.input = ["input1"]
        node1.output = ["output1"]
        mock_origin_model.graph.node = [node1]
        mock_get_valid_name.side_effect = lambda name: f"valid_{name}"
        input_map = {"valid_input1": "input_data"}
        output_map = {"valid_output1": "output_data"}
        self.writer.through_inputs = MagicMock()
        self.writer.through_outputs = MagicMock()
        self.writer.summ_dump_data(input_map, output_map, mock_origin_model, mock_model_session)
        self.writer.through_inputs.assert_called_once_with(["input1"], "node1", input_map)
        self.writer.through_outputs.assert_called_once_with(["output1"], "node1", output_map)
        self.assertIn("valid_node1", self.writer.cache_dump_json["data"])
