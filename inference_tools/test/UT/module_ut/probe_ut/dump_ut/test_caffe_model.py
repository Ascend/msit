import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from msit.module.probe.dump.caffe_model import CaffeModelActuator, CaffeModelDataWriter
from msit.utils.exceptions import MsitException


class TestCaffeModelActuator(unittest.TestCase):
    @patch("msit.module.probe.dump.caffe_model.load_caffe_model")
    def test_load_model(self, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        actuator = CaffeModelActuator(
            model_path="model.prototxt",
            input_shape=(1, 3, 224, 224),
            input_path="input.npy",
            weight_path="model.caffemodel",
        )
        actuator.load_model()
        mock_load_model.assert_called_once_with("model.prototxt", "model.caffemodel")
        self.assertEqual(actuator.model, mock_model)

    def test_missing_weight_raises_exception(self):
        with self.assertRaises(MsitException) as context:
            CaffeModelActuator(model_path="model.prototxt", input_shape=(1, 3, 224, 224), input_path="input.npy")
        self.assertIn("a weight file", str(context.exception))

    @patch("msit.module.probe.dump.caffe_model.logger")
    def test_get_input_tensor_info(self, mock_logger):
        mock_model = MagicMock()
        mock_blob = MagicMock()
        mock_blob.data = np.zeros((1, 3, 224, 224), dtype=np.float32)
        mock_model.blobs = {"data": mock_blob}
        mock_model.inputs = ["data"]

        actuator = CaffeModelActuator(
            model_path="model.prototxt",
            input_shape=(1, 3, 224, 224),
            input_path="input.npy",
            weight_path="model.caffemodel",
        )
        actuator.model = mock_model
        result = actuator.get_input_tensor_info()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "data")
        self.assertEqual(result[0]["shape"], (1, 3, 224, 224))
        self.assertEqual(result[0]["type"], "float32")

    def test_infer_success(self):
        actuator = CaffeModelActuator(
            model_path="model.prototxt",
            input_shape=(1, 3, 224, 224),
            input_path="input.npy",
            weight_path="model.caffemodel",
        )

        mock_model = MagicMock()
        mock_blob = MagicMock()
        mock_blob.data = np.zeros((1, 3, 224, 224))
        mock_model.blobs = {"data": mock_blob}
        mock_model.forward.return_value = {"prob": np.array([1.0])}
        actuator.model = mock_model

        input_data = {"data": np.zeros((1, 3, 224, 224))}
        result = actuator.infer(input_data)
        self.assertIn("prob", result)

    def test_infer_failure(self):
        actuator = CaffeModelActuator(
            model_path="model.prototxt",
            input_shape=(1, 3, 224, 224),
            input_path="input.npy",
            weight_path="model.caffemodel",
        )
        mock_model = MagicMock()
        mock_blob = MagicMock()
        mock_blob.data = np.zeros((1, 3, 224, 224))
        mock_model.blobs = {"data": mock_blob}
        mock_model.forward.side_effect = RuntimeError("Failure")
        actuator.model = mock_model
        with self.assertRaises(MsitException):
            actuator.infer({"data": np.zeros((1, 3, 224, 224))})


class TestCaffeModelDataWriter(unittest.TestCase):

    def test_get_input_output_map(self):
        writer = CaffeModelDataWriter(task="mock_task", dump_mode=["input", "output"])
        mock_net = MagicMock()
        mock_net.blobs = {"conv1": MagicMock(data=np.ones((1, 3, 224, 224)))}
        mock_net.params = {"conv1": [MagicMock(data=np.ones((64, 3, 3, 3))), MagicMock(data=np.ones((64,)))]}
        mock_net.bottom_names = {"conv1": []}
        input_map, output_map = writer.get_input_output_map(mock_net)
        self.assertIn("conv1_weight", input_map)
        self.assertIn("conv1_bias", input_map)
        self.assertIn("conv1", output_map)

    @patch("msit.module.probe.base.dump_writer.save_json")
    @patch("msit.module.probe.dump.onnx_model.DirPool.get_model_dir", return_value="/mock/model/dir")
    @patch("msit.module.probe.dump.onnx_model.get_valid_name")
    def test_summ_dump_data(self, mock_get_valid_name, mock_model_dir, mock_save_json):
        mock_get_valid_name.side_effect = lambda name: f"valid_{name}"
        writer = CaffeModelDataWriter(task="mock_task", dump_mode=["input", "output"])
        mock_net = MagicMock()
        mock_net.outputs = ["fc"]
        mock_net.blobs = {"fc": MagicMock(data=np.ones((1, 1000)))}
        mock_net.top_names = {"fc": ["fc_out"]}
        mock_net.bottom_names = {"fc": ["fc_in"]}
        writer.caffe_net = mock_net
        writer.through_inputs = MagicMock()
        writer.through_outputs = MagicMock()
        input_map = {"fc_weight": np.ones((1000, 512)), "fc_bias": np.ones((1000,))}
        output_map = {"fc": np.ones((1, 1000))}
        writer.summ_dump_data(input_map, output_map)
        writer.through_inputs.assert_called()
        writer.through_outputs.assert_called()
