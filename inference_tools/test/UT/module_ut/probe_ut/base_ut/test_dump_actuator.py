import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from msit.module.probe.base import OfflineModelActuator
from msit.utils.exceptions import MsitException


class TestOfflineModelActuator(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()
        self.mock_dependent = MagicMock()
        self.mock_DirPool = MagicMock()
        self.mock_save_npy = MagicMock()
        self.mock_load_npy = MagicMock()
        self.mock_load_bin = MagicMock()

        self.patcher1 = patch("msit.module.probe.base.dump_actuator.logger", self.mock_logger)
        self.patcher2 = patch("msit.module.probe.base.dump_actuator.dependent", self.mock_dependent)
        self.patcher3 = patch("msit.module.probe.base.dump_actuator.DirPool", self.mock_DirPool)
        self.patcher4 = patch("msit.module.probe.base.dump_actuator.save_npy", self.mock_save_npy)
        self.patcher5 = patch("msit.module.probe.base.dump_actuator.load_npy", self.mock_load_npy)
        self.patcher6 = patch("msit.module.probe.base.dump_actuator.load_bin_data", self.mock_load_bin)

        self.patcher1.start()
        self.patcher2.start()
        self.patcher3.start()
        self.patcher4.start()
        self.patcher5.start()
        self.patcher6.start()

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        self.patcher6.stop()

    def test_is_dynamic_shape(self):
        self.assertFalse(OfflineModelActuator._is_dynamic_shape([1, 3, 224, 224]))
        self.assertTrue(OfflineModelActuator._is_dynamic_shape([None, 3, 224, 224]))
        self.assertTrue(OfflineModelActuator._is_dynamic_shape(["batch", 3, 224, 224]))

    def test_process_tensor_shape_dynamic_valid(self):
        actuator = OfflineModelActuator(
            model_path="model.onnx", input_shape={"input1": [1, 224, 224, 3]}, input_path=""
        )
        result = actuator.process_tensor_shape("input1", "tensor(float32)", [None, 224, 224, 3])
        expected = [{"name": "input1", "shape": [1, 224, 224, 3], "type": "tensor(float32)"}]
        self.assertEqual(result, expected)
        self.mock_logger.info.assert_called_with("The dynamic shape of input1 has been fixed to [1, 224, 224, 3].")

    def test_process_tensor_shape_dynamic_missing_input(self):
        actuator = OfflineModelActuator(model_path="model.onnx", input_shape={}, input_path="")
        with self.assertRaises(MsitException) as context:
            actuator.process_tensor_shape("input1", "tensor(float32)", [None, 224, 224, 3])
        self.assertIn("dynamic shape", str(context.exception))

    def test_check_input_shape_mismatch(self):
        with self.assertRaises(MsitException) as context:
            OfflineModelActuator._check_input_shape("input1", [1, 3, 224, 224], [1, 4, 224, 224])
        self.assertIn("does not match", str(context.exception))

    @patch("os.path.exists")
    def test_get_inputs_data_generate_random(self, mock_exists):
        mock_exists.return_value = True
        self.mock_DirPool.get_input_dir.return_value = "/mock/input"
        self.mock_dependent.get_tensorflow.return_value = (None, None, None)
        actuator = OfflineModelActuator(model_path="model.onnx", input_shape={}, input_path="")
        inputs_info = [{"name": "input1", "shape": [1, 3, 224, 224], "type": "tensor(float16)"}]
        with patch("numpy.random.random") as mock_random:
            mock_random.return_value = np.zeros((1, 3, 224, 224), dtype=np.float32)
            result = actuator.get_inputs_data(inputs_info)
        self.mock_save_npy.assert_called_once()
        self.assertIn("input1", result)
        self.assertEqual(result["input1"].shape, (1, 3, 224, 224))

    def test_get_inputs_data_load_existing(self):
        self.mock_dependent.get_tensorflow.return_value = (None, None, None)
        actuator = OfflineModelActuator(model_path="model.onnx", input_shape={}, input_path=["input1.npy"])
        inputs_info = [{"name": "input1", "shape": [1, 3, 224, 224], "type": "tensor(float16)"}]
        self.mock_load_npy.return_value = np.zeros((1, 3, 224, 224), dtype=np.float32)
        result = actuator.get_inputs_data(inputs_info)
        self.mock_load_npy.assert_called_with("input1.npy")
        self.assertEqual(result["input1"].shape, (1, 3, 224, 224))

    def test_read_input_shape_mismatch(self):
        self.mock_dependent.get_tensorflow.return_value = (None, None, None)
        actuator = OfflineModelActuator(model_path="model.onnx", input_shape={}, input_path=["input1.bin"])
        inputs_info = [{"name": "input1", "shape": [1, 3, 224, 224], "type": "tensor(float16)"}]
        self.mock_load_bin.return_value = np.zeros((2, 3, 224, 224))
        with self.assertRaises(MsitException) as context:
            actuator.get_inputs_data(inputs_info)
        self.assertIn("does not match", str(context.exception))

    def test_type_conversion(self):
        self.mock_dependent.get_tensorflow.return_value = (None, None, None)
        dtype = OfflineModelActuator._tensor2numpy_for_type("tensor(int32)")
        self.assertEqual(dtype, np.int32)

    def test_invalid_type_conversion(self):
        self.mock_dependent.get_tensorflow.return_value = (None, None, None)
        with self.assertRaises(MsitException) as context:
            OfflineModelActuator._tensor2numpy_for_type("tensor(unknown)")
        self.assertIn("invalid data type", str(context.exception))

    def test_valid_static_shape(self):
        OfflineModelActuator._check_input_shape("input1", model_shape=[1, 3, 224, 224], input_shape=[1, 3, 224, 224])

    def test_missing_input_shape(self):
        with self.assertRaises(MsitException) as ctx:
            OfflineModelActuator._check_input_shape("input1", model_shape=[1, 3, 224, 224], input_shape=[])
        self.assertIn("Required argument missing", str(ctx.exception))

    def test_dimension_mismatch(self):
        with self.assertRaises(MsitException) as ctx:
            OfflineModelActuator._check_input_shape("input1", model_shape=[1, 3, 224, 224], input_shape=[1, 3, 224])
        self.assertIn("Unequal lengths", str(ctx.exception))

    def test_dynamic_dimension_skip(self):
        OfflineModelActuator._check_input_shape("input1", model_shape=[None, 3, 224, 224], input_shape=[2, 3, 224, 224])
        OfflineModelActuator._check_input_shape(
            "input1", model_shape=["batch", 3, 224, 224], input_shape=[4, 3, 224, 224]
        )

    def test_static_shape_processing(self):
        actuator = OfflineModelActuator(model_path="model.onnx", input_shape={}, input_path="")
        result = actuator.process_tensor_shape(
            tensor_name="input1", tensor_type="tensor(float16)", tensor_shape=[1, 3, 224, 224]
        )
        self.assertEqual(result, [{"name": "input1", "shape": [1, 3, 224, 224], "type": "tensor(float16)"}])
