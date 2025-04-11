import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from msit.utils.dependencies import DependencyManager, temporary_tf_log_level
from msit.utils.exceptions import MsitException


class TestDependencyManager(unittest.TestCase):
    def setUp(self):
        DependencyManager._instance = None
        self.manager = DependencyManager()

    def tearDown(self):
        if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
            del os.environ["TF_CPP_MIN_LOG_LEVEL"]

    @patch.dict(os.environ, {"TF_CPP_MIN_LOG_LEVEL": "0"})
    def test_temporary_tf_log_level(self):
        @temporary_tf_log_level
        def mock_function():
            return os.environ["TF_CPP_MIN_LOG_LEVEL"]

        self.assertEqual(mock_function(), "2")
        self.assertEqual(os.environ["TF_CPP_MIN_LOG_LEVEL"], "0")

    @patch("msit.utils.dependencies.import_module")
    def test_get_tensorflow(self, mock_import_module):
        mock_tf = MagicMock()
        mock_tf.__version__ = "2.6.5"
        mock_rewriter_config = MagicMock()
        mock_convert_variables = MagicMock()

        def side_effect(name):
            if name == "tensorflow":
                return mock_tf
            return MagicMock()

        mock_import_module.side_effect = side_effect
        sys.modules["tensorflow"] = mock_tf
        sys.modules["tensorflow.core.protobuf.rewriter_config_pb2"] = MagicMock(RewriterConfig=mock_rewriter_config)
        sys.modules["tensorflow.python.framework.graph_util"] = MagicMock(
            convert_variables_to_constants=mock_convert_variables
        )
        dm = DependencyManager()
        tf, re_writer_config, sm2pb = dm.get_tensorflow()

        self.assertIsNotNone(tf, "TensorFlow is not None")
        self.assertEqual(tf, mock_tf)
        self.assertEqual(re_writer_config, mock_rewriter_config)
        self.assertEqual(sm2pb, mock_convert_variables)

    @patch("msit.utils.dependencies.import_module")
    def test_import_package_non_tensorflow(self, mock_import):
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        result = self.manager._import_package("abc")
        mock_import.assert_called_once_with("abc")
        self.assertEqual(result, mock_module)
        self.assertIn("abc", self.manager._dependencies)

    @patch.object(DependencyManager, "_import_tensorflow")
    def test_import_package_tensorflow(self, mock_import_tf):
        mock_tf = MagicMock()

        def simulate_import():
            self.manager._dependencies["tensorflow"] = mock_tf
            return mock_tf

        mock_import_tf.side_effect = simulate_import
        result = self.manager._import_package("tensorflow")
        mock_import_tf.assert_called_once()
        self.assertEqual(result, mock_tf)
        self.assertIn("tensorflow", self.manager._dependencies)

    @patch("msit.utils.dependencies.import_module")
    def test_import_tensorflow_wrong_version(self, mock_import):
        mock_tf = MagicMock()
        mock_tf.__version__ = "2.7.0"
        mock_import.return_value = mock_tf
        with self.assertRaises(MsitException) as context:
            self.manager._import_tensorflow()
        self.assertIn("Incompatible versions", str(context.exception))

    @patch("msit.utils.dependencies.import_module")
    def test_import_tensorflow_environment_reset(self, mock_import):
        original_level = "0"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = original_level
        mock_tf = MagicMock()
        mock_tf.__version__ = "2.6.5"
        mock_import.return_value = mock_tf
        self.manager._import_tensorflow()
        self.assertEqual(os.environ["TF_CPP_MIN_LOG_LEVEL"], original_level)

    @patch("msit.utils.dependencies.import_module")
    def test_import_package_missing_dependency(self, mock_import):
        mock_import.side_effect = ImportError("No module named 'missing_package'")
        result = self.manager._import_package("missing_package")
        self.assertIsNone(result)
        self.assertNotIn("missing_package", self.manager._dependencies)

    @patch("msit.utils.dependencies.logger.warning")
    @patch("msit.utils.dependencies.import_module")
    def test_safely_import_decorator(self, mock_import, mock_warning):
        mock_import.side_effect = ImportError("Test error")
        result = self.manager._import_package("test_package")
        self.assertIsNone(result)
        mock_warning.assert_called_once_with("test_package is not installed. Please install it if needed.")
        mock_warning.reset_mock()
        result = self.manager._import_package("test_package")
        self.assertIsNone(result)
        mock_warning.assert_not_called()
