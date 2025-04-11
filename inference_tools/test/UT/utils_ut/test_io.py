import os
import pickle
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd

from msit.utils.constants import MsgConst
from msit.utils.dependencies import dependent
from msit.utils.exceptions import MsitException
from msit.utils.io import (
    SafelyOpen,
    _load_dir,
    _load_file,
    _save_dir,
    _save_file,
    load_bin_to_ndarray,
    load_caffe_model,
    load_csv_by_builtin,
    load_csv_by_pandas,
    load_json,
    load_npy,
    load_npy_from_buffer,
    load_om_model,
    load_onnx_model,
    load_onnx_session,
    load_pb_frozen_graph_model,
    load_saved_model,
    load_torch_obj,
    load_yaml,
    save_bin_from_ndarray,
    save_csv_by_pandas,
    save_json,
    save_npy,
    save_onnx_model,
    save_pb_frozen_graph_model,
    save_yaml,
    savedmodel2pb,
)
from msit.utils.path import AUTHORITY_DIR, AUTHORITY_FILE, MsitPath, PathConst, change_permission


class TestSafelyOpen(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_existing_file(self):
        file_path = os.path.join(self.temp_path, "test.txt")
        with open(file_path, "w") as f:
            f.write("content")
        with SafelyOpen(file_path, "r", path_exist=True) as f:
            self.assertEqual(f.read(), "content")

    def test_write_new_file(self):
        file_path = os.path.join(self.temp_path, "new.txt")
        with SafelyOpen(file_path, "w", path_exist=False) as f:
            f.write("content")
        with open(file_path, "r") as f:
            self.assertEqual(f.read(), "content")

    def test_suffix_mismatch(self):
        file_path = os.path.join(self.temp_path, "file.csv")
        with open(file_path, "w") as f:
            f.write("data")
        with self.assertRaises(MsitException):
            SafelyOpen(file_path, "r", suffix=".txt", path_exist=True)

    def test_file_size_exceeded(self):
        file_path = os.path.join(self.temp_path, "large.txt")
        with open(file_path, "w") as f:
            f.write("a" * 1024)
        with self.assertRaises(MsitException):
            SafelyOpen(file_path, "r", file_size_limitation=512, path_exist=True)


class TestMsitPath(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_file_is_dir(self):
        dir_path = os.path.join(self.temp_path, "dir")
        os.mkdir(dir_path)
        with self.assertRaises(MsitException):
            MsitPath(dir_path, PathConst.FILE, "r").check(path_exist=True)

    def test_suffix_check(self):
        file_path = os.path.join(self.temp_path, "file.txt")
        with open(file_path, "w") as f:
            f.write("data")
        MsitPath(file_path, PathConst.FILE, "r", suffix=".txt").check()
        with self.assertRaises(MsitException):
            MsitPath(file_path, PathConst.FILE, "r", suffix=".csv").check()

    def test_file_size_limitation(self):
        file_path = os.path.join(self.temp_path, "file.txt")
        with open(file_path, "w") as f:
            f.write("a" * 1024)
        with self.assertRaises(MsitException):
            MsitPath(file_path, PathConst.FILE, "r", size_limitation=512).check()

    def test_path_existence(self):
        non_existent = os.path.join(self.temp_path, "nonexistent.txt")
        with self.assertRaises(MsitException):
            MsitPath(non_existent, PathConst.FILE, "r").check(path_exist=True)
        MsitPath(non_existent, PathConst.FILE, "w").check(path_exist=False)


class TestDecorators(unittest.TestCase):
    @staticmethod
    @_load_file("r", None, ".txt", True)
    def dummy_load(f):
        return f.read()

    def test_load_file_decorator_file_not_found(self):
        with self.assertRaises(MsitException):
            self.dummy_load("nonexistent.txt")

    @staticmethod
    @_save_file("w", None, ".txt", True)
    def dummy_save(data, f):
        f.write(data)

    @patch("msit.utils.path.MsitPath._check_write_permission_for_group_others")
    def test_save_file_decorator_success(self, mock_check):
        mock_check.return_value = None
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        temp_file.close()
        self.dummy_save("data", temp_path)
        with open(temp_path, "r") as f:
            self.assertEqual(f.read(), "data")
        os.unlink(temp_path)

    @patch("builtins.open")
    def test_save_file_decorator_permission_error(self, mock_open):
        mock_open.side_effect = PermissionError("Permission denied")
        with self.assertRaises(MsitException):
            self.dummy_save("data", "/unauthorized.txt")


class TestSaveDirDecorator(unittest.TestCase):
    @patch("msit.utils.io.MsitPath.check")
    @patch("msit.utils.io.change_permission")
    def test_save_dir_success(self, mock_change_perm, mock_msit_path):
        mock_path_instance = MagicMock()
        mock_msit_path.return_value = mock_path_instance

        @_save_dir(dir_size=1024)
        def test_func(data, path, *args, **kwargs):
            return

        result = test_func("test_data", "/test/path")
        mock_msit_path.assert_called_once_with(path_exist=False)
        mock_change_perm.assert_called_once_with(mock_path_instance, AUTHORITY_DIR)
        self.assertEqual(result, None)

    @patch("msit.utils.io.MsitPath.check")
    @patch("msit.utils.io.change_permission")
    def test_save_dir_exception_handling(self, mock_change_perm, mock_msit_path):
        mock_path_instance = MagicMock()
        mock_msit_path.return_value = mock_path_instance

        @_save_dir(dir_size=2048)
        def failing_func(data, path, *args, **kwargs):
            raise MsitException("Test error")

        with self.assertRaises(MsitException) as cm:
            failing_func("test_data", "/failing/path")
        self.assertIn(MsgConst.IO_FAILURE, cm.exception.error_msg)
        mock_change_perm.assert_not_called()


class TestLoadNpyFromBuffer(unittest.TestCase):
    def test_load_valid_buffer(self):
        expected_array = np.array([1, 2, 3, 4], dtype=np.int32)
        raw_data = expected_array.tobytes()
        result = load_npy_from_buffer(raw_data, dtype=np.int32, shape=(4,))
        np.testing.assert_array_equal(result, expected_array)

    def test_invalid_dtype(self):
        test_data = np.array([1, 2, 3, 4], dtype=np.int32)
        raw_data = test_data.tobytes()
        with self.assertRaises(MsitException) as cm:
            load_npy_from_buffer(raw_data, dtype=np.float64, shape=(4,))
        self.assertIn(MsgConst.IO_FAILURE, cm.exception.error_msg)
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_mismatched_shape(self):
        test_data = np.array([1, 2, 3, 4], dtype=np.int32)
        raw_data = test_data.tobytes()
        with self.assertRaises(MsitException) as cm:
            load_npy_from_buffer(raw_data, dtype=np.int32, shape=(2, 3))
        self.assertIn(MsgConst.IO_FAILURE, cm.exception.error_msg)
        self.assertIn("reshape", str(cm.exception.__cause__).lower())

    def test_invalid_raw_data_type(self):
        with self.assertRaises(MsitException) as cm:
            load_npy_from_buffer("invalid_data", dtype=np.int32, shape=(1,))

        self.assertIn(MsgConst.IO_FAILURE, cm.exception.error_msg)
        self.assertIsInstance(cm.exception.__cause__, TypeError)


class TestPermissionManagement(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)
        self.temp_dir.cleanup()

    @patch("os.chmod")
    def test_change_permission_file(self, mock_chmod):
        change_permission(self.temp_file.name, 0o644)
        mock_chmod.assert_called_once_with(self.temp_file.name, 0o644)

    @patch("os.chmod")
    def test_change_permission_dir(self, mock_chmod):
        change_permission(self.temp_dir.name, 0o755)
        mock_chmod.assert_called_once_with(self.temp_dir.name, 0o755)


class TestModelLoading(unittest.TestCase):
    def setUp(self):
        self.mock_onnx = MagicMock()
        self.mock_ort = MagicMock()
        self.mock_caffe = MagicMock()
        self.mock_tf = MagicMock()
        self.mock_rewriter_config = MagicMock()
        self.mock_convert_vars = MagicMock()
        self.mock_graph = MagicMock()
        self.mock_session = MagicMock()
        self.mock_saved_model = MagicMock()
        self.mock_graph_def = MagicMock()

        dependent._dependencies["onnx"] = self.mock_onnx
        dependent._dependencies["onnxruntime"] = self.mock_ort
        dependent._dependencies["caffe"] = self.mock_caffe
        dependent._dependencies["tensorflow"] = self.mock_tf
        dependent._dependencies["tensorflow/RewriterConfig"] = self.mock_rewriter_config
        dependent._dependencies["tensorflow/convert_variables_to_constants"] = self.mock_convert_vars

        self.mock_tf.compat.v1.Graph.return_value = self.mock_graph
        self.mock_tf.compat.v1.Session.return_value = self.mock_session
        self.mock_tf.compat.v1.saved_model.loader.load.return_value = self.mock_saved_model
        self.mock_tf.compat.v1.gfile.GFile.return_value.read.return_value = b"proto_data"
        self.mock_tf.compat.v1.GraphDef.return_value = self.mock_graph_def

    def tearDown(self):
        dependent._dependencies.clear()

    @patch("msit.utils.path.MsitPath.check")
    def test_load_onnx_model(self, mock_check):
        mock_check.return_value = "dummy.onnx"
        mock_model = MagicMock()
        self.mock_onnx.load_model.return_value = mock_model
        result = load_onnx_model("dummy.onnx")
        mock_check.assert_called_once()
        self.mock_onnx.load_model.assert_called_once_with("dummy.onnx")
        self.assertEqual(result, mock_model)

    @patch("msit.utils.path.MsitPath.check")
    def test_load_onnx_session(self, mock_check):
        mock_check.return_value = "dummy.onnx"
        mock_session = MagicMock()
        self.mock_ort.InferenceSession.return_value = mock_session
        result = load_onnx_session("dummy.onnx", provider="CPUExecutionProvider")
        mock_check.assert_called_once()
        self.mock_ort.InferenceSession.assert_called_once_with(
            "dummy.onnx", sess_options=self.mock_ort.SessionOptions(), providers=["CPUExecutionProvider"]
        )
        self.assertEqual(result, mock_session)

    @patch("msit.utils.path.MsitPath.check")
    def test_load_caffe_model(self, mock_check):
        mock_check.return_value = "model.prototxt"
        mock_net = MagicMock()
        self.mock_caffe.Net.return_value = mock_net
        result = load_caffe_model("model.prototxt", "weights.caffemodel")
        mock_check.assert_called_once()
        self.mock_caffe.Net.assert_called_once_with("model.prototxt", "weights.caffemodel", self.mock_caffe.TEST)
        self.assertEqual(result, mock_net)

    @patch("msit.utils.path.MsitPath.check")
    @patch("msit.utils.dependencies.dependent.get")
    def test_save_small_model(self, mock_dependent_get, mock_check):
        mock_check.return_value = "model.onnx"
        mock_onnx = MagicMock()
        mock_onnx.save_model = MagicMock()
        mock_dependent_get.return_value = mock_onnx

        mock_model = MagicMock()
        mock_model.ByteSize.return_value = PathConst.SIZE_2G - 1
        save_onnx_model(mock_model, "model.onnx")
        mock_check.assert_called_once()
        mock_dependent_get.assert_called_once_with("onnx")
        mock_onnx.save_model.assert_called_once_with(mock_model, "model.onnx", save_as_external_data=False)

    @patch("msit.utils.path.MsitPath.check")
    @patch("msit.utils.dependencies.dependent.get")
    def test_save_large_model(self, mock_dependent_get, mock_check):
        mock_check.return_value = "large_model.onnx"
        mock_onnx = MagicMock()
        mock_dependent_get.return_value = mock_onnx
        mock_model = MagicMock()
        mock_model.ByteSize.return_value = PathConst.SIZE_2G + 1
        save_onnx_model(mock_model, "large_model.onnx")
        mock_check.assert_called_once()
        mock_onnx.save_model.assert_called_once_with(mock_model, "large_model.onnx", save_as_external_data=True)

    @patch("msit.utils.path.MsitPath.check")
    @patch("msit.utils.dependencies.dependent.get")
    def test_onnx_dependency_missing(self, mock_dependent_get, mock_check):
        mock_check.return_value = "model.onnx"
        mock_dependent_get.return_value = None
        mock_model = MagicMock()
        with self.assertRaises(MsitException) as ctx:
            save_onnx_model(mock_model, "model.onnx")
        self.assertIn("using <save_onnx_model>. Please check permissions or disk space.", str(ctx.exception))

    @patch("numpy.load")
    @patch("msit.utils.path.MsitPath.check")
    def test_load_npy(self, mock_check, mock_np_load):
        mock_check.return_value = "data.npy"
        mock_data = MagicMock()
        mock_np_load.return_value = mock_data
        result = load_npy("data.npy")
        mock_check.assert_called_once()
        mock_np_load.assert_called_once_with("data.npy", allow_pickle=False)
        np.testing.assert_array_equal(result, mock_data)

    @patch("numpy.save")
    @patch("msit.utils.path.MsitPath.check")
    def test_save_npy(self, mock_check, mock_np_save):
        mock_check.return_value = "save.npy"
        data = np.array([1, 2, 3])
        save_npy(data, "save.npy")
        mock_check.assert_called_once()
        mock_np_save.assert_called_once_with("save.npy", data)

    @patch("msit.utils.path.MsitPath.check")
    def test_load_saved_model(self, mock_check):
        mock_check.return_value = "saved_model"
        result_model, result_sess = load_saved_model("saved_model", ["serve"])
        tf_module, rewriter_config, convert_vars = dependent.get_tensorflow()
        self.assertIsNotNone(tf_module)
        self.assertIsNotNone(rewriter_config)
        self.assertIsNotNone(convert_vars)
        self.mock_tf.compat.v1.reset_default_graph.assert_called_once()
        self.mock_tf.compat.v1.Graph.assert_called_once()
        self.mock_tf.compat.v1.Session.assert_called_once_with(graph=self.mock_graph)
        self.mock_tf.compat.v1.saved_model.loader.load.assert_called_once_with(
            self.mock_session, set(["serve"]), "saved_model"
        )
        self.assertEqual(result_model, self.mock_saved_model)
        self.assertEqual(result_sess, self.mock_session)

    @patch("msit.utils.path.MsitPath.check", side_effect=MsitException("File not found"))
    def test_load_onnx_model_failure(self, mock_check):
        with self.assertRaises(MsitException):
            load_onnx_model("invalid.onnx")

    def test_load_caffe_model_no_dependency(self):
        dependent._dependencies["caffe"] = None
        with self.assertRaises(MsitException):
            result = load_caffe_model("model.prototxt", "weights.caffemodel")
            self.assertIsNone(result)

    @patch("msit.utils.path.MsitPath.check")
    def test_load_pb_frozen_graph_model_success(self, mock_check):
        mock_check.return_value = "model.pb"
        result = load_pb_frozen_graph_model("model.pb")
        self.mock_tf.compat.v1.gfile.GFile.assert_called_once_with("model.pb", "rb")
        self.mock_graph_def.ParseFromString.assert_called_once_with(b"proto_data")
        self.assertEqual(result, self.mock_graph_def)

    def test_load_pb_frozen_graph_model_no_tf(self):
        dependent._dependencies["tensorflow"] = None
        with self.assertRaises(MsitException):
            result = load_pb_frozen_graph_model("model.pb")
            self.assertIsNone(result)

    @patch("msit.utils.path.MsitPath.check")
    def test_save_pb_frozen_graph_model(self, mock_check):
        mock_gfile = MagicMock()
        mock_gfile_instance = MagicMock()
        mock_gfile.__enter__.return_value = mock_gfile_instance
        self.mock_tf.io.gfile.GFile.return_value = mock_gfile
        mock_check.return_value = "save.pb"
        mock_frozen_graph = b"dummy_frozen_graph_data"
        save_pb_frozen_graph_model(mock_frozen_graph, "save.pb")
        self.mock_tf.io.gfile.GFile.assert_called_once_with("save.pb", "wb")
        mock_gfile_instance.write.assert_called_once_with(mock_frozen_graph)


class TestBinFileOperations(unittest.TestCase):
    @patch("msit.utils.io.np.fromfile")
    @patch("msit.utils.io.get_file_size")
    @patch("msit.utils.path.MsitPath.check")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_bin_float32_with_valid_size(self, mock_open_file, mock_check, mock_get_size, mock_fromfile):

        mock_check.return_value = "data.bin"
        mock_get_size.return_value = 8

        mock_fromfile.side_effect = [
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        ]

        result = load_bin_to_ndarray("data.bin", dtype=np.float32, shape=(2, 2))

        mock_get_size.assert_called_once_with("data.bin")
        mock_fromfile.assert_any_call("data.bin", dtype=np.float16)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    @patch("msit.utils.io.np.fromfile")
    @patch("msit.utils.io.get_file_size")
    @patch("msit.utils.io.MsitPath.check")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_bin_float32_with_invalid_size(self, mock_open_file, mock_check, mock_get_size, mock_fromfile):
        mock_check.return_value = "data.bin"
        mock_get_size.return_value = 10
        mock_fromfile.return_value = np.array([1.0, 2.0], dtype=np.float32)
        result = load_bin_to_ndarray("data.bin", dtype=np.float32, shape=(2, 2))
        mock_fromfile.assert_called_once_with("data.bin", dtype=np.float32)
        self.assertEqual(result.dtype, np.float32)


class TestSavedModelToPb(unittest.TestCase):
    def setUp(self):
        self.mock_tf = MagicMock()
        self.mock_rewriter_config = MagicMock()
        self.mock_sm2pb = MagicMock()
        self.mock_sess = MagicMock()
        self.mock_meta_graph = MagicMock()

        self.mock_tf.compat.v1.saved_model.loader.load.return_value = self.mock_meta_graph
        dependent._dependencies["tensorflow"] = self.mock_tf
        dependent._dependencies["tensorflow/RewriterConfig"] = self.mock_rewriter_config
        dependent._dependencies["tensorflow/convert_variables_to_constants"] = self.mock_sm2pb

    @patch("msit.utils.io.load_saved_model")
    @patch("msit.utils.io.save_pb_frozen_graph_model")
    def test_savedmodel2pb_success(self, mock_save_pb, mock_load_model):
        mock_load_model.return_value = (self.mock_meta_graph, self.mock_sess)
        mock_signature = MagicMock()
        self.mock_meta_graph.signature_def.get.return_value = mock_signature
        mock_signature.inputs = {"input": MagicMock(name="input:0")}
        mock_signature.outputs = {"output": MagicMock(name="output:0")}

        result = savedmodel2pb("model_dir", ["serve"], "serving_default", "output_dir")
        self.mock_sm2pb.assert_called_once()
        mock_save_pb.assert_called_once()
        self.assertIn("model_dir.pb", result)

    def test_savedmodel2pb_signature_not_found(self):
        with self.assertRaises(MsitException):
            savedmodel2pb("model_dir", ["serve"], "invalid_signature", "output_dir")


class TestYamlJsonOperations(unittest.TestCase):
    @patch("yaml.safe_load")
    @patch("msit.utils.io.SafelyOpen")
    @patch("msit.utils.path.MsitPath.check")
    def test_load_yaml(self, mock_check, mock_safely_open, mock_yaml_load):
        mock_check.return_value = "dummy.yaml"
        mock_file = MagicMock()
        mock_safely_open.return_value.__enter__.return_value = mock_file
        mock_yaml_load.return_value = {"key": "value"}
        result = load_yaml("dummy.yaml")
        mock_safely_open.assert_called_once_with("dummy.yaml", "r", PathConst.SIZE_500M, PathConst.SUFFIX_YAML, "utf-8")
        mock_yaml_load.assert_called_once_with(mock_file)
        self.assertEqual(result, {"key": "value"})

    @patch("yaml.dump")
    @patch("msit.utils.io.SafelyOpen")
    @patch("msit.utils.path.MsitPath.check")
    def test_save_yaml(self, mock_check, mock_safely_open, mock_yaml_dump):
        mock_check.return_value = "save.yaml"
        mock_file = MagicMock()
        mock_safely_open.return_value.__enter__.return_value = mock_file
        save_yaml({"key": "value"}, "save.yaml")
        mock_safely_open.assert_called_once_with("save.yaml", "w", None, PathConst.SUFFIX_YAML, path_exist=False)
        mock_yaml_dump.assert_called_once_with({"key": "value"}, mock_file)

    @patch("json.load")
    @patch("msit.utils.io.SafelyOpen")
    @patch("msit.utils.path.MsitPath.check")
    def test_load_json(self, mock_check, mock_safely_open, mock_json_load):
        mock_check.return_value = "data.json"
        mock_file = MagicMock()
        mock_safely_open.return_value.__enter__.return_value = mock_file
        mock_json_load.return_value = {"name": "test"}
        result = load_json("data.json")
        mock_safely_open.assert_called_once_with("data.json", "r", PathConst.SIZE_2G, PathConst.SUFFIX_JSON, "utf-8")
        self.assertEqual(result, {"name": "test"})

    @patch("json.dump")
    @patch("msit.utils.io.SafelyOpen")
    @patch("msit.utils.path.MsitPath.check")
    def test_save_json(self, mock_check, mock_safely_open, mock_json_dump):
        mock_check.return_value = "save.json"
        mock_file = MagicMock()
        mock_safely_open.return_value.__enter__.return_value = mock_file
        save_json({"id": 1}, "save.json")
        mock_safely_open.assert_called_once_with("save.json", "w", None, PathConst.SUFFIX_JSON, path_exist=False)
        mock_json_dump.assert_called_once_with({"id": 1}, mock_file, indent=None, default=str)


class TestCsvOperations(unittest.TestCase):
    @patch("csv.reader")
    @patch("msit.utils.io.SafelyOpen")
    @patch("msit.utils.path.MsitPath.check")
    def test_load_csv_by_builtin(self, mock_check, mock_safely_open, mock_csv_reader):
        mock_check.return_value = "data.csv"
        mock_file = MagicMock()
        mock_safely_open.return_value.__enter__.return_value = mock_file
        mock_csv_reader.return_value = [["a", "1"], ["b", "2"]]
        result = load_csv_by_builtin("data.csv")
        mock_safely_open.assert_called_once_with(
            "data.csv", "r", PathConst.SIZE_500M, PathConst.SUFFIX_CSV, "utf-8-sig"
        )
        self.assertEqual(result, [["a", "1"], ["b", "2"]])

    @patch("pandas.read_csv")
    @patch("msit.utils.path.MsitPath.check")
    def test_load_csv_by_pandas(self, mock_check, mock_pd_read):
        mock_check.return_value = "data.csv"
        mock_df = pd.DataFrame({"col1": ["a", "b"], "col2": [1, 2]})
        mock_pd_read.return_value = mock_df
        result = load_csv_by_pandas("data.csv")
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("pandas.DataFrame.to_csv")
    @patch("msit.utils.path.MsitPath.check")
    def test_save_csv_by_pandas(self, mock_check, mock_to_csv):
        mock_check.return_value = "save.csv"
        df = pd.DataFrame({"A": [1, 2]})
        save_csv_by_pandas(df, "save.csv")
        mock_to_csv.assert_called_once_with("save.csv", sep=",", index=False)


class TestTorchOperations(unittest.TestCase):
    def setUp(self):
        self.mock_torch = MagicMock()
        dependent._dependencies["torch"] = self.mock_torch

    @patch("msit.utils.io.is_input_yes")
    @patch("msit.utils.io.MsitPath.check")
    def test_load_torch_obj_safe(self, mock_check, mock_input):
        mock_check.return_value = "model.pt"
        self.mock_torch.load.side_effect = [pickle.UnpicklingError(), MagicMock()]
        mock_input.return_value = True
        result = load_torch_obj("model.pt")
        self.mock_torch.load.assert_called_with("model.pt", weights_only=False)
