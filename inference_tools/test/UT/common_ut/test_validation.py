import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from msit.common.validation import (
    CheckConfigPath,
    CheckExec,
    CheckFramework,
    MsitPath,
    check_int_border,
    parse_hyphen,
    valid_config_path,
    valid_exec,
    valid_framework,
    valid_level,
    valid_log_level,
    valid_seed,
    valid_step_or_rank,
    valid_task,
)
from msit.utils.exceptions import MsitException


class TestValidationFunctions(unittest.TestCase):
    def setUp(self):
        self.mock_cfgconst = MagicMock()
        self.mock_cfgconst.ALL_TASK = ["train", "eval", "predict"]
        self.mock_cfgconst.ALL_FRAMEWORK = ["tf", "pytorch"]
        self.mock_cfgconst.ALL_LEVEL = ["info", "debug", "warning"]
        self.patcher = patch.dict(
            "sys.modules",
            {
                "msit.utils.constants.CfgConst": self.mock_cfgconst,
                "msit.utils.constants.PathConst": MagicMock(
                    SUFFIX_SH=".sh",
                    SUFFIX_PY=".py",
                    SUFFIX_OFFLINE_MODEL=(".onnx", ".pb"),
                    SUFFIX_ONLINE_SCRIPT=(".sh", ".py"),
                    SUFFIX_JSON=".json",
                    DIR="dir",
                    FILE="file",
                ),
            },
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_valid_task_valid(self):
        self.assertEqual(valid_task("tensor"), "tensor")

    def test_valid_task_invalid(self):
        with self.assertRaises(MsitException) as cm:
            valid_task("invalid_task")
        self.assertIn("must be one of ", str(cm.exception))

    def test_valid_task_type_invalid(self):
        with self.assertRaises(MsitException) as cm:
            valid_task(123)
        self.assertIn("[ERROR] invalid data type.", str(cm.exception))

    def test_valid_exec_none(self):
        self.assertEqual(valid_exec([]), [])

    def test_valid_exec_type_invalid(self):
        with self.assertRaises(MsitException) as cm:
            valid_exec(123)
        self.assertIn("[ERROR] invalid data type.", str(cm.exception))

    @patch("msit.common.validation.is_dir")
    @patch("msit.common.validation.MsitPath")
    def test_valid_exec_directory(self, mock_msitpath, mock_is_dir):
        mock_is_dir.return_value = True
        values = "/valid/directory"
        result = valid_exec(values)
        self.assertEqual(result, [values])
        mock_msitpath.assert_called_once()

    def test_valid_exec_bash_valid(self):
        values = "bash script.sh"
        self.assertEqual(valid_exec(values), ["bash", "script.sh"])

    def test_valid_exec_bash_invalid(self):
        with self.assertRaises(MsitException) as cm:
            valid_exec("bash invalid_script.py")
        self.assertIn("[ERROR] Parsing failed.", str(cm.exception))

    def test_valid_exec_python_invalid(self):
        with self.assertRaises(MsitException) as cm:
            valid_exec("python invalid_script.sh")
        self.assertIn("[ERROR] Parsing failed.", str(cm.exception))

    @patch("msit.common.validation.is_file")
    @patch("msit.common.validation.MsitPath")
    def test_valid_exec_model_file(self, mock_msitpath, mock_is_file):
        mock_is_file.return_value = True
        values = "model.onnx"
        self.assertEqual(valid_exec(values), [values])
        mock_msitpath.assert_called_once()

    @patch("msit.common.validation.is_file")
    @patch("msit.common.validation.MsitPath")
    def test_invalid_exec_model_file(self, mock_msitpath, mock_is_file):
        mock_is_file.return_value = True
        values = "model.tfv"
        with self.assertRaises(MsitException) as cm:
            valid_exec(values)
        self.assertIn("('.pb', '.onnx', '.om', '.prototxt', '.py', '.sh').", str(cm.exception))

    @patch("msit.common.validation.is_dir")
    @patch("msit.common.validation.is_file")
    @patch("msit.common.validation.MsitPath")
    def test_check_exec_action(self, mock_msitpath, mock_is_file, mock_is_dir):
        mock_is_dir.return_value = False
        mock_is_file.return_value = True
        action = CheckExec(option_strings=["-e", "--exec"], dest="exec")
        mock_namespace = Namespace()
        test_values = "valid_script.sh"
        with patch.object(MsitPath, "check") as mock_check:
            mock_check.return_value = test_values[0]
            action(None, mock_namespace, test_values)
            self.assertEqual(mock_namespace.exec, [test_values])
            mock_is_dir.assert_called_once_with(test_values)
            mock_is_file.assert_called_once_with(test_values)

    @patch("msit.common.validation.MsitPath")
    def test_valid_config_path_valid(self, mock_msitpath):
        mock_msitpath.return_value.check.return_value = "valid.json"
        result = valid_config_path("config.json")
        self.assertEqual(result, "valid.json")

    def test_valid_config_path(self):
        self.option_strings = ["-c", "--config"]
        self.dest = "config_path"
        self.action = CheckConfigPath(option_strings=self.option_strings, dest=self.dest)
        test_value = "/valid/path/config.json"
        expected_result = "/verified/path/config.json"
        mock_namespace = Namespace()

        with patch("msit.common.validation.valid_config_path") as mock_validator:
            mock_validator.return_value = expected_result
            self.action(parser=MagicMock(), namespace=mock_namespace, values=test_value)
            mock_validator.assert_called_once_with(test_value)
            self.assertEqual(getattr(mock_namespace, self.dest), expected_result)

    def test_valid_framework_valid(self):
        self.assertEqual(valid_framework("mindie_llm"), "mindie_llm")

    def test_valid_framework_invalid(self):
        self.assertEqual(valid_framework(""), "")

    def test_valid_framework_type_invalid(self):
        with self.assertRaises(MsitException) as cm:
            valid_framework(123)
        self.assertIn("[ERROR] invalid data type.", str(cm.exception))

    def test_valid_framework_more_element_invalid(self):
        with self.assertRaises(MsitException) as cm:
            valid_framework("invalid_fw")
        self.assertIn('[ERROR] invalid argument. "framework" must be one of', str(cm.exception))

    def test_check_framework(self):
        self.option_strings = ["-f", "--framework"]
        self.dest = "framework"
        self.action = CheckFramework(option_strings=self.option_strings, dest=self.dest)
        test_value = "mindie_llm"
        mock_namespace = Namespace()
        with patch("msit.common.validation.valid_framework") as mock_validator:
            mock_validator.return_value = test_value
            self.action(parser=MagicMock(), namespace=mock_namespace, values=test_value)
            mock_validator.assert_called_once_with(test_value)
            self.assertEqual(getattr(mock_namespace, self.dest), test_value)

    def test_check_int_border_valid(self):
        check_int_border(0, 500000, 1000000)

    def test_valid_check_int_type_invalid(self):
        with self.assertRaises(MsitException) as cm:
            check_int_border([0.35])
        self.assertIn("[ERROR] invalid data type.", str(cm.exception))

    def test_check_int_border_invalid(self):
        with self.assertRaises(MsitException) as cm:
            check_int_border(-1)
        self.assertIn("The integer range is limited to [0, 1000000.0], currently: -1.", str(cm.exception))
        with self.assertRaises(MsitException):
            check_int_border(1000001)

    def test_parse_hyphen_valid(self):
        self.assertEqual(parse_hyphen("100-200"), list(range(100, 201)))
        self.assertEqual(parse_hyphen("100-200-2"), list(range(100, 201, 2)))

    def test_parse_hyphen_invalid(self):
        with self.assertRaises(MsitException):
            parse_hyphen("100-200-300-400")
        with self.assertRaises(MsitException):
            parse_hyphen("200-100")

    def test_valid_step_or_rank(self):
        self.assertEqual(valid_step_or_rank([10, "20-22", "30-35-2"]), [10, 20, 21, 22, 30, 32, 34])

    def test_valid_step_or_rank_none(self):
        self.assertEqual(valid_step_or_rank([]), [])

    def test_valid_step_or_rank_type_invalid(self):
        with self.assertRaises(MsitException) as cm:
            valid_step_or_rank(123)

    def test_valid_step_or_rank_invalid(self):
        with self.assertRaises(MsitException) as cm:
            valid_step_or_rank([0.35])

    def test_valid_level_valid_none(self):
        self.assertEqual(valid_level(""), "")

    def test_valid_level_invalid_type(self):
        with self.assertRaises(MsitException) as cm:
            valid_level(123)
        self.assertIn("[ERROR] invalid data type.", str(cm.exception))

    def test_valid_level_valid(self):
        self.assertEqual(valid_level(["kernel", "layer"]), ["kernel", "layer"])

    def test_valid_log_level_valid_none(self):
        self.assertEqual(valid_log_level(""), "")

    def test_valid_log_level_invalid_type(self):
        with self.assertRaises(MsitException) as cm:
            valid_log_level(123)
        self.assertIn("[ERROR] invalid data type.", str(cm.exception))

    def test_valid_level_invalid(self):
        with self.assertRaises(MsitException):
            valid_level(["invalid_level"])

    def test_valid_log_level_valid(self):
        self.assertEqual(valid_log_level("info"), "info")

    def test_valid_log_level_invalid(self):
        with self.assertRaises(MsitException):
            valid_log_level("invalid")

    def test_valid_seed_valid_none(self):
        self.assertEqual(valid_seed(""), "")

    def test_valid_seed_valid(self):
        self.assertEqual(valid_seed(42), 42)

    def test_valid_seed_invalid(self):
        with self.assertRaises(MsitException):
            valid_seed("not_an_int")
        with self.assertRaises(MsitException):
            valid_seed(-1)
