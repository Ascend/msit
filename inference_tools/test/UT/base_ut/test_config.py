import unittest
from unittest.mock import MagicMock, patch

from msit.base import BaseConfig, Dict2Class
from msit.utils.constants import CfgConst, MsgConst
from msit.utils.exceptions import MsitException


class ConcreteConfig(BaseConfig):
    def check_config(self):
        pass


class TestBaseConfig(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            CfgConst.TASK: "test_task",
            "test_task": {"key": "value"},
            CfgConst.FRAMEWORK: "test_framework",
            CfgConst.STEP: [],
            CfgConst.RANK: [],
            CfgConst.LEVEL: [CfgConst.LEVEL_API],
            CfgConst.LOG_LEVEL: "info",
            CfgConst.SEED: None,
        }
        self.config_path = "dummy_path.json"

    @patch("msit.base.config.load_json")
    def test_initialization(self, mock_load_json):
        mock_load_json.return_value = self.mock_config
        config = ConcreteConfig(self.config_path, task="test_task", step=[], level=[])
        self.assertEqual(config.config_path, self.config_path)
        self.assertEqual(config.config, self.mock_config)
        self.assertEqual(config.task, "test_task")
        self.assertEqual(config.step, [])
        self.assertEqual(config.level, [])
        mock_load_json.assert_called_once_with(self.config_path)

    @patch("msit.base.config.load_json")
    def test_common_check_calls(self, mock_load_json):
        mock_load_json.return_value = self.mock_config
        config = ConcreteConfig(self.config_path)

        with patch.multiple(
            "msit.base.config",
            valid_task=MagicMock(return_value="test_task"),
            valid_framework=MagicMock(return_value="valid_framework"),
            valid_step_or_rank=MagicMock(side_effect=lambda x: x),
            valid_level=MagicMock(return_value=["valid_level"]),
            valid_log_level=MagicMock(return_value="valid_log_level"),
            valid_seed=MagicMock(return_value=42),
        ) as mocks:
            config.common_check()

            self.assertEqual(config.config[CfgConst.TASK], "test_task")
            self.assertEqual(config.config[CfgConst.FRAMEWORK], "valid_framework")
            self.assertEqual(config.config[CfgConst.STEP], [])
            self.assertEqual(config.config[CfgConst.RANK], [])
            self.assertEqual(config.config[CfgConst.LEVEL], ["valid_level"])
            self.assertEqual(config.config[CfgConst.LOG_LEVEL], "valid_log_level")
            self.assertEqual(config.config[CfgConst.SEED], 42)

    @patch("msit.base.config.load_json")
    def test_get_task_dict_success(self, mock_load_json):
        mock_load_json.return_value = {CfgConst.TASK: "existing_task", "existing_task": {"key": "value"}}
        config = ConcreteConfig(self.config_path)
        task_dict = config.get_task_dict()
        self.assertEqual(task_dict, {"key": "value"})

    @patch("msit.base.config.load_json")
    def test_get_task_dict_raises_exception(self, mock_load_json):
        mock_load_json.return_value = {CfgConst.TASK: "non_existing_task"}
        config = ConcreteConfig(self.config_path)
        with self.assertRaises(MsitException) as context:
            config.get_task_dict()
        self.assertIn(f'Missing dictionary for key "non_existing_task".', context.exception.error_msg)

    @patch("msit.base.config.load_json")
    def test_update_config(self, mock_load_json):
        mock_load_json.return_value = self.mock_config
        config = ConcreteConfig(self.config_path)
        test_dict = {}
        mock_check = MagicMock(return_value="checked_value")
        config._update_config(test_dict, "test_key", mock_check, "test_value")
        mock_check.assert_called_once_with("test_value")
        self.assertEqual(test_dict["test_key"], "checked_value")

    @patch("msit.base.config.load_json")
    def test_check_config_wrapper(self, mock_load_json):
        mock_load_json.return_value = self.mock_config
        config = ConcreteConfig(self.config_path)
        with patch.object(config, "common_check") as mock_common_check, patch.object(
            config, "check_config"
        ) as mock_check_config:
            config.check_config()
            mock_common_check.assert_called_once()
            mock_check_config.assert_called_once()
            self.assertEqual(config.task_config, {"key": "value"})


class TestDict2Class(unittest.TestCase):
    def test_basic_conversion(self):
        data = {"name": "test", "value": 10}
        obj = Dict2Class(data)
        self.assertEqual(obj.name, "test")
        self.assertEqual(obj.value, 10)

    def test_nested_dict_conversion(self):
        data = {"nested": {"key": "value"}}
        obj = Dict2Class(data)
        self.assertIsInstance(obj.nested, Dict2Class)
        self.assertEqual(obj.nested.key, "value")

    def test_service_key_processing(self):
        data = {CfgConst.TASK: "special", "special": {"input": [[224, 224], "path/to/input"], "param": 5}}
        obj = Dict2Class(data)
        self.assertEqual(obj.input_shape, [224, 224])
        self.assertEqual(obj.input_path, "path/to/input")
        self.assertEqual(obj.param, 5)

    def test_max_recursion_depth(self):
        data = {}
        current = data
        for _ in range(MsgConst.MAX_RECURSION_DEPTH + 1):
            current["nested"] = {}
            current = current["nested"]
        with self.assertRaises(MsitException) as context:
            Dict2Class(data)
        self.assertIn(f"Maximum recursion depth of {MsgConst.MAX_RECURSION_DEPTH}", str(context.exception))

    def test_missing_attribute(self):
        obj = Dict2Class({"existing": 1})
        with self.assertRaises(MsitException) as context:
            _ = obj.non_existing
        self.assertIn("has no attribute non_existing", str(context.exception))
