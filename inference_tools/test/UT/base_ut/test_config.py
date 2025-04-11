import unittest
from argparse import Namespace
from unittest.mock import patch

from msit.base import BaseConfig, Dict2Class
from msit.utils.constants import CfgConst, MsgConst
from msit.utils.exceptions import MsitException


class TestBaseConfig(unittest.TestCase):
    class ConcreteConfig(BaseConfig):
        def check_config(self):
            pass

    @patch("msit.base.config.load_json")
    def setUp(self, mock_load_json):
        self.mock_load_json = mock_load_json
        self.mock_load_json.return_value = {
            CfgConst.TASK: CfgConst.TASK_STAT,
            CfgConst.EXEC: [],
            CfgConst.FRAMEWORK: "mindie_llm",
            CfgConst.STEP: [0],
            CfgConst.RANK: [1],
            CfgConst.LEVEL: [CfgConst.LEVEL_API],
            CfgConst.LOG_LEVEL: "info",
            CfgConst.SEED: 42,
        }
        self.config_path = "dummy_path.json"
        self.config = self.ConcreteConfig(self.config_path)

    def test_initialization(self):
        self.mock_load_json.assert_called_once_with(self.config_path)
        self.assertEqual(self.config.config, self.mock_load_json.return_value)
        self.assertFalse(self.config.is_from_cmd)

    @patch("msit.base.config.valid_task")
    @patch("msit.base.config.valid_exec")
    @patch("msit.base.config.valid_framework")
    @patch("msit.base.config.valid_step_or_rank")
    @patch("msit.base.config.valid_log_level")
    @patch("msit.base.config.valid_seed")
    def test_common_check_from_file(self, mock_seed, mock_log, mock_step, mock_framework, mock_exec, mock_task):
        args = Namespace(exec=[])
        step = [1, 2]
        self.config.common_check(step=step, args=args)
        mock_task.assert_called_once_with(CfgConst.TASK_STAT)
        mock_exec.assert_called_once_with([])
        mock_framework.assert_called_once_with("mindie_llm")
        mock_step.assert_any_call([1, 2])
        mock_step.assert_any_call([1])
        mock_log.assert_called_once_with("info")
        mock_seed.assert_called_once_with(42)


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
