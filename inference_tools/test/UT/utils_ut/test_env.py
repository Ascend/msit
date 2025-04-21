import os
import unittest
from unittest import mock

from msit.utils.env import EnvVarManager
from msit.utils.exceptions import MsitException


class TestEnvVarManager(unittest.TestCase):
    def setUp(self):
        self.manager = EnvVarManager()
        self.manager.set_prefix("")
        self.env_patcher = mock.patch.dict(os.environ, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_singleton_instance(self):
        manager1 = EnvVarManager()
        manager2 = EnvVarManager()
        self.assertIs(manager1, manager2)

    def test_set_prefix(self):
        self.manager.set_prefix("TEST_")
        self.assertEqual(self.manager.prefix, "TEST_")

    def test_get_existing_var_no_prefix(self):
        os.environ["KEY"] = "value"
        result = self.manager.get("KEY")
        self.assertEqual(result, "value")

    def test_get_existing_var_with_prefix(self):
        self.manager.set_prefix("TEST_")
        os.environ["TEST_KEY"] = "value"
        result = self.manager.get("KEY")
        self.assertEqual(result, "value")

    def test_get_missing_var_required(self):
        with self.assertRaises(MsitException) as cm:
            self.manager.get("MISSING_KEY", required=True)
        self.assertIn("MISSING_KEY", str(cm.exception))

    def test_get_missing_var_optional_with_default(self):
        result = self.manager.get("MISSING_KEY", default="default_val", required=False)
        self.assertEqual(result, "default_val")

    def test_get_cast_type_success(self):
        os.environ["INT_VAL"] = "123"
        result = self.manager.get("INT_VAL", cast_type=int)
        self.assertEqual(result, 123)

    def test_get_cast_type_failure(self):
        os.environ["INVALID_INT"] = "abc"
        with self.assertRaises(MsitException) as cm:
            self.manager.get("INVALID_INT", cast_type=int)
        self.assertIn("Failed to cast", str(cm.exception))

    def test_set_var_with_prefix(self):
        self.manager.set_prefix("TEST_")
        self.manager.set("NEW_KEY", "value")
        self.assertEqual(os.environ["TEST_NEW_KEY"], "value")

    def test_delete_existing_var(self):
        os.environ["TEST_KEY"] = "value"
        self.manager.set_prefix("TEST_")
        self.manager.delete("KEY")
        self.assertNotIn("TEST_KEY", os.environ)

    def test_delete_non_existing_var(self):
        self.manager.set_prefix("TEST_")
        try:
            self.manager.delete("NON_EXISTENT")
        except Exception:
            self.fail("Deleting non-existent variable raised unexpected exception")

    def test_list_all_with_prefix(self):
        os.environ.update({"TEST_A": "1", "TEST_B": "2", "OTHER": "3"})
        self.manager.set_prefix("TEST_")
        result = self.manager.list_all()
        expected = {"TEST_A": "1", "TEST_B": "2"}
        self.assertDictEqual(result, expected)

    def test_list_all_without_prefix(self):
        os.environ["KEY"] = "value"
        result = self.manager.list_all()
        self.assertIn("KEY", result)

    @mock.patch("msit.utils.log.logger.debug")
    def test_logging_on_get(self, mock_debug):
        os.environ["LOGGED_KEY"] = "log_value"
        self.manager.get("LOGGED_KEY")
        mock_debug.assert_called_with("Accessed environment variable LOGGED_KEY, Value: log_value.")

    @mock.patch("msit.utils.log.logger.debug")
    def test_logging_on_set(self, mock_debug):
        self.manager.set("LOGGED_SET", "value")
        mock_debug.assert_called_with("Set environment variable LOGGED_SET to value.")
