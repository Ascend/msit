import unittest
from unittest.mock import patch

from msit.base import BaseComponent
from msit.module.probe.components.dumper_offline_model import OfflineModelActuatorComp


class TestOfflineModelActuatorComp(unittest.TestCase):
    def test_inheritance(self):
        self.assertTrue(issubclass(OfflineModelActuatorComp, BaseComponent))

    @patch.object(BaseComponent, "__init__", return_value=None)
    def test_init_default_priority(self, mock_base_init):
        OfflineModelActuatorComp()
        mock_base_init.assert_called_once_with(100)

    @patch.object(BaseComponent, "__init__", return_value=None)
    def test_init_custom_priority(self, mock_base_init):
        custom_priority = 200
        OfflineModelActuatorComp(priority=custom_priority)
        mock_base_init.assert_called_once_with(custom_priority)

    @patch.object(BaseComponent, "__init__", return_value=None)
    def test_instance_type(self, mock_base_init):
        instance = OfflineModelActuatorComp()
        self.assertIsInstance(instance, BaseComponent)
