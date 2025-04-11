import unittest
from unittest.mock import MagicMock, call, create_autospec, patch

from msit.base import BaseComponent, BaseService, Scheduler, Service
from msit.utils.constants import CfgConst, CmdConst


class TestService(unittest.TestCase):
    def setUp(self):
        Service._services_map.clear()

    @patch("msit.base.service.manager.load_json")
    def test_service_initialization(self, mock_load_json):
        mock_load_json.return_value = {CfgConst.TASK: CfgConst.TASK_STAT}
        mock_service_cls = MagicMock()
        Service._services_map[CmdConst.DUMP] = mock_service_cls
        service = Service("dummy_path", key="value")
        mock_load_json.assert_called_once_with("dummy_path")
        mock_service_cls.assert_called_once_with("dummy_path", key="value")
        self.assertEqual(service.service_instance, mock_service_cls.return_value)

    def test_service_registration(self):
        @Service.register("test_service")
        class TestServiceImpl:
            pass

        self.assertEqual(Service._services_map["test_service"], TestServiceImpl)

    def test_service_getattr_delegation(self):
        mock_instance = MagicMock()
        mock_instance.target_method = MagicMock(return_value="mocked_result")

        with patch("msit.base.service.manager.Service._services_map", {CmdConst.DUMP: MagicMock()}) as mock_map, patch(
            "msit.base.service.manager.load_json"
        ) as mock_load, patch("msit.base.service.manager.valid_task") as mock_valid, patch(
            "os.path.exists", return_value=True
        ):
            mock_load.return_value = {CfgConst.TASK: CfgConst.TASK_STAT}
            mock_valid.return_value = CfgConst.TASK_STAT
            mock_service_cls = mock_map[CmdConst.DUMP]
            mock_service_cls.return_value = mock_instance
            service = Service("any_dummy_path")
            result = service.target_method("test_arg", kwarg=123)
            mock_instance.target_method.assert_called_once_with("test_arg", kwarg=123)
            self.assertEqual(result, "mocked_result")
            mock_service_cls.assert_called_once_with("any_dummy_path")


class TestBaseService(unittest.TestCase):
    @patch.object(Scheduler, "add")
    @patch.object(Scheduler, "remove")
    def test_full_lifecycle(self, mock_remove, mock_add):
        class TestService(BaseService):
            def construct(self):
                self.high_pri = create_autospec(BaseComponent, name="high_pri")
                self.high_pri.priority = 1
                self.low_pri = create_autospec(BaseComponent, name="low_pri")
                self.low_pri.priority = 2
                self.non_comp = "non comp"

        service = TestService()
        service.start()
        mock_add.assert_called_once()

    @patch.object(BaseService, "init_start")
    @patch.object(BaseService, "finalize_start")
    def test_hook_execution_order(self, mock_final, mock_init):
        class HookTestService(BaseService):
            def construct(self):
                pass

        service = HookTestService()
        service.start()
        mock_init.assert_called_once()
        mock_final.assert_called_once()
        self.assertEqual(mock_init.call_args_list[0], call())
        self.assertEqual(mock_final.call_args_list[-1], call())
