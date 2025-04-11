import unittest
from unittest.mock import ANY, MagicMock, PropertyMock, call, patch

from msit.common.cli import CfgConst, MainCommand, MsitException


class TestMainCommand(unittest.TestCase):
    def setUp(self):
        self.main_cmd = MainCommand()
        self.mock_second_commands = {"cmd1": MagicMock(), "cmd2": MagicMock()}
        self.main_cmd.second_commands = self.mock_second_commands

    @patch("msit.common.cli.ArgumentParser")
    def test_init(self, mock_argparse):
        main_cmd = MainCommand()
        mock_argparse.assert_called_once_with(prog="msit", description=ANY, formatter_class=ANY)
        self.assertEqual(main_cmd.subcommand_level, 1)
        self.assertIsNotNone(main_cmd.parser)
        self.assertIsNotNone(main_cmd.subparser)

    def test_register(self):
        with patch.object(MainCommand, "input_module", new_callable=PropertyMock) as mock_input_module:
            mock_input_module.return_value = "cmd1"
            mock_subparser = MagicMock()
            self.main_cmd.subparser = mock_subparser
            self.main_cmd.register()
            expected_calls = [call(name="cmd1", help=None, formatter_class=self.main_cmd.formatter_class)]
            mock_subparser.add_parser.assert_has_calls(expected_calls, any_order=True)
            self.mock_second_commands["cmd1"].add_arguments.assert_called_once()
            self.assertEqual(self.main_cmd.subcommand_level, 2)

    def test_parse(self):
        mock_args = MagicMock()
        self.main_cmd.parser.parse_args = MagicMock(return_value=mock_args)
        result = self.main_cmd.parse()
        self.assertEqual(result, mock_args)
        self.main_cmd.parser.parse_args.assert_called_once()

    @patch("msit.common.cli.set_ld_preload")
    @patch("msit.common.cli.cann.get_atb_probe_so_path")
    def test_set_env_success(self, mock_get_so, mock_set_preload):
        mock_get_so.return_value = "/fake/path.so"
        self.main_cmd.set_env(CfgConst.FRAMEWORK_MINDIE_LLM)
        mock_set_preload.assert_called_once_with("/fake/path.so")

    @patch("msit.common.cli.cann.get_atb_probe_so_path")
    def test_set_env_failure(self, mock_get_so):
        mock_get_so.return_value = None
        with self.assertRaises(MsitException) as context:
            self.main_cmd.set_env("invalid_framework")
        self.assertIn(".so library path for invalid_framework not found.", str(context.exception))

    @patch("msit.common.cli.Service")
    @patch("msit.common.cli.run_subprocess")
    @patch("msit.common.cli.set_ld_preload")
    @patch("msit.common.cli.cann.get_atb_probe_so_path")
    @patch("msit.common.cli.argv", ["msit", "valid_service"])
    def test_execute_service_with_framework(self, mock_get_so, mock_set_preload, mock_run_subprocess, mock_service):
        mock_get_so.return_value = "/fake/mindie_llm.so"
        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service.get.return_value = True
        args = MagicMock()
        args.framework = CfgConst.FRAMEWORK_MINDIE_LLM
        args.msitx = False
        args.config = "config.yaml"
        args.exec = "exec_command"
        self.main_cmd.execute(args)
        mock_get_so.assert_called_once()
        mock_set_preload.assert_called_once_with("/fake/mindie_llm.so")
        mock_service.assert_called_once_with("config.yaml", args=args)
        mock_service_instance.run_cli.assert_called_once()
        mock_run_subprocess.assert_not_called()

    @patch("msit.common.cli.Service")
    @patch("msit.common.cli.run_subprocess")
    @patch("msit.common.cli.set_ld_preload")
    @patch("msit.common.cli.cann.get_atb_probe_so_path")
    @patch("msit.common.cli.argv", ["msit", "valid_service"])
    def test_execute_msitx_mode(self, mock_get_so, mock_set_preload, mock_run_subprocess, mock_service):
        self.main_cmd.subcommand_level = 2
        mock_get_so.return_value = None
        mock_service.get.return_value = True
        args = MagicMock()
        args.msitx = True
        args.exec = "exec_command"
        args.framework = None
        self.main_cmd.execute(args)
        mock_service.get.assert_called_once_with("valid_service")
        mock_run_subprocess.assert_called_once_with("exec_command")
        mock_set_preload.assert_not_called()
        mock_get_so.assert_not_called()
        mock_service.return_value.run_cli.assert_not_called()

    @patch("msit.common.cli.Service")
    @patch("msit.common.cli.argv", ["msit", "invalid_service"])
    def test_execute_invalid_service(self, mock_service):
        mock_service.get.return_value = False
        args = MagicMock()
        with self.assertRaises(MsitException) as context:
            self.main_cmd.execute(args)
        self.assertIn(" utility is not registered", str(context.exception))
