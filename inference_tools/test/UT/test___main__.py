import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch


class TestMainFunction(TestCase):
    @patch("msit.__main__.MainCommand")
    def test_main_execution_flow(self, mock_main_command):
        mock_instance = MagicMock()
        mock_main_command.return_value = mock_instance
        mock_args = MagicMock()
        mock_instance.parse.return_value = mock_args
        from msit.__main__ import main

        main()
        mock_main_command.assert_called_once()
        mock_instance.parse.assert_called_once()
        mock_instance.execute.assert_called_once_with(mock_args)

    @patch("msit.__main__.MainCommand")
    def test_direct_execution(self, mock_main_command):
        with patch("sys.argv", ["script_name"]):
            from msit.__main__ import main

            main()
            mock_main_command.return_value.execute.assert_called_once()

    def test_main_called_in_if_main(self):
        mock_instance = MagicMock()
        mock_instance.parse.return_value = "mock_args"
        MockMainCommand = MagicMock(return_value=mock_instance)
        with patch("msit.__main__.MainCommand", MockMainCommand):
            from msit.__main__ import main

            main()
        MockMainCommand.assert_called_once()
        mock_instance.register.assert_called_once()
        mock_instance.parse.assert_called_once()
        mock_instance.execute.assert_called_once_with("mock_args")
