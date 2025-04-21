import unittest
from argparse import RawTextHelpFormatter
from unittest.mock import MagicMock, patch

from msit.base import Command, MsitCommand
from msit.utils.constants import CmdConst, MsgConst
from msit.utils.exceptions import MsitException


class TestCommandRegistration(unittest.TestCase):
    def setUp(self):
        Command._cmd_map.clear()

    def test_register_command(self):
        parent_cmd = None
        cmd_name = "test"

        @Command.register(parent_cmd, cmd_name)
        class TestCommand(MsitCommand):
            pass

        self.assertIn(parent_cmd, Command._cmd_map)
        self.assertIn(cmd_name, Command._cmd_map[parent_cmd])
        self.assertIs(Command._cmd_map[parent_cmd][cmd_name], TestCommand)

    def test_get_command(self):
        parent1, parent2 = "parent1", "parent2"
        cmd1, cmd2 = "cmd1", "cmd2"

        @Command.register(parent1, cmd1)
        class Cmd1(MsitCommand):
            pass

        @Command.register(parent2, cmd2)
        class Cmd2(MsitCommand):
            pass

        self.assertEqual(Command.get(parent1), {cmd1: Cmd1})
        self.assertEqual(Command.get(parent2), {cmd2: Cmd2})
        self.assertEqual(Command.get("invalid_parent"), {})


class TestMsitCommand(unittest.TestCase):
    class ConcreteCommand(MsitCommand):
        def add_arguments(self, parse):
            pass

    def setUp(self):
        self.cmd = self.ConcreteCommand()
        self.cmd.subcommand_level = 0

    @patch("msit.base.cmd.argv", ["script", "arg1", "arg2"])
    def test_input_module_valid(self):
        self.cmd.subcommand_level = 1
        self.assertEqual(self.cmd.input_module, "arg1")

    @patch("msit.base.cmd.argv", ["script"])
    def test_input_module_insufficient_args(self):
        self.cmd.subcommand_level = 1
        self.assertIsNone(self.cmd.input_module)

    def test_input_module_invalid_level(self):
        self.cmd.subcommand_level = "invalid"
        with self.assertRaises(MsitException) as cm:
            _ = self.cmd.input_module
        self.assertEqual(str(cm.exception), f"{MsgConst.INVALID_ARGU} Subcommand level must be a positive integer.")

    @patch("msit.base.Command.get")
    def test_build_parser_with_subcommands(self, mock_get):
        class MockSubCommand:
            @classmethod
            def add_arguments(cls, parser):
                pass

        mock_get.side_effect = [{"subcmd": MockSubCommand}, {}]
        parent_parser = MagicMock()
        fake_subparser = MagicMock()
        subparsers = MagicMock()
        parent_parser.add_subparsers.return_value = subparsers
        subparsers.add_parser.return_value = fake_subparser
        self.cmd.subcommand_level = 0
        self.cmd.build_parser(parent_parser, MagicMock())
        parent_parser.add_subparsers.assert_called_once_with(dest="L1command")
        subparsers.add_parser.assert_called_once_with(
            name="subcmd", help=CmdConst.HELP_TOOL_MAP.get("subcmd"), formatter_class=RawTextHelpFormatter
        )
