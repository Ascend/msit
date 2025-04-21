import unittest
from unittest.mock import patch

from msit.module.probe.base import BaseDumper


class TestBaseDumper(unittest.TestCase):
    def setUp(self):
        class ConcreteDumper(BaseDumper):
            def register_hook(self):
                pass

        self.dumper = ConcreteDumper()

    def test_init_handler_is_empty_list(self):
        self.assertEqual(self.dumper.handler, [])

    @patch("msit.module.probe.base.dump_dumper.release")
    def test_release_hook_with_multiple_handlers(self, mock_release):
        test_handlers = [0xDEADBEEF, 0xCAFEBABE]
        self.dumper.handler = test_handlers
        self.dumper.release_hook()
        self.assertEqual(mock_release.call_count, len(test_handlers))
        mock_release.assert_any_call(test_handlers[0])
        mock_release.assert_any_call(test_handlers[1])

    @patch("msit.module.probe.base.dump_dumper.release")
    def test_release_hook_with_empty_handler(self, mock_release):
        self.dumper.handler = []
        self.dumper.release_hook()
        mock_release.assert_not_called()

    def test_abstract_method_enforcement(self):
        with self.assertRaises(TypeError):

            class InvalidDumper(BaseDumper):
                pass

            InvalidDumper()
