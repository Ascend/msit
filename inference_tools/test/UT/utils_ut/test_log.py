import time
import unittest
from unittest.mock import patch

from msit.utils.log import LOG_LEVEL, MsitLogger, get_current_timestamp, logger, print_log_with_star
from msit.lib.msit_c import log


class TestGetCurrentTimestamp(unittest.TestCase):
    def test_used_for_log_true(self):
        result = get_current_timestamp(used_for_log=True)
        self.assertRegex(result, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_used_for_log_false_no_microsecond(self):
        result = get_current_timestamp(used_for_log=False, microsecond=False)
        self.assertIsInstance(result, int)
        self.assertAlmostEqual(result, int(time.time()), delta=1)

    @patch("msit.utils.log.time")
    def test_used_for_log_false_with_microsecond(self, mock_time):
        mock_time.return_value = 1620000000.123456
        expected = round(1620000000.123456 * 1e6) % 10**10
        result = get_current_timestamp(used_for_log=False, microsecond=True)
        self.assertEqual(result, expected)


class TestPrintLogWithStar(unittest.TestCase):
    @patch.object(logger, "info")
    def test_print_log_with_star_normal(self, mock_info):
        test_message = "Test Message"
        print_log_with_star(test_message)
        self.assertEqual(mock_info.call_count, 3)
        args_list = [call.args[0] for call in mock_info.call_args_list]
        self.assertEqual(args_list[0], "*" * 80)
        self.assertEqual(args_list[2], "*" * 80)
        middle_line = args_list[1]
        self.assertEqual(len(middle_line), 80)
        self.assertTrue(middle_line.startswith("*"))
        self.assertTrue(middle_line.endswith("*"))
        expected_content = f"*{test_message.center(78)}*"
        self.assertEqual(middle_line, expected_content)

    @patch.object(logger, "info")
    def test_print_log_with_star_long_message(self, mock_info):
        test_message = "A" * 79
        print_log_with_star(test_message)
        middle_line = mock_info.call_args_list[1].args[0]
        self.assertEqual(len(middle_line), 81)


class TestMsitLogger(unittest.TestCase):
    def setUp(self):
        MsitLogger._instance = None
        self.logger = MsitLogger()

    def tearDown(self):
        MsitLogger._instance = None

    def test_get_level_id_valid(self):
        for idx, level in enumerate(LOG_LEVEL):
            self.assertEqual(MsitLogger.get_level_id(level), idx, f"Failed for level: {level}")

    def test_get_level_id_case_insensitive(self):
        self.assertEqual(MsitLogger.get_level_id("debug"), LOG_LEVEL.index("DEBUG"))

    def test_get_level_id_invalid(self):
        self.assertEqual(MsitLogger.get_level_id("INVALID_LEVEL"), LOG_LEVEL.index("INFO"))

    def test_set_level_valid(self):
        test_levels = ["ERROR", "WARNING", "DEBUG", "INFO"]
        for level in test_levels:
            with self.subTest(level=level):
                self.logger.set_level(level)
                self.assertEqual(log.get_log_level(), LOG_LEVEL.index(level))

    def test_set_level_invalid(self):
        self.logger.set_level("INVALID_LEVEL")
        self.assertEqual(log.get_log_level(), LOG_LEVEL.index("INFO"))

    @patch.object(log, "print_log")
    def test_error_log_when_level_allows(self, mock_print):
        self.logger.set_level("ERROR")
        test_msg = "Test error message"
        self.logger.error(test_msg)
        mock_print.assert_called_once_with(LOG_LEVEL.index("ERROR"), test_msg)

    @patch.object(log, "print_log")
    def test_error_log_when_level_denies(self, mock_print):
        self.logger.set_level("WARNING")
        self.logger.error("Should print")
        mock_print.assert_called()
        mock_print.reset_mock()
        self.logger.set_level("INVALID_LEVEL")
        self.logger.error("Should ALSO print")
        mock_print.assert_called()

    @patch.object(log, "print_log")
    def test_error_special_char_filter(self, mock_print):
        test_msg = "Bad\nmessage\twith\rspecial"
        expected_msg = "Bad_message_with_special"

        self.logger.error(test_msg)
        mock_print.assert_called_once_with(LOG_LEVEL.index("ERROR"), test_msg)

    @patch.object(log, "print_log")
    def test_debug_log_when_level_allows(self, mock_print):
        self.logger.set_level("DEBUG")
        test_msg = "Debug message"
        self.logger.debug(test_msg)
        mock_print.assert_called_once_with(LOG_LEVEL.index("DEBUG"), test_msg)

    @patch.object(log, "print_log")
    def test_debug_special_char_filter(self, mock_print):
        test_msg = f"Special\tchars"
        expected_msg = "Special_chars"

        self.logger.set_level("DEBUG")
        self.logger.debug(test_msg)
        mock_print.assert_called_once_with(LOG_LEVEL.index("DEBUG"), test_msg)
