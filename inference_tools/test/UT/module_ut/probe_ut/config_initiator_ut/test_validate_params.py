import unittest
from unittest.mock import MagicMock, patch

from msit.module.probe.config_initiator.validate_params import (
    valid_dump_extra,
    valid_dump_ge_graph,
    valid_dump_graph_level,
    valid_dump_last_logits,
    valid_dump_mode,
    valid_dump_path,
    valid_dump_time,
    valid_dump_weight,
    valid_list,
    valid_op_id,
    valid_fusion_switch_file,
    valid_device,
    valid_weight_path,
    valid_onnx_fusion_switch,
    valid_saved_model_tag,
    valid_saved_model_signature,
    valid_input,
    OfflineModelInput
)
from msit.utils.exceptions import MsitException


class TestValidators(unittest.TestCase):

    @patch("msit.module.probe.config_initiator.validate_params.MsitPath")
    def test_valid_dump_path(self, mock_msit_path):
        mock_instance = MagicMock()
        mock_instance.check.return_value = "valid"
        mock_msit_path.return_value = mock_instance
        result = valid_dump_path("some/path")
        self.assertEqual(result, "valid")
        mock_msit_path.assert_called_once()

    def test_valid_list_with_none(self):
        value = ("", ["level1", "level2"])
        result = valid_list(value)
        self.assertEqual(result, {"level1": ""})

    def test_valid_list_with_dict(self):
        value = ({"level1": [1]}, ["level1", "level2"])
        result = valid_list(value)
        self.assertEqual(result, {"level1": [1]})

    def test_valid_list_invalid_dict_key(self):
        value = ({"bad": [1]}, ["allowed"])
        with self.assertRaises(MsitException):
            valid_list(value)

    def test_valid_list_invalid_list(self):
        value = ({"level1": 12}, ["level1", "level2"])
        with self.assertRaises(MsitException):
            valid_list(value)

    def test_invalid_list(self):
        value = (12, ["level1", "level2"])
        with self.assertRaises(MsitException):
            valid_list(value)

    def test_valid_dump_mode_none(self):
        result = valid_dump_mode([])
        self.assertEqual(result, [])

    @patch("msit.module.probe.config_initiator.validate_params.DumpConst.ALL_DUMP_MODE", ["mode1"])
    def test_valid_dump_mode_valid(self):
        result = valid_dump_mode(["mode1"])
        self.assertEqual(result, ["mode1"])

    @patch("msit.module.probe.config_initiator.validate_params.DumpConst.ALL_DUMP_MODE", ["mode1"])
    def test_valid_dump_mode_invalid(self):
        with self.assertRaises(MsitException):
            valid_dump_mode(["invalid"])

    @patch("msit.module.probe.config_initiator.validate_params.DumpConst.ALL_DUMP_MODE", ["mode1"])
    def test_invalid_dump_mode(self):
        with self.assertRaises(MsitException):
            valid_dump_mode(12)

    @patch("msit.module.probe.config_initiator.validate_params.DumpConst.ALL_DUMP_MODE", ["mode1", "mode2"])
    def test_invalid_dump_mode_more_element(self):
        with self.assertRaises(MsitException):
            valid_dump_mode(["mode1", "mode2"])

    @patch("msit.module.probe.config_initiator.validate_params.DumpConst.ALL_DUMP_EXTRA", ["extra1"])
    def valid_dump_extra_none(self):
        self.assertIsNone(valid_dump_extra(None))
        result = valid_dump_extra([])
        self.assertEqual(result, [])

    @patch("msit.module.probe.config_initiator.validate_params.DumpConst.ALL_DUMP_EXTRA", ["extra1"])
    def test_valid_dump_extra_valid(self):
        result = valid_dump_extra(["extra1"])
        self.assertEqual(result, ["extra1"])

    @patch("msit.module.probe.config_initiator.validate_params.DumpConst.ALL_DUMP_EXTRA", ["extra1"])
    def test_valid_dump_extra_invalid(self):
        with self.assertRaises(MsitException):
            valid_dump_extra(123)
        with self.assertRaises(MsitException):
            valid_dump_extra(["bad"])       

    @patch("msit.module.probe.config_initiator.validate_params.DumpConst.ALL_DUMP_TIME", ["before", "after"])
    def test_valid_dump_time(self):
        result = valid_dump_time("")
        self.assertEqual(result, "")
        result = valid_dump_time("before")
        self.assertEqual(result, "before")

    @patch("msit.module.probe.config_initiator.validate_params.DumpConst.ALL_DUMP_TIME", ["before", "after"])
    def test_valid_dump_time_invalid_element(self):
        with self.assertRaises(MsitException):
            valid_dump_time(["bad"])
        with self.assertRaises(MsitException):
            valid_dump_time("invalid")

    def test_valid_op_id_none(self):
        result = valid_op_id("")
        self.assertEqual(result, "")

    def test_valid_op_id(self):
        valid_list = [1, "3_1", "4_2_3"]
        result = valid_op_id(valid_list)
        self.assertEqual(result, valid_list)       

    def test_valid_op_id_invalid_element_format(self):
        with self.assertRaises(MsitException):
            valid_op_id(12)
        with self.assertRaises(MsitException):
            valid_op_id([["invalid"]])

    def test_valid_dump_last_logits(self):
        self.assertIsNone(valid_dump_last_logits(None))
        self.assertTrue(valid_dump_last_logits(True))
        with self.assertRaises(MsitException):
            valid_dump_last_logits("true")

    def test_valid_dump_weight(self):
        self.assertIsNone(valid_dump_weight(None))
        self.assertTrue(valid_dump_weight(True))
        with self.assertRaises(MsitException):
            valid_dump_weight("true")

    def test_valid_dump_ge_graph(self):
        self.assertIsNone(valid_dump_ge_graph(None))
        with self.assertRaises(MsitException):
            valid_dump_ge_graph(123)
        with self.assertRaises(MsitException):
            valid_dump_ge_graph("8")
        self.assertEqual(valid_dump_ge_graph("2"), "2")

    def test_valid_dump_graph_level(self):
        self.assertIsNone(valid_dump_graph_level(None))
        with self.assertRaises(MsitException):
            valid_dump_graph_level(123)
        with self.assertRaises(MsitException):
            valid_dump_graph_level("8")
        self.assertEqual(valid_dump_graph_level("2"), "2")

    @patch("msit.module.probe.config_initiator.validate_params.MsitPath")
    def test_valid_fusion_switch_file(self, mock_msit_path):
        self.assertIsNone(valid_fusion_switch_file(None))
        mock_instance = MagicMock()
        mock_instance.check.return_value = "valid"
        mock_msit_path.return_value = mock_instance
        result = valid_fusion_switch_file("some/path")
        self.assertEqual(result, "valid")
        mock_msit_path.assert_called_once()

    def test_valid_device(self):
        self.assertIsNone(valid_device(None))
        self.assertEqual(valid_device("cpu"), "cpu")
        with self.assertRaises(MsitException):
            valid_device("gpu")
        with self.assertRaises(MsitException):
            valid_device(123)

    @patch("msit.module.probe.config_initiator.validate_params.MsitPath")
    def test_valid_weight_path(self, mock_path):
        self.assertIsNone(valid_weight_path(None))
        mock_path().check.return_value = "checked"
        result = valid_weight_path("file.caffemodel")
        self.assertEqual(result, "checked")

    def test_valid_onnx_fusion_switch(self):
        self.assertIsNone(valid_onnx_fusion_switch(None))
        self.assertTrue(valid_onnx_fusion_switch(True))
        with self.assertRaises(MsitException):
            valid_onnx_fusion_switch(123)

    def test_valid_saved_model_tag(self):
        self.assertIsNone(valid_saved_model_tag(None))
        with self.assertRaises(MsitException):
            valid_saved_model_tag(123)
        with self.assertRaises(MsitException):
            valid_saved_model_tag(["%qsc/"])
        self.assertEqual(valid_saved_model_tag(["qazx"]), ["qazx"])

    def test_valid_saved_model_signature(self):
        self.assertIsNone(valid_saved_model_signature(None))
        with self.assertRaises(MsitException):
            valid_saved_model_signature(["%qsc/"])
            valid_saved_model_signature(123)
        self.assertEqual(valid_saved_model_signature("wsx"), "wsx")


class TestValidInputAndOfflineModelInput(unittest.TestCase):
    def test_valid_input_none(self):
        self.assertIsNone(valid_input(None))
    @patch("msit.module.probe.config_initiator.validate_params.OfflineModelInput")
    def test_valid_input_calls_parse(self, mock_input_cls):
        mock_parser = MagicMock()
        mock_input_cls.return_value = mock_parser
        valid_input([{"name": "x"}])
        mock_parser.parse.assert_called_once()

    def test_check_form_not_list(self):
        with self.assertRaisesRegex(MsitException, "The input must be a list."):
            OfflineModelInput("invalid")

    def test_check_form_element_not_dict(self):
        with self.assertRaisesRegex(MsitException, "Each element in the input must be a dictionary."):
            OfflineModelInput([1, 2])

    def test_check_name_missing(self):
        with self.assertRaisesRegex(MsitException, "Each input must have a name."):
            OfflineModelInput([{}])._check_name({})

    def test_check_input_shape_invalid_type(self):
        with self.assertRaisesRegex(MsitException, "must be a list"):
            OfflineModelInput([{}])._check_input_shape({"shape": "not_list"}, "input1")

    def test_check_input_shape_element_not_int(self):
        with self.assertRaisesRegex(MsitException, "support only integers"):
            OfflineModelInput([{}])._check_input_shape({"shape": [1, "a"]}, "input1")

    @patch("msit.module.probe.config_initiator.validate_params.MsitPath")
    def test_check_input_path_invalid_type(self, mock_path):
        with self.assertRaisesRegex(MsitException, "must be a string"):
            OfflineModelInput([{}])._check_input_path({"path": 123}, "input1")

    @patch("msit.module.probe.config_initiator.validate_params.MsitPath")
    def test_check_input_path_invalid_suffix(self, mock_path):
        with self.assertRaisesRegex(MsitException, "can only accept .npy or .bin"):
            OfflineModelInput([{}])._check_input_path({"path": "file.txt"}, "input1")

    @patch("msit.module.probe.config_initiator.validate_params.MsitPath")
    def test_check_input_path_valid(self, mock_path):
        mock_check = mock_path.return_value.check
        mock_check.return_value = True
        self.assertIsNone(
            OfflineModelInput([{}])._check_input_path({"path": "input.npy"}, "input1")
        )

    @patch("msit.module.probe.config_initiator.validate_params.parse_hyphen")
    def test_parse_shape_range_for_str_hyphen(self, mock_parse):
        mock_parse.return_value = [1, 2]
        result = OfflineModelInput([{}])._parse_shape_range_for_str("1-2")
        self.assertEqual(result, [1, 2])

    def test_parse_shape_range_for_str_comma_valid(self):
        result = OfflineModelInput([{}])._parse_shape_range_for_str("2,3")
        self.assertEqual(result, [2, 3])

    def test_parse_shape_range_for_str_invalid_format(self):
        with self.assertRaisesRegex(MsitException, 'can only contain hyphen'):
            OfflineModelInput([{}])._parse_shape_range_for_str("wrong")

    @patch("msit.module.probe.config_initiator.validate_params.check_int_border")
    @patch("msit.module.probe.config_initiator.validate_params.OfflineModelInput._parse_shape_range_for_str")
    def test_parse_dym_shape_range_mixed(self, mock_parse, mock_check):
        mock_parse.return_value = [1, 2]
        input_obj = OfflineModelInput([{}])
        result = input_obj._parse_dym_shape_range(["1-2", 3], "input1")
        self.assertIsInstance(result, list)

    def test_parse_dym_shape_range_invalid_type(self):
        with self.assertRaisesRegex(MsitException, "must be a list"):
            OfflineModelInput([{}])._parse_dym_shape_range("not_list", "input1")

    def test_parse_dym_shape_range_element_type_error(self):
        with self.assertRaisesRegex(MsitException, "support only string and integers"):
            OfflineModelInput([{}])._parse_dym_shape_range([1.5], "input1")

    @patch("msit.module.probe.config_initiator.validate_params.logger")
    @patch.object(OfflineModelInput, "_parse_dym_shape_range")
    def test_check_dym_shape_with_path(self, mock_parse_dym, mock_logger):
        mock_parse_dym.return_value = [[1, 2], [3, 4]]
        input_obj = OfflineModelInput([{}])
        result = input_obj._check_dym_shape({"name": "x", "dym_shape": ["1-2"], "path": "input.npy"}, "x")
        self.assertEqual(result["path"], "")
        self.assertEqual(result["shape"], [])

    def test_draw_shape_and_path_static(self):
        input_obj = OfflineModelInput([{}])
        input_obj.is_need_expand_shape = False
        shapes, paths = input_obj._draw_shape_and_path([{"name": "x", "shape": [1, 2], "path": "x.npy"}])
        self.assertEqual(shapes["x"], [1, 2])
        self.assertEqual(paths, ["x.npy"])

    def test_draw_shape_and_path_dynamic_valid(self):
        input_obj = OfflineModelInput([{}])
        input_obj.is_need_expand_shape = True
        input_data = [
            {"name": "x", "dym_shape": [[1], [2]]},
            {"name": "y", "dym_shape": [[3], [4]]}
        ]
        shapes, paths = input_obj._draw_shape_and_path(input_data)
        self.assertEqual(len(shapes), 2)
        self.assertIsNone(paths)

    def test_draw_shape_and_path_dynamic_invalid(self):
        input_obj = OfflineModelInput([{}])
        input_obj.is_need_expand_shape = True
        input_data = [
            {"name": "x", "dym_shape": [[1], [2], [3]]},
            {"name": "y", "dym_shape": [[4], [5]]}
        ]
        with self.assertRaisesRegex(MsitException, "same expanded dynamic shape length"):
            input_obj._draw_shape_and_path(input_data)

    @patch.object(OfflineModelInput, "_check_name", return_value="x")
    @patch.object(OfflineModelInput, "_check_input_shape")
    @patch.object(OfflineModelInput, "_check_input_path")
    @patch.object(OfflineModelInput, "_check_dym_shape", side_effect=lambda x, y: x)
    @patch.object(OfflineModelInput, "_draw_shape_and_path", return_value=({}, []))
    def test_parse_calls_all(self, mock_draw, mock_dym, mock_path, mock_shape, mock_name):
        input_obj = OfflineModelInput([{"name": "x"}])
        result = input_obj.parse()
        self.assertEqual(result, ({}, []))
