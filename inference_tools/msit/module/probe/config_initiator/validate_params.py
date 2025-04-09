# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from itertools import product

from msit.common.validation import check_int_border, parse_hyphen
from msit.utils.constants import CfgConst, DumpConst, MsgConst, PathConst
from msit.utils.exceptions import MsitException
from msit.utils.log import logger
from msit.utils.path import MsitPath

_OP_ID_PATTERN = r"^\d{1,10}(_\d{1,10}){0,9}$"
_ALL_DEVICE = {"cpu", "npu"}
_VALID_CHAR = r"^[a-zA-Z0-9_.-]+$"


def valid_dump_path(value: str):
    return MsitPath(value, PathConst.DIR, "w").check()


def valid_dump_format(value: str):
    if not value:
        return value
    if value not in DumpConst.ALL_DUMP_FORMAT:
        raise MsitException(
            MsgConst.INVALID_ARGU, f'"dump_format" must be one of {DumpConst.ALL_DUMP_FORMAT}, currently: {value}.'
        )
    return value


def valid_list(value: dict):
    if not value:
        return value
    if not isinstance(value, dict):
        raise MsitException(
            MsgConst.INVALID_DATA_TYPE,
            f"The list must be a dictionary with keys like {CfgConst.ALL_LEVEL}, "
            "and the values must be in list format.",
        )
    for key, vv in value.items():
        if key not in CfgConst.ALL_LEVEL:
            raise MsitException(
                MsgConst.INVALID_ARGU, f"Key not in allowed list {CfgConst.ALL_LEVEL}, currently: {key}."
            )
        if not isinstance(vv, list):
            raise MsitException(MsgConst.INVALID_DATA_TYPE, f"Value must be a list, got {type(vv)} instead.")
    return value


def valid_dump_mode(value: str):
    if not value:
        return value
    if value not in DumpConst.ALL_DUMP_MODE:
        raise MsitException(
            MsgConst.INVALID_ARGU, f'"dump_mode" must be one of {DumpConst.ALL_DUMP_MODE}, currently: {value}.'
        )
    return value


def valid_dump_extra(values: list):
    if not values:
        return values
    for value in values:
        if value not in DumpConst.ALL_DUMP_EXTRA:
            raise MsitException(
                MsgConst.INVALID_ARGU, f'"dump_extra" must be one of {DumpConst.ALL_DUMP_EXTRA}, currently: {value}.'
            )
    return values


def valid_dump_time(value):
    if not value:
        return value
    if value not in DumpConst.ALL_DUMP_TIME:
        raise MsitException(
            MsgConst.INVALID_ARGU, f'"dump_time" must be one of {DumpConst.ALL_DUMP_TIME}, currently: {value}.'
        )
    return str(value)


def valid_op_id(value: list):
    res = []
    for element in value:
        if isinstance(element, int):
            check_int_border(element)
            res.append(element)
        elif isinstance(element, str) and re.match(_OP_ID_PATTERN, element):
            res.append(element)
        else:
            raise MsitException(
                MsgConst.INVALID_DATA_TYPE,
                '"op_id" is only supported in the ATB dump scenario, '
                f"with formats like 2, 3_1, or 3_1_2, currently: {element}.",
            )
    return res


def valid_dump_last_logits(value):
    if not value:
        return value
    if not isinstance(value, bool):
        raise MsitException(MsgConst.INVALID_DATA_TYPE, '"dump_last_logits" must be a boolean.')
    return value


def valid_dump_weight(value):
    if not value:
        return value
    if not isinstance(value, bool):
        raise MsitException(MsgConst.INVALID_DATA_TYPE, '"dump_weight" must be a boolean.')
    return value


def valid_dump_ge_graph(value):
    if not value:
        return value
    if value not in DumpConst.ALL_DUMP_GE_GRAPH:
        raise MsitException(
            MsgConst.INVALID_ARGU, f'"dump_ge_graph" must be one of {DumpConst.ALL_DUMP_GE_GRAPH}, currently: {value}.'
        )
    return str(value)


def valid_dump_graph_level(value):
    if not value:
        return value
    if value not in DumpConst.ALL_DUMP_GRAPH_LEVEL:
        raise MsitException(
            MsgConst.INVALID_ARGU,
            f'"dump_graph_level" must be one of {DumpConst.ALL_DUMP_GRAPH_LEVEL}, currently: {value}.',
        )
    return str(value)


def valid_fusion_switch_file(value):
    if not value:
        return value
    return MsitPath(value, PathConst.FILE, "r", PathConst.SIZE_500M, PathConst.SUFFIX_FSF).check()


def valid_device(value: str):
    if not value:
        return value
    if value not in _ALL_DEVICE:
        raise MsitException(MsgConst.INVALID_ARGU, f'"device" must be one of {_ALL_DEVICE}, currently: {value}.')
    return value


def valid_input(value: list):
    if not value:
        return value
    return OfflineModelInput(value).parse()


class OfflineModelInput:
    def __init__(self, input_list):
        self.input_list = input_list
        self._check_form()
        self.is_need_expand_shape = False

    @staticmethod
    def _check_name(infile: dict):
        if not infile.get("name"):
            raise MsitException(MsgConst.PARSING_FAILED, "Each input must have a name.")
        return infile.get("name")

    @staticmethod
    def _check_input_shape(infile: dict, name):
        inshape = infile.get("shape")
        if inshape:
            if not isinstance(inshape, list):
                raise MsitException(MsgConst.INVALID_DATA_TYPE, f'"shape" of the input {name} must be a list.')
            for vv in inshape:
                if not isinstance(vv, int):
                    raise MsitException(
                        MsgConst.INVALID_DATA_TYPE, f'Elements in "shape" of the input {name} support only integers.'
                    )

    @staticmethod
    def _check_input_path(infile: dict, name):
        inpath = infile.get("path")
        if inpath:
            if not isinstance(inpath, str):
                raise MsitException(MsgConst.INVALID_DATA_TYPE, f'"path" of the input {name} must be a string.')
            if not inpath.endswith((PathConst.SUFFIX_BIN, PathConst.SUFFIX_NPY)):
                raise MsitException(
                    MsgConst.INVALID_ARGU, f'"path" of {name} can only accept .npy or .bin files, currently: {inpath}.'
                )
            _ = MsitPath(inpath, PathConst.FILE, "r", PathConst.SIZE_10G).check()

    @staticmethod
    def _parse_shape_range_for_str(shape):
        if "-" in shape:
            ranges = parse_hyphen(shape)
        elif "," in shape and shape.count(",") == 1:
            try:
                ranges = list(map(int, shape.split(",")))
            except Exception as e:
                raise MsitException(
                    MsgConst.INVALID_ARGU,
                    f"Both sides of the hyphen (-) in the input must be numbers, currently: {shape}.",
                ) from e
        else:
            raise MsitException(
                MsgConst.INVALID_ARGU, 'The "dym_shape" of the input can only contain hyphen (-) or a comma (,).'
            )
        return ranges

    def parse(self):
        logger.info("Start parsing the input list.")
        modify_file = []
        for infile in self.input_list:
            name = self._check_name(infile)
            self._check_input_shape(infile, name)
            self._check_input_path(infile, name)
            infile = self._check_dym_shape(infile, name)
            modify_file.append(infile)
        shapes, paths = self._draw_shape_and_path(modify_file)
        return shapes, paths

    def _parse_dym_shape_range(self, shapes, name):
        if not isinstance(shapes, list):
            raise MsitException(MsgConst.INVALID_DATA_TYPE, f'"dym_shape" of the input {name} must be a list.')
        shapes_list = []
        for shape in shapes:
            if isinstance(shape, str):
                ranges = self._parse_shape_range_for_str(shape)
            elif isinstance(shape, int):
                check_int_border(shape)
                ranges = [shape]
            else:
                raise MsitException(
                    MsgConst.INVALID_DATA_TYPE,
                    f'Elements in "dym_shape" of the input support only string and integers, currently: {shape}.',
                )
            shapes_list.append(ranges)
        return [list(s) for s in list(product(*shapes_list))]

    def _check_form(self):
        if isinstance(self.input_list, list):
            for vv in self.input_list:
                if not isinstance(vv, dict):
                    raise MsitException(MsgConst.INVALID_DATA_TYPE, "Each element in the input must be a dictionary.")
        else:
            raise MsitException(MsgConst.INVALID_DATA_TYPE, "The input must be a list.")

    def _check_dym_shape(self, infile: dict, name: str):
        if infile.get("dym_shape"):
            self.is_need_expand_shape = True
            infile["dym_shape"] = self._parse_dym_shape_range(infile["dym_shape"], name)
            infile["shape"] = []
            if infile.get("path"):
                infile["path"] = ""
            logger.info('Since "dym_shape" is used, "shape" and "path" will not take effect.')
        return infile

    def _draw_shape_and_path(self, modify_file):
        if self.is_need_expand_shape:
            dym_shapes = [item["dym_shape"] for item in modify_file]
            if all(len(shapes) == len(dym_shapes[0]) for shapes in dym_shapes):
                shapes = [dict(zip([item["name"] for item in modify_file], shapes)) for shapes in zip(*dym_shapes)]
                paths = None
            else:
                raise MsitException(
                    MsgConst.INVALID_ARGU, "Ensure all inputs have the same expanded dynamic shape length."
                )
        else:
            shapes, paths = {}, []
            for item in modify_file:
                shapes[item["name"]] = item.get("shape")
                if item.get("path"):
                    paths.append(item["path"])
        return shapes, paths


def valid_onnx_fusion_switch(value):
    if not value:
        return value
    if not isinstance(value, bool):
        raise MsitException(MsgConst.INVALID_DATA_TYPE, f'"onnx_fusion_switch" must be a boolean, currently: {value}.')
    return value


def valid_saved_model_tag(value):
    if not value:
        return value
    if not isinstance(value, list):
        raise MsitException(MsgConst.INVALID_DATA_TYPE, "saved_model_tag msut be a list.")
    for vv in value:
        if not (isinstance(vv, str) and re.match(_VALID_CHAR, vv)):
            raise MsitException(MsgConst.RISK_ALERT, f"Invalid input: contains unsafe characters: {vv}.")
    return value


def valid_saved_model_signature(value):
    if not value:
        return value
    if not (isinstance(value, str) and re.match(_VALID_CHAR, value)):
        raise MsitException(MsgConst.RISK_ALERT, f"Invalid input: contains unsafe characters: {value}.")
    return value


def valid_weight_path(value: str):
    if not value:
        return value
    return MsitPath(value, PathConst.FILE, "r", PathConst.SIZE_50G, PathConst.SUFFIX_CAFFEMODEL).check()
