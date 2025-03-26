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
from argparse import Action
from itertools import product

from msit.utils.constants import DumpConst, MsgConst, PathConst
from msit.utils.exceptions import MsitException
from msit.utils.io import load_json
from msit.utils.log import _LOG_LEVEL, logger
from msit.utils.path import MsitPath, is_dir, is_file

_INT_BORDER = [0, 1e6]
_HYPHEN_NUM_PATTERN = r"^(?:\d+-\d+|\d+-\d+-\d+)$"
_OP_ID_PATTERN = r"^\d{1,10}(_\d{1,10}){0,9}$"
_ALL_DEVICE = ["cpu", "npu"]


def check_int_border(*args):
    for num in args:
        if not (_INT_BORDER[0] <= num <= _INT_BORDER[1]):
            raise MsitException(
                MsgConst.INVALID_ARGU, f"The integer range is limited to {_INT_BORDER}, currently: {num}."
            )


def parse_hyphen(element):
    if not re.match(_HYPHEN_NUM_PATTERN, element):
        raise MsitException(MsgConst.INVALID_ARGU, 'Only accepts numbers or a range like "123-456", "123-456-2".')
    split_ele = element.split("-")
    if len(split_ele) == 2 or len(split_ele) == 3:
        start = int(split_ele[0])
        end = int(split_ele[1])
        check_int_border(start, end)
        if start > end:
            raise MsitException(
                MsgConst.INVALID_ARGU, f"The left value must be smaller than the right, currently: {start} v.s. {end}."
            )
        step = int(split_ele[2]) if len(split_ele) == 3 else 1
        ranges = [i for i in range(start, end + 1, step)]
        return ranges
    else:
        raise MsitException(MsgConst.INVALID_ARGU, "The hyphen must split into two or three parts.")


def valid_exec(values: list):
    if not values:
        return values
    first_keyword = values[0]
    if is_dir(first_keyword):
        _ = MsitPath(first_keyword, PathConst.DIR, "r", PathConst.SIZE_50G).check()
    elif first_keyword == "bash":
        try:
            if not values[1].endswith(PathConst.SUFFIX_SH):
                raise MsitException(
                    MsgConst.INVALID_ARGU, "The interpreter must start with bash when the script ends with .sh."
                )
        except Exception as e:
            raise MsitException(MsgConst.PARSING_FAILED) from e
    elif first_keyword in ["python", "python3"]:
        try:
            if not values[1].endswith(PathConst.SUFFIX_PY):
                raise MsitException(
                    MsgConst.INVALID_ARGU, "The interpreter must start with python when the script ends with .py."
                )
        except Exception as e:
            raise MsitException(MsgConst.PARSING_FAILED) from e
    elif is_file(first_keyword):
        if not first_keyword.endswith(PathConst.SUFFIX_OFFLINE_MODEL + PathConst.SUFFIX_ONLINE_SCRIPT):
            raise MsitException(
                MsgConst.INVALID_ARGU,
                "A single readable or executable file must end with "
                f"{PathConst.SUFFIX_OFFLINE_MODEL + PathConst.SUFFIX_ONLINE_SCRIPT}.",
            )
        _ = MsitPath(first_keyword, PathConst.FILE, "r", PathConst.SIZE_50G).check()
    else:
        raise MsitException(MsgConst.INVALID_ARGU, f'Please check the "exec" or `--exec (-e)`, currently: {values}.')
    return values


class CheckExec(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_exec(values)
        setattr(namespace, self.dest, values)


def valid_dump_path(value: str):
    return MsitPath(value, PathConst.DIR, "w").check()


class CheckDumpPath(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_dump_path(values)
        setattr(namespace, self.dest, values)


def valid_dump_task(value: str):
    if value not in DumpConst.ALL_DUMP_TASK:
        raise MsitException(
            MsgConst.INVALID_ARGU,
            f'"task" or `--dump-task (-task)` must be one of {DumpConst.ALL_DUMP_TASK}, currently: {value}.',
        )
    return value


class CheckDumpTask(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_dump_task(values)
        setattr(namespace, self.dest, values)


def valid_dump_level(values: list):
    for value in values:
        if value not in DumpConst.ALL_DUMP_LEVEL:
            raise MsitException(
                MsgConst.INVALID_ARGU,
                f'"level" or `--dump-level (-level)` must be one of {DumpConst.ALL_DUMP_LEVEL}, currently: {value}.',
            )
    return values


class CheckDumpLevel(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_dump_level(values)
        setattr(namespace, self.dest, values)


def valid_dump_mode(value: str):
    if value not in DumpConst.ALL_DUMP_MODE:
        raise MsitException(
            MsgConst.INVALID_ARGU,
            f'"dump_mode" or `--dump-mode (-mode)` must be one of {DumpConst.ALL_DUMP_MODE}, currently: {value}.',
        )
    return value


class CheckDumpMode(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_dump_mode(values)
        setattr(namespace, self.dest, values)


def valid_log_level(value: str):
    log_level = [level.lower() for level in _LOG_LEVEL]
    if value not in log_level:
        raise MsitException(
            MsgConst.INVALID_ARGU,
            f'"log_level" or `--log-level (-logl)` must be one of {log_level}, currently: {value}.',
        )
    return value


class CheckLogLevel(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_log_level(values)
        setattr(namespace, self.dest, values)


def valid_seed(value):
    if not value:
        return value
    if isinstance(value, int):
        check_int_border(value)
        return value
    else:
        raise MsitException(
            MsgConst.INVALID_ARGU, f"Seed value must be an integer in the range {_INT_BORDER}, currently: {value}."
        )


class CheckSeed(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_seed(values)
        setattr(namespace, self.dest, values)


def valid_step_or_rank(values: list):
    res = []
    for element in values:
        if isinstance(element, str):
            res.extend(parse_hyphen(element))
        elif isinstance(element, int):
            check_int_border(element)
            res.append(element)
        else:
            raise MsitException(
                MsgConst.INVALID_DATA_TYPE, 'Elements in the "rank" or "step" support only strings and integers.'
            )
    res = list(set(res))
    res.sort()
    return res


class CheckStepOrRank(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_step_or_rank(values)
        setattr(namespace, self.dest, values)


def valid_dump_extra(values: list):
    for vv in values:
        if vv not in DumpConst.ALL_DUMP_EXTRA:
            raise MsitException(
                MsgConst.INVALID_ARGU,
                f'"dump_extra" or `--dump-extra (-extra)` must be one of {DumpConst.ALL_DUMP_EXTRA}, currently: {vv}.',
            )
    return values


class CheckDumpExtra(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_dump_extra(values)
        setattr(namespace, self.dest, values)


def valid_dump_time(value):
    if value not in DumpConst.ALL_DUMP_TIME:
        raise MsitException(
            MsgConst.INVALID_ARGU,
            f'"dump_time" or `--dump-time (-time)` must be one of {DumpConst.ALL_DUMP_TIME}, currently: {value}.',
        )
    return str(value)


class CheckDumpTime(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_dump_time(values)
        setattr(namespace, self.dest, values)


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
                '"op_id" or --operation-id (-op-id) is only supported in the ATB dump scenario, '
                f"with formats like 2, 3_1, or 3_1_2, currently: {element}.",
            )
    return res


class CheckOpId(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_op_id(values)
        setattr(namespace, self.dest, values)


def valid_dump_last_logits(value):
    if not isinstance(value, bool):
        raise MsitException(
            MsgConst.INVALID_DATA_TYPE, '"dump_last_logits" or `--dump-last-logits (-llogits)` must be a boolean.'
        )
    return value


def valid_dump_weight(value):
    if not isinstance(value, bool):
        raise MsitException(MsgConst.INVALID_DATA_TYPE, '"dump_weight" or `--dump-weight (-w)` must be a boolean.')
    return value


def valid_dump_ge_graph(value):
    if value not in DumpConst.ALL_DUMP_GE_GRAPH:
        raise MsitException(
            MsgConst.INVALID_ARGU,
            '"dump_ge_graph" or `--dump-ge-graph (--geg)` must be one of '
            f"{DumpConst.ALL_DUMP_GE_GRAPH}, currently: {value}.",
        )
    return str(value)


class CheckDumpGeGraph(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_dump_ge_graph(values)
        setattr(namespace, self.dest, values)


def valid_dump_graph_level(value):
    if value not in DumpConst.ALL_DUMP_GRAPH_LEVEL:
        raise MsitException(
            MsgConst.INVALID_ARGU,
            '"dump_graph_level" or `--dump-graph-level (-gegl)` must be one of '
            f"{DumpConst.ALL_DUMP_GRAPH_LEVEL}, currently: {value}.",
        )
    return str(value)


class CheckDumpGraphLevel(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_dump_graph_level(values)
        setattr(namespace, self.dest, values)


def valid_fusion_switch_file(value):
    if value:
        return MsitPath(value, PathConst.FILE, "r", PathConst.SIZE_500M, PathConst.SUFFIX_FSF).check()
    else:
        return ""


class CheckFusionSwitchFile(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_fusion_switch_file(values)
        setattr(namespace, self.dest, values)


def valid_device(value: str):
    if not value:
        return value
    if value not in _ALL_DEVICE:
        raise MsitException(
            MsgConst.INVALID_ARGU, f'"device" or `--device (-d)` must be one of {_ALL_DEVICE}, currently: {value}.'
        )
    return value


class CheckDevice(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_device(values)
        setattr(namespace, self.dest, values)


def valid_input_json(values: str):
    if values:
        return MsitPath(values, PathConst.FILE, "r", PathConst.SIZE_10G, PathConst.SUFFIX_JSON).check()
    else:
        raise MsitException(
            MsgConst.REQUIRED_ARGU_MISSING,
            f'"input_json" or `--input-json (-injson)` must be configured, currently: {values}.',
        )


class CheckInputJson(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_input_json(values)
        setattr(namespace, self.dest, values)


class InputJson:
    def __init__(self, json_path):
        self.input_file = load_json(valid_input_json(json_path))
        self._check_form()
        self.is_need_expand_shape = False

    @staticmethod
    def _check_name(infile: dict):
        if not infile.get("name"):
            raise MsitException(MsgConst.PARSING_FAILED, "The input JSON file must specify a name for each input.")

    @staticmethod
    def _check_input_shape(infile: dict):
        inshape = infile.get("shape")
        if inshape:
            if not isinstance(inshape, list):
                raise MsitException(MsgConst.INVALID_DATA_TYPE, '"shape" of the input JSON must be a list.')
            for vv in inshape:
                if not isinstance(vv, int):
                    raise MsitException(
                        MsgConst.INVALID_DATA_TYPE, 'Elements in "shape" of the input JSON support only integers.'
                    )

    @staticmethod
    def _check_input_path(infile: dict):
        inpath = infile.get("path")
        if inpath:
            if not isinstance(inpath, str):
                raise MsitException(MsgConst.INVALID_DATA_TYPE, '"path" of the input JSON must be a string.')
            if not inpath.endswith((PathConst.SUFFIX_BIN, PathConst.SUFFIX_NPY)):
                raise MsitException(
                    MsgConst.INVALID_ARGU, f'"path" can only accept .npy or .bin files, currently: {inpath}.'
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
                    f"Both sides of the hyphen (-) in the input JSON must be numbers, currently: {shape}.",
                ) from e
        else:
            raise MsitException(
                MsgConst.INVALID_ARGU, 'The "dym_shape" of the input JSON can only contain hyphen (-) or a comma (,).'
            )
        return ranges

    def parse(self):
        logger.info("Start parsing the input JSON file.")
        modify_file = []
        for infile in self.input_file:
            self._check_name(infile)
            self._check_input_shape(infile)
            self._check_input_path(infile)
            infile = self._check_dym_shape(infile)
            modify_file.append(infile)
        shapes, paths = self._draw_shape_and_path(modify_file)
        return shapes, paths

    def _parse_dym_shape_range(self, shapes):
        if not isinstance(shapes, list):
            raise MsitException(MsgConst.INVALID_DATA_TYPE, '"dym_shape" of the input JSON must be a list.')
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
                    f'Elements in "dym_shape" of the input JSON support only string and integers, currently: {shape}.',
                )
            shapes_list.append(ranges)
        return [list(s) for s in list(product(*shapes_list))]

    def _check_form(self):
        if isinstance(self.input_file, list):
            for vv in self.input_file:
                if not isinstance(vv, dict):
                    raise MsitException(
                        MsgConst.INVALID_DATA_TYPE, "Each element in the input JSON must be a dictionary."
                    )
        else:
            raise MsitException(MsgConst.INVALID_DATA_TYPE, "The input JSON file must be a non-empty list.")

    def _check_dym_shape(self, infile: dict):
        if infile.get("dym_shape"):
            self.is_need_expand_shape = True
            infile["dym_shape"] = self._parse_dym_shape_range(infile["dym_shape"])
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
    if not isinstance(value, bool):
        raise MsitException(
            MsgConst.INVALID_DATA_TYPE,
            f'"onnx_fusion_switch" or `--onnx-fusion-switch (-ofs)` must be a boolean, currently: {value}.',
        )
    return value


def valid_weight_path(value: str):
    if value:
        return MsitPath(value, PathConst.FILE, "r", PathConst.SIZE_50G, PathConst.SUFFIX_CAFFEMODEL).check()
    else:
        raise MsitException(
            MsgConst.REQUIRED_ARGU_MISSING, "When using Caffe for inference, a weight file (.caffemodel) is required."
        )


class CheckWeightPath(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_weight_path(values)
        setattr(namespace, self.dest, values)
