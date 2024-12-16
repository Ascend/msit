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

import os
from glob import glob
from re import match
from argparse import Action
from itertools import product

from msit.utils.path import MsitPath
from msit.common.constants import PathConst, MsgConst
from msit.common.exceptions import MsitException


def check_int_border(*args):
    if not all(MsgConst.INT_BORDER[0] <= num <= MsgConst.INT_BORDER[1] for num in args):
        raise MsitException(MsgConst.INVALID_ARGU, f"The integer range is limited to {MsgConst.INT_BORDER}.")


def parse_tilde(element):
    split_ele = element.split("~")
    if len(split_ele) == 2 or len(split_ele) == 3:
        start = int(split_ele[0])
        end = int(split_ele[1])
        check_int_border(start, end)
        if start > end:
            raise MsitException(MsgConst.INVALID_ARGU, \
                                f"The left value must be smaller than the right. Currently: {start} v.s. {end}.")
        step = int(split_ele[2]) if len(split_ele) == 3 else 1
        ranges = [i for i in range(start, end + 1, step)]
    else:
        raise MsitException(MsgConst.INVALID_ARGU, "The tilde must split into two or three parts.")
    return ranges


class CheckExec(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > 2:
            raise MsitException(MsgConst.INVALID_ARGU, \
                                f"The length of the `--exec` argument cannot exceed 2.")
        elif len(values) == 2:
            interpreter, script = values
            if not script.endswith(PathConst.SUFFIX_ONLINE_SCRIPT):
                raise MsitException(MsgConst.INVALID_ARGU, \
                                    f"The online script must be one of {PathConst.SUFFIX_ONLINE_SCRIPT}")
            elif script.endswith(PathConst.SUFFIX_SH):
                if not interpreter.startswith(PathConst.INTERPRETER_BASH):
                    raise MsitException(MsgConst.INVALID_ARGU, \
                                        f"The interpreter must start with bash when the script ends with .sh")
            elif script.endswith(PathConst.SUFFIX_PY):
                if not interpreter.startswith(PathConst.INTERPRETER_PYTHON):
                    raise MsitException(MsgConst.INVALID_ARGU, \
                                        f"The interpreter must start with python when the script ends with .py")
        elif len(values) == 1:
            model_path = values[0]
            if os.path.isdir(model_path):
                _ = MsitPath(model_path, PathConst.DIR, "r", PathConst.SIZE_50G).check()
            elif os.path.isfile(model_path):
                if not model_path.endswith(PathConst.SUFFIX_OFFLINE_MODEL):
                    raise MsitException(MsgConst.INVALID_ARGU, \
                                        f"The offline model must be one of {PathConst.SUFFIX_OFFLINE_MODEL}.")
                _ = MsitPath(model_path, PathConst.FILE, "r", PathConst.SIZE_50G).check()
            else:
                raise MsitException(MsgConst.INVALID_ARGU, \
                                    f"The offline model is neither a valid directory nor a valid file. "
                                    f"Please check if the path exists.")
        setattr(namespace, self.dest, values)


class CheckDumpPath(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = MsitPath(values, PathConst.DIR, "w").check()
        setattr(namespace, self.dest, values)


class CheckRankorStep(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        res = []
        for element in values:
            if not match(MsgConst.TILDE_NUM_PATTERN, element):
                raise MsitException(MsgConst.INVALID_ARGU, \
                                    'The rank or step only accepts numbers or a range like "123~456", "123~456~2".')
            if "~" in element:
                res.extend(parse_tilde(element))
            else:
                check_int_border(int(element))
                res.append(int(element))
        res = list(set(res))
        res.sort()
        setattr(namespace, self.dest, res)


class CheckInputShape(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > 0:
            input_shape = {}
            for name_shape in values:
                if ":" not in name_shape:
                    raise MsitException(MsgConst.INVALID_ARGU, \
                                        f"Input shape must be connected with a colon between the name and the shape.")
                split_name_shape = name_shape.split(":")
                if len(split_name_shape) != 2:
                    raise MsitException(MsgConst.INVALID_ARGU, \
                                        f'The format for input shape should be like "input0:1,224,224,3".')
                name, shape = split_name_shape
                try:
                    input_shape[name] = list(map(int, shape.split(",")))
                except Exception as e:
                    raise MsitException(MsgConst.INVALID_ARGU, \
                                        f'The correct format for input shape should be "input0:1,224,224,3".') from e
        else:
            input_shape = values
        setattr(namespace, self.dest, input_shape)


class CheckInputPath(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1:
            if os.path.isdir(values[0]):
                values = glob(f"{values[0]}/*{PathConst.SUFFIX_NPY}") + glob(f"{values[0]}/*{PathConst.SUFFIX_BIN}")
        elif len(values) > 1:
            for file in values:
                if not file.endswith((PathConst.SUFFIX_BIN, PathConst.SUFFIX_NPY)):
                    raise MsitException(MsgConst.INVALID_ARGU, f"Input path can only accept .npy or .bin files.")
                _ = MsitPath(file, PathConst.FILE, "r", PathConst.SIZE_10G).check()
        setattr(namespace, self.dest, values)


class CheckDymShapeRange(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        shapes_dict = {}
        for name_shapes in values:
            if ":" not in name_shapes:
                raise MsitException(MsgConst.INVALID_ARGU, f"No colon in the dynamic shape range.")
            if len(name_shapes.split(":")) != 2:
                raise MsitException(MsgConst.INVALID_ARGU, MsgConst.DSR_ERROR)
            name, shapes = name_shapes.split(":")
            if match(MsgConst.DSR_PATTERN, shapes):
                shapes_dict[name] = self._parse_dym_shape_range(shapes)
            else:
                raise MsitException(MsgConst.INVALID_ARGU, MsgConst.DSR_ERROR)  
        setattr(namespace, self.dest, shapes_dict)

    @staticmethod
    def _parse_dym_shape_range(shapes):
        shapes_list = []
        for shape in shapes.split(","):
            if "~" in shape:
                ranges = parse_tilde(shape)
            elif "-" in shape:
                ranges = list(map(int, shape.split("-")))
                if len(ranges) != 2:
                    raise MsitException(MsgConst.INVALID_ARGU, MsgConst.DSR_ERROR)
            else:
                ranges = [int(shape)]
            shapes_list.append(ranges)
        return [list(s) for s in list(product(*shapes_list))]
