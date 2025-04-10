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

from msit.utils.constants import CfgConst, MsgConst, PathConst
from msit.utils.exceptions import MsitException
from msit.utils.log import LOG_LEVEL
from msit.utils.path import MsitPath, is_dir, is_file

_INT_BORDER = [0, 1e6]
_HYPHEN_NUM_PATTERN = r"^(?:\d+-\d+|\d+-\d+-\d+)$"


def valid_task(value: str):
    if value not in CfgConst.ALL_TASK:
        raise MsitException(MsgConst.INVALID_ARGU, f'"task" must be one of {CfgConst.ALL_TASK}, currently: {value}.')
    return value


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
        raise MsitException(MsgConst.INVALID_ARGU, f"Please check the `--exec (-e)`, currently: {values}.")
    return values


class CheckExec(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_exec(values)
        setattr(namespace, self.dest, values)


def valid_config_path(value: str):
    return MsitPath(value, PathConst.FILE, "r", PathConst.SIZE_2G, PathConst.SUFFIX_JSON).check()


class CheckConfigPath(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_config_path(values)
        setattr(namespace, self.dest, values)


def valid_framework(value: str):
    if not value:
        return value
    if value not in CfgConst.ALL_FRAMEWORK:
        raise MsitException(
            MsgConst.INVALID_ARGU, f'"framework" must be one of {CfgConst.ALL_FRAMEWORK}, currently: {value}.'
        )
    return value


class CheckFramework(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = valid_framework(values)
        setattr(namespace, self.dest, values)


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


def valid_level(values: list):
    if not values:
        return values
    for value in values:
        if value not in CfgConst.ALL_LEVEL:
            raise MsitException(
                MsgConst.INVALID_ARGU, f'"level" must be one of {CfgConst.ALL_LEVEL}, currently: {value}.'
            )
    return values


def valid_log_level(value: str):
    if not value:
        return value
    log_level = {level.lower() for level in LOG_LEVEL}
    if value not in log_level:
        raise MsitException(MsgConst.INVALID_ARGU, f'"log_level" must be one of {log_level}, currently: {value}.')
    return value


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
