# -*- coding: utf-8 -*-
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

from enum import Enum
from typing import Dict
from dataclasses import dataclass
from pathlib import Path

_PREFILL = "prefill"
_DECODE = "decode"


class StateType(Enum):
    DEFAULT = 0
    LINE = 1


@dataclass
class State:
    prefill: int = 0
    decode: int = 0
    batch_prefill: int = 0
    batch_decode: int = 0

    def __repr__(self):
        return f"TT_{self.prefill}_{self.decode}_{self.batch_prefill}_{self.batch_decode}"

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __ne__(self, other):
        return self.__repr__() != other.__repr__()

    def sum(self):
        return self.prefill + self.decode + self.batch_prefill + self.batch_decode


def computer_speed(line_node, field):
    return 1 / (getattr(line_node, field) * 10 ** -3)


def computer_speed_with_second(line_node, field):
    return 1 / (getattr(line_node, field) * 10 ** -3)


def my_std(nums):
    n = len(nums)
    avg = sum(nums) / n
    return (sum(map(lambda e: (e - avg) * (e - avg), nums)) / n) ** 0.5


def get_train_sub_path(base_path: Path = Path("state_eval/tmp/pd_content/train")):
    # 给训练输出目录生成新的目录
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
    _sub_len = len([0 for _ in base_path.iterdir()])
    _sub_dir = base_path.joinpath(f"{_sub_len + 1}")
    _sub_dir.mkdir(parents=True, exist_ok=True)
    return _sub_dir


def update_global_coefficient(global_coefficient: Dict, key: State, value: float) -> None:
    if key not in global_coefficient:
        global_coefficient[key] = [value]
    else:
        global_coefficient[key].append(value)
