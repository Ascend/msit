# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
from abc import ABC, abstractmethod
from enum import Enum

from ..utils.ascend import get_weight_dir, get_framework


class CmdType(Enum):
    PRECHECK = "precheck"
    DUMP = "dump"
    COMPARE = "compare"


class CmdStrategy(ABC):
    def __init__(self):
        self._framework = get_framework()
        self._weight_dir = get_weight_dir()

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the command strategy"""
