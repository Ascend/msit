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
# -------------------------------------------------------------------------

__all__ = [
    "BannerPresenter",
    "CmdStrategy",
    "CmdType",
    "setup_compare_parser",
    "Compare",
    "setup_dump_parser",
    "Dump",
]

from .banner import BannerPresenter
from .base import CmdStrategy, CmdType
from .compare import Compare, setup_compare_parser
from .dump import Dump, setup_dump_parser
from .precheck import setup_precheck_parser


class CmdStrategyFactory:
    def __init__(self) -> None:
        self._registry = {
            # CmdType.PRECHECK: Precheck,
            CmdType.DUMP: Dump,
            CmdType.COMPARE: Compare,
        }

    def register(self, cmd_type, strategy_cls) -> None:
        if not issubclass(strategy_cls, CmdStrategy):
            raise TypeError(
                f"Expected 'strategy_cls' to be 'Cmdstrategy' class name. Got {strategy_cls} instead"
            )

        self._registry[cmd_type] = strategy_cls

    def get(self, cmd: CmdType) -> CmdStrategy:
        if cmd not in self._registry:
            raise ValueError(f"No strategy registered for command: {cmd}")

        return self._registry[cmd]()
