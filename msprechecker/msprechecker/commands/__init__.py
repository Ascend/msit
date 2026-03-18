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

from enum import Enum
from abc import ABC, abstractmethod


class CmdStrategy(ABC):
    @abstractmethod
    def execute(self) -> int:
        """Execute the command."""
        pass


class CmdType(Enum):
    PRECHECK = "precheck"
    DUMP = "dump"
    COMPARE = "compare"
    RUN = "run"
    INSPECT = "inspect"
    SYNC = "sync"


class CmdStrategyFactory:
    from .cmate import Inspect, Run
    from .compare import Compare
    from .dump import Dump
    from .precheck import Precheck
    from .sync import SyncCmd

    _registry = {
        CmdType.PRECHECK: Precheck,
        CmdType.DUMP: Dump,
        CmdType.COMPARE: Compare,
        CmdType.RUN: Run,
        CmdType.INSPECT: Inspect,
        CmdType.SYNC: SyncCmd,
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
