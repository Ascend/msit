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

import logging
from typing import Any, Dict

from ..strategies import CollectStrategy, CollectStrategyGroup
from ..utils import LOGGER


class Collector:
    def __init__(self, collect_strategies=None):
        self._collect_strategies = []
        if collect_strategies is not None:
            try:
                self._collect_strategies = list(collect_strategies)
            except TypeError:
                LOGGER.error(
                    "collect_strategies must be an iterable. Got %s instead",
                    collect_strategies,
                )
                raise

        if not all(
                isinstance(strategy, (CollectStrategy, CollectStrategyGroup))
                for strategy in self._collect_strategies
        ):
            raise TypeError(
                "All collect_strategies must be instances of CollectStrategy"
            )

    def add_strategy(self, collect_strategy) -> None:
        if not isinstance(collect_strategy, CollectStrategy):
            raise TypeError("collect_strategy must be an instance of CollectStrategy")

        self._collect_strategies.append(collect_strategy)

    def collect(self) -> Dict[str, Dict[Any, Any]]:
        return {
            strategy.name: strategy.execute() for strategy in self._collect_strategies
        }

    def sync(self, target):
        for strategy in self._collect_strategies:
            strategy.sync(target)
