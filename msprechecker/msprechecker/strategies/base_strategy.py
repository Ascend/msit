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
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Optional, List, Dict

from ..utils import SingletonMeta, Utils


class CollectStrategy(ABC, metaclass=SingletonMeta):

    def __init__(self, name):
        self._name = name
        self._target = None

    @property
    def name(self) -> str:
        return self._name

    @lru_cache(maxsize=None)
    @abstractmethod
    def execute(self) -> Any:
        pass

    def sync(self, target_data: dict) -> Any:
        target_dict = target_data.get(self.name)
        if target_dict is None:
            Utils.log_error_and_exit(f"Failed to sync {self.name} strategy, data from dumped file is empty.")
        self._target = target_dict


class CollectStrategyGroup(ABC):
    __metaclass__ = type

    def __init__(
            self,
            name: str,
            strategies: Optional[List[CollectStrategy]] = None,
    ) -> None:
        self._name = name
        self._target = None
        self._strategies: List[CollectStrategy] = []

        if strategies is not None:
            try:
                strategies = list(strategies)
            except TypeError:
                Utils.log_error_and_exit(
                    "strategies must be an iterable. Got {} instead".format(strategies)
                )

            for strategy in strategies:
                self.add(strategy)

    @property
    def name(self) -> str:
        return self._name

    def add(self, strategy: CollectStrategy) -> "CollectStrategyGroup":
        if not isinstance(strategy, CollectStrategy):
            raise TypeError("collect_strategy must be an instance of CollectStrategy")
        if any(s.name == strategy.name for s in self._strategies):
            raise ValueError(
                f"A strategy with name {strategy.name!r} already exists in this group"
            )
        self._strategies.append(strategy)
        return self

    def execute(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for strategy in self._strategies:
            try:
                results[strategy.name] = strategy.execute()
            except Exception:
                Utils.log_error_and_exit("Strategy {} failed", strategy.name)
                results[strategy.name] = None
        return results

    def sync(self, target_data: dict) -> None:
        target_dict = target_data.get(self.name)
        if target_dict is None:
            Utils.log_error_and_exit(f"Failed to sync {self.name} strategy, data from dumped file is empty.")
        self._target = target_dict
        for strategy in self._strategies:
            strategy.sync(self._target)
