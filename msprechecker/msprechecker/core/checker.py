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

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List, Optional, Union


class Severity(IntEnum):
    """
    Failure severity.  IntEnum gives free ordering so callers can write:

        if outcome.severity >= Severity.WARNING:
            log.warning(...)

    instead of:

        if outcome.severity == Severity.ERROR or outcome.severity == Severity.WARNING:
            log.warning(...)

    which is both more verbose and vulnerable to human error.
    """

    INFO = 0  # Suboptimal but not blocking
    WARNING = 1  # Possible issue; worth investigating
    ERROR = 2  # Blocking; the deployment should not proceed


@dataclass(frozen=True)
class Passed:
    """The check ran and found no problems."""

    result_text: str = "ok"


@dataclass(frozen=True)
class Skipped:
    """The check was intentionally not run (e.g. not applicable to this scenario)."""

    reason: str


@dataclass(frozen=True)
class Failed:
    """The check detected a problem."""

    msg: str
    severity: Severity
    result_text: str
    # Only populated for unexpected exceptions captured by the runner.
    traceback: Optional[str] = None


#: Union of all possible check outcomes.  Use isinstance() to discriminate.
CheckOutcome = Union[Passed, Skipped, Failed]


@dataclass(frozen=True)
class CheckGroup:
    """
    A logical section that groups related checks in the report.

    Attributes:
        key:   Machine-readable identifier (e.g. ``"system"``).
        title: Human-readable section header shown in the collection log.
    """

    key: str
    title: str


@dataclass(frozen=True)
class Check:
    """
    A single executable check.

    The ``fn`` callable takes no arguments; any runtime parameters
    (framework, scene, thresholds, …) must be captured via closure or
    ``functools.partial`` at construction time in ``suite.py``.

    Attributes:
        description: One-line label shown next to the result in the report.
        group:       Section this check belongs to.
        fn:          Zero-argument callable returning a ``CheckOutcome``.
    """

    description: str
    group: CheckGroup
    fn: Callable[[], CheckOutcome]


@dataclass
class CheckRecord:
    """Pairs a ``Check`` with the ``CheckOutcome`` produced when it ran."""

    check: Check
    outcome: CheckOutcome

    @property
    def passed(self) -> bool:
        return isinstance(self.outcome, Passed)

    @property
    def skipped(self) -> bool:
        return isinstance(self.outcome, Skipped)

    @property
    def failed(self) -> bool:
        return isinstance(self.outcome, Failed)


def has_errors(records: List[CheckRecord]) -> bool:
    """Return True if any record failed with ERROR severity."""
    return any(r.failed and r.outcome.severity == Severity.ERROR for r in records)
