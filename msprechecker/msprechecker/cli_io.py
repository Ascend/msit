# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
#
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""User-visible CLI output vs diagnostic logging.

``precheck`` stdout layout::

    [现状]  banner / environment snapshot
    ---
    checking → logs (if any) → summary → notes (if any)   ← ``print`` only

Libraries log via ``logging.getLogger(__name__)``; they never ``print``.
"""

from __future__ import annotations

import logging
import shutil
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .util import LOG_FORMAT, LOG_LEVELS

if TYPE_CHECKING:
    from collections.abc import Iterator

_DEFAULT_LOG_LEVEL = "warning"


def configure_cli_logging(level_name: str | None = None) -> None:
    """Stream library logs to stderr (dump/compare/run without layered layout)."""
    name = (level_name or _DEFAULT_LOG_LEVEL).lower()
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    logging.basicConfig(
        format=LOG_FORMAT,
        level=LOG_LEVELS.get(name, logging.WARNING),
        stream=sys.stderr,
    )


def user_print(*parts: object, sep: str = " ", end: str = "\n") -> None:
    """User-facing content on stdout (banner, report, log section). Not logging."""
    print(*parts, sep=sep, end=end, flush=True)


def report(*parts: object, sep: str = " ", end: str = "\n") -> None:
    """Alias for :func:`user_print`."""
    user_print(*parts, sep=sep, end=end)


def report_error(*parts: object, sep: str = " ", end: str = "\n") -> None:
    """Fatal CLI errors on stderr (still not the logging system)."""
    print(*parts, sep=sep, end=end, file=sys.stderr, flush=True)


def section_divider_line() -> str:
    """Full-width rule between stdout sections (matches banner width)."""
    cols, _ = shutil.get_terminal_size()
    return "-" * cols


def print_section_divider() -> None:
    user_print(section_divider_line())


class _BufferHandler(logging.Handler):
    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__()
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(record)


@contextmanager
def capture_cli_logs(level_name: str) -> Iterator[list[logging.LogRecord]]:
    """Capture root logger output into a list; nothing is written until :func:`print_log_section`."""
    level = LOG_LEVELS.get(level_name.lower(), logging.WARNING)
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    records: list[logging.LogRecord] = []

    handler = _BufferHandler(records)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    try:
        yield records
    finally:
        root.handlers.clear()
        for old in saved_handlers:
            root.addHandler(old)
        root.level = saved_level


def print_log_section(records: list[logging.LogRecord]) -> None:
    """Print the [日志] section on stdout. No output when there are no records."""
    if not records:
        return
    formatter = logging.Formatter(LOG_FORMAT)
    for record in records:
        user_print(formatter.format(record))
