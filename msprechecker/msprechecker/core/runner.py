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

import shutil
import sys
import traceback as tb

from typing import List, Optional

from colorama import Fore, Style

from .checker import (
    Check,
    CheckGroup,
    CheckOutcome,
    CheckRecord,
    Failed,
    Passed,
    Severity,
    Skipped,
)

# ── colour / label tables ─────────────────────────────────────────────────────

_SEV_COLOR = {
    Severity.ERROR: Fore.RED,
    Severity.WARNING: Fore.YELLOW,
    Severity.INFO: Fore.CYAN,
}

# INFO → RECOMMEND makes the intent clearer to operators than "INFO".
_SEV_LABEL = {
    Severity.ERROR: "ERROR",
    Severity.WARNING: "WARNING",
    Severity.INFO: "RECOMMEND",
}

_DIVIDER = "-" * shutil.get_terminal_size().columns


# ── runner ────────────────────────────────────────────────────────────────────


class PrecheckRunner:
    """
    Execute a list of ``Check`` objects and render a two-phase report.

    Phase 1 – Collection log (printed live during execution)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Grouped by ``Check.group``, printed as each check completes::

        - Checking system environment
            - CPU high performance mode - off
            - Kernel version - 5.15.0
        - Checking system environment - done

    Phase 2 – Issues summary (printed after all checks finish)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Only items that failed **and** meet ``min_severity`` are listed::

        -------------------------------------------------------------------
        - [RECOMMEND] 建议开启 CPU 高性能模式 …
        - [ERROR] 当前驱动版本 24.0 低于推荐版本 25.0 …
            Traceback:
                …
    """

    def __init__(self, min_severity: Severity = Severity.INFO) -> None:
        self.min_severity = min_severity

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, checks: List[Check]) -> List[CheckRecord]:
        """
        Execute *checks* in order, print the collection log, then print the
        issues summary.  Returns all records (pass, skip, and fail).
        """
        records: List[CheckRecord] = []
        current_group: Optional[CheckGroup] = None

        for check in checks:
            if check.group != current_group:
                if current_group is not None:
                    self._print_group_done(current_group)
                self._print_group_header(check.group)
                current_group = check.group

            outcome = self._execute(check)
            record = CheckRecord(check=check, outcome=outcome)
            records.append(record)
            self._print_item(record)

        if current_group is not None:
            self._print_group_done(current_group)

        self._print_issues(records)
        return records

    # ── execution ─────────────────────────────────────────────────────────────

    @staticmethod
    def _execute(check: Check) -> CheckOutcome:
        """
        Call ``check.fn()``.  Any unexpected exception is caught and returned
        as a ``Failed`` outcome with the full traceback attached, so every
        error path ends up in the formatted issues report.
        """
        try:
            return check.fn()
        except Exception:
            exc = sys.exc_info()[1]
            return Failed(
                msg=repr(exc),
                severity=Severity.ERROR,
                result_text="error",
                traceback=tb.format_exc(),
            )

    # ── rendering ─────────────────────────────────────────────────────────────

    @staticmethod
    def _print_group_header(group: CheckGroup) -> None:
        print(f"- {Fore.CYAN}{group.title}{Style.RESET_ALL}")

    @staticmethod
    def _print_group_done(group: CheckGroup) -> None:
        print(f"- {Fore.CYAN}{group.title} - done{Style.RESET_ALL}")

    def _print_item(self, record: CheckRecord) -> None:
        print(f"    - {record.check.description} - {self._format_status(record)}")

    def _print_issues(self, records: List[CheckRecord]) -> None:
        """Phase 2: print a summary of all failures that meet min_severity."""
        issues = [
            r for r in records if r.failed and r.outcome.severity >= self.min_severity
        ]
        if not issues:
            return

        print()
        print(_DIVIDER)
        print()

        for record in issues:
            outcome: Failed = record.outcome
            color = _SEV_COLOR[outcome.severity]
            label = _SEV_LABEL[outcome.severity]
            print(f"- {color}[{label}]{Style.RESET_ALL} {outcome.msg}")
            if outcome.traceback:
                print(f"    {Fore.RED}Traceback:{Style.RESET_ALL}")
                for line in outcome.traceback.splitlines():
                    print(f"        {line}")

    @staticmethod
    def _format_status(record: CheckRecord) -> str:
        """Return a coloured status string for the collection-log line."""
        outcome = record.outcome
        if isinstance(outcome, Skipped):
            return f"{Fore.WHITE}skipped{Style.RESET_ALL}"
        if isinstance(outcome, Passed):
            return f"{Fore.GREEN}{outcome.result_text or 'ok'}{Style.RESET_ALL}"
        # Failed
        color = _SEV_COLOR[outcome.severity]
        return f"{color}{outcome.result_text or 'failed'}{Style.RESET_ALL}"
