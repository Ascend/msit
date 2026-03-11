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


import pytest

from msprechecker.core.checker import (
    Check,
    CheckGroup,
    CheckRecord,
    Failed,
    Passed,
    Severity,
    Skipped,
)
from msprechecker.core.runner import PrecheckRunner


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def system_group():
    return CheckGroup(key="system", title="System Checks")


@pytest.fixture
def network_group():
    return CheckGroup(key="network", title="Network Checks")


@pytest.fixture
def sample_checks(system_group, network_group):
    return [
        Check(description="CPU check", group=system_group, fn=lambda: Passed("ok")),
        Check(description="Memory check", group=system_group, fn=lambda: Passed("ok")),
        Check(description="Ping check", group=network_group, fn=lambda: Passed("ok")),
    ]


# =============================================================================
# Tests for PrecheckRunner initialization
# =============================================================================


def test_runner_init_with_default_severity():
    runner = PrecheckRunner()
    assert runner.min_severity == Severity.INFO


def test_runner_init_with_custom_severity():
    runner = PrecheckRunner(min_severity=Severity.WARNING)
    assert runner.min_severity == Severity.WARNING


# =============================================================================
# Tests for _execute method
# =============================================================================


def test_execute_with_successful_check(system_group):
    runner = PrecheckRunner()
    check = Check(description="Test", group=system_group, fn=lambda: Passed("success"))
    outcome = runner._execute(check)
    assert isinstance(outcome, Passed)
    assert outcome.result_text == "success"


def test_execute_with_failed_check(system_group):
    runner = PrecheckRunner()
    check = Check(
        description="Test",
        group=system_group,
        fn=lambda: Failed(msg="Error", severity=Severity.ERROR, result_text="fail"),
    )
    outcome = runner._execute(check)
    assert isinstance(outcome, Failed)
    assert outcome.msg == "Error"
    assert outcome.severity == Severity.ERROR


def test_execute_with_exception(system_group):
    runner = PrecheckRunner()

    def raise_exception():
        raise ValueError("Test exception")

    check = Check(description="Test", group=system_group, fn=raise_exception)
    outcome = runner._execute(check)
    assert isinstance(outcome, Failed)
    assert outcome.severity == Severity.ERROR
    assert outcome.result_text == "error"
    assert outcome.traceback is not None
    assert "ValueError" in outcome.traceback


def test_execute_with_skipped_result(system_group):
    runner = PrecheckRunner()
    check = Check(
        description="Test",
        group=system_group,
        fn=lambda: Skipped(reason="not applicable"),
    )
    outcome = runner._execute(check)
    assert isinstance(outcome, Skipped)
    assert outcome.reason == "not applicable"


# =============================================================================
# Tests for _format_status method
# =============================================================================


def test_format_status_with_passed():
    runner = PrecheckRunner()
    check = Check(
        description="Test", group=CheckGroup("test", "Test"), fn=lambda: Passed()
    )
    record = CheckRecord(check=check, outcome=Passed("success"))
    status = runner._format_status(record)
    assert "success" in status


def test_format_status_with_skipped():
    runner = PrecheckRunner()
    check = Check(
        description="Test", group=CheckGroup("test", "Test"), fn=lambda: Passed()
    )
    record = CheckRecord(check=check, outcome=Skipped(reason="not applicable"))
    status = runner._format_status(record)
    assert "skipped" in status


def test_format_status_with_failed():
    runner = PrecheckRunner()
    check = Check(
        description="Test", group=CheckGroup("test", "Test"), fn=lambda: Passed()
    )
    record = CheckRecord(
        check=check,
        outcome=Failed(msg="Error", severity=Severity.ERROR, result_text="failed"),
    )
    status = runner._format_status(record)
    assert "failed" in status


# =============================================================================
# Tests for rendering methods
# =============================================================================


def test_print_group_header(system_group, capsys):
    runner = PrecheckRunner()
    runner._print_group_header(system_group)
    captured = capsys.readouterr()
    assert "System Checks" in captured.out


def test_print_group_done(system_group, capsys):
    runner = PrecheckRunner()
    runner._print_group_done(system_group)
    captured = capsys.readouterr()
    assert "System Checks - done" in captured.out


def test_print_item(system_group, capsys):
    runner = PrecheckRunner()
    check = Check(description="CPU check", group=system_group, fn=lambda: Passed("ok"))
    record = CheckRecord(check=check, outcome=Passed("ok"))
    runner._print_item(record)
    captured = capsys.readouterr()
    assert "CPU check" in captured.out


# =============================================================================
# Tests for run method
# =============================================================================


def test_run_with_empty_checks():
    runner = PrecheckRunner()
    records = runner.run([])
    assert records == []


def test_run_with_single_group(sample_checks):
    runner = PrecheckRunner()
    records = runner.run(sample_checks)
    assert len(records) == 3
    assert all(r.passed for r in records)


def test_run_with_mixed_results(system_group, network_group):
    runner = PrecheckRunner()
    checks = [
        Check(description="Pass check", group=system_group, fn=lambda: Passed("ok")),
        Check(
            description="Fail check",
            group=system_group,
            fn=lambda: Failed(msg="Error", severity=Severity.ERROR, result_text="fail"),
        ),
        Check(
            description="Skip check",
            group=network_group,
            fn=lambda: Skipped(reason="not applicable"),
        ),
    ]
    records = runner.run(checks)
    assert len(records) == 3
    assert records[0].passed
    assert records[1].failed
    assert records[2].skipped


def test_run_returns_check_records(sample_checks):
    runner = PrecheckRunner()
    records = runner.run(sample_checks)
    assert all(isinstance(r, CheckRecord) for r in records)


# =============================================================================
# Tests for _print_issues
# =============================================================================


def test_print_issues_with_no_failures(capsys):
    runner = PrecheckRunner()
    check = Check(
        description="Test", group=CheckGroup("test", "Test"), fn=lambda: Passed()
    )
    records = [CheckRecord(check=check, outcome=Passed())]
    runner._print_issues(records)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_issues_with_failures_below_min_severity(capsys):
    runner = PrecheckRunner(min_severity=Severity.WARNING)
    check = Check(
        description="Test", group=CheckGroup("test", "Test"), fn=lambda: Passed()
    )
    records = [
        CheckRecord(
            check=check,
            outcome=Failed(msg="Info", severity=Severity.INFO, result_text="info"),
        )
    ]
    runner._print_issues(records)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_issues_with_failures_at_min_severity(capsys):
    runner = PrecheckRunner(min_severity=Severity.WARNING)
    check = Check(
        description="Test", group=CheckGroup("test", "Test"), fn=lambda: Passed()
    )
    records = [
        CheckRecord(
            check=check,
            outcome=Failed(
                msg="Warning", severity=Severity.WARNING, result_text="warn"
            ),
        )
    ]
    runner._print_issues(records)
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_print_issues_with_traceback(capsys):
    runner = PrecheckRunner()
    check = Check(
        description="Test", group=CheckGroup("test", "Test"), fn=lambda: Passed()
    )
    records = [
        CheckRecord(
            check=check,
            outcome=Failed(
                msg="Error",
                severity=Severity.ERROR,
                result_text="error",
                traceback="Traceback: line 1\nline 2",
            ),
        )
    ]
    runner._print_issues(records)
    captured = capsys.readouterr()
    assert "Traceback" in captured.out
    assert "line 1" in captured.out
