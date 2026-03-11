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
    has_errors,
    Passed,
    Severity,
    Skipped,
)


# =============================================================================
# Tests for Severity enum
# =============================================================================


def test_severity_enum_values():
    assert Severity.INFO == 0
    assert Severity.WARNING == 1
    assert Severity.ERROR == 2


def test_severity_enum_ordering():
    assert Severity.INFO < Severity.WARNING < Severity.ERROR
    assert Severity.ERROR > Severity.WARNING
    assert Severity.WARNING >= Severity.INFO


# =============================================================================
# Tests for Passed dataclass
# =============================================================================


def test_passed_with_default_result_text():
    passed = Passed()
    assert passed.result_text == "ok"


def test_passed_with_custom_result_text():
    passed = Passed(result_text="success")
    assert passed.result_text == "success"


def test_passed_is_frozen():
    passed = Passed()
    with pytest.raises(AttributeError):
        passed.result_text = "modified"


# =============================================================================
# Tests for Skipped dataclass
# =============================================================================


def test_skipped_with_reason():
    skipped = Skipped(reason="not applicable")
    assert skipped.reason == "not applicable"


def test_skipped_is_frozen():
    skipped = Skipped(reason="test")
    with pytest.raises(AttributeError):
        skipped.reason = "modified"


# =============================================================================
# Tests for Failed dataclass
# =============================================================================


def test_failed_with_required_fields():
    failed = Failed(
        msg="Something went wrong",
        severity=Severity.ERROR,
        result_text="failed",
    )
    assert failed.msg == "Something went wrong"
    assert failed.severity == Severity.ERROR
    assert failed.result_text == "failed"
    assert failed.traceback is None


def test_failed_with_traceback():
    failed = Failed(
        msg="Error",
        severity=Severity.WARNING,
        result_text="warn",
        traceback="Traceback (most recent call last):...",
    )
    assert failed.traceback == "Traceback (most recent call last):..."


def test_failed_is_frozen():
    failed = Failed(msg="test", severity=Severity.INFO, result_text="info")
    with pytest.raises(AttributeError):
        failed.msg = "modified"


# =============================================================================
# Tests for CheckGroup dataclass
# =============================================================================


def test_check_group_creation():
    group = CheckGroup(key="system", title="System Checks")
    assert group.key == "system"
    assert group.title == "System Checks"


def test_check_group_is_frozen():
    group = CheckGroup(key="test", title="Test")
    with pytest.raises(AttributeError):
        group.key = "modified"


# =============================================================================
# Tests for Check dataclass
# =============================================================================


def test_check_creation():
    group = CheckGroup(key="system", title="System Checks")
    fn = lambda: Passed()  # noqa: E731
    check = Check(description="Test check", group=group, fn=fn)
    assert check.description == "Test check"
    assert check.group == group
    assert check.fn == fn


def test_check_fn_execution():
    group = CheckGroup(key="system", title="System Checks")
    check = Check(description="Test", group=group, fn=lambda: Passed("success"))
    result = check.fn()
    assert isinstance(result, Passed)
    assert result.result_text == "success"


# =============================================================================
# Tests for CheckRecord dataclass
# =============================================================================


def test_check_record_with_passed_outcome():
    group = CheckGroup(key="system", title="System Checks")
    check = Check(description="Test", group=group, fn=lambda: Passed())
    record = CheckRecord(check=check, outcome=Passed())
    assert record.passed is True
    assert record.skipped is False
    assert record.failed is False


def test_check_record_with_skipped_outcome():
    group = CheckGroup(key="system", title="System Checks")
    check = Check(description="Test", group=group, fn=lambda: Passed())
    record = CheckRecord(check=check, outcome=Skipped(reason="not applicable"))
    assert record.passed is False
    assert record.skipped is True
    assert record.failed is False


def test_check_record_with_failed_outcome():
    group = CheckGroup(key="system", title="System Checks")
    check = Check(description="Test", group=group, fn=lambda: Passed())
    record = CheckRecord(
        check=check,
        outcome=Failed(msg="Error", severity=Severity.ERROR, result_text="fail"),
    )
    assert record.passed is False
    assert record.skipped is False
    assert record.failed is True


def test_check_record_is_mutable():
    group = CheckGroup(key="system", title="System Checks")
    check = Check(description="Test", group=group, fn=lambda: Passed())
    record = CheckRecord(check=check, outcome=Passed())
    record.outcome = Failed(msg="Error", severity=Severity.ERROR, result_text="fail")
    assert record.failed is True


# =============================================================================
# Tests for has_errors function
# =============================================================================


def test_has_errors_with_empty_list():
    assert has_errors([]) is False


def test_has_errors_with_all_passed():
    group = CheckGroup(key="system", title="System Checks")
    records = [
        CheckRecord(
            check=Check(description="Test1", group=group, fn=lambda: Passed()),
            outcome=Passed(),
        ),
        CheckRecord(
            check=Check(description="Test2", group=group, fn=lambda: Passed()),
            outcome=Passed(),
        ),
    ]
    assert has_errors(records) is False


def test_has_errors_with_skipped():
    group = CheckGroup(key="system", title="System Checks")
    records = [
        CheckRecord(
            check=Check(description="Test", group=group, fn=lambda: Passed()),
            outcome=Passed(),
        ),
        CheckRecord(
            check=Check(description="Test2", group=group, fn=lambda: Passed()),
            outcome=Skipped(reason="not applicable"),
        ),
    ]
    assert has_errors(records) is False


def test_has_errors_with_warning_failed():
    group = CheckGroup(key="system", title="System Checks")
    records = [
        CheckRecord(
            check=Check(description="Test", group=group, fn=lambda: Passed()),
            outcome=Passed(),
        ),
        CheckRecord(
            check=Check(description="Test2", group=group, fn=lambda: Passed()),
            outcome=Failed(
                msg="Warning", severity=Severity.WARNING, result_text="warn"
            ),
        ),
    ]
    assert has_errors(records) is False


def test_has_errors_with_error_failed():
    group = CheckGroup(key="system", title="System Checks")
    records = [
        CheckRecord(
            check=Check(description="Test", group=group, fn=lambda: Passed()),
            outcome=Passed(),
        ),
        CheckRecord(
            check=Check(description="Test2", group=group, fn=lambda: Passed()),
            outcome=Failed(msg="Error", severity=Severity.ERROR, result_text="error"),
        ),
    ]
    assert has_errors(records) is True


def test_has_errors_with_multiple_error_severities():
    group = CheckGroup(key="system", title="System Checks")
    records = [
        CheckRecord(
            check=Check(description="Test1", group=group, fn=lambda: Passed()),
            outcome=Failed(
                msg="Warning", severity=Severity.WARNING, result_text="warn"
            ),
        ),
        CheckRecord(
            check=Check(description="Test2", group=group, fn=lambda: Passed()),
            outcome=Failed(msg="Error", severity=Severity.ERROR, result_text="error"),
        ),
        CheckRecord(
            check=Check(description="Test3", group=group, fn=lambda: Passed()),
            outcome=Failed(msg="Info", severity=Severity.INFO, result_text="info"),
        ),
    ]
    assert has_errors(records) is True
