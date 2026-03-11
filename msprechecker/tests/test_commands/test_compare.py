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

import argparse
import json
from unittest.mock import Mock, mock_open, patch

import pytest

from msprechecker.commands.compare import (
    Compare,
    ComparisonResult,
    ConfigComparator,
    ConfigFlattener,
    DiffEntry,
    DiffFormatter,
    MISSING,
    setup_compare,
)


# =============================================================================
# Tests for setup_compare
# =============================================================================


def test_setup_compare_creates_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    result = setup_compare(subparsers)
    assert result is not None


# =============================================================================
# Tests for DiffEntry
# =============================================================================


def test_diff_entry_creation():
    entry = DiffEntry(
        path=("key1", "key2"), values_by_file={"file1": "a", "file2": "b"}
    )
    assert entry.path == ("key1", "key2")
    assert entry.values_by_file == {"file1": "a", "file2": "b"}


def test_diff_entry_sorted_values():
    entry = DiffEntry(path=("key",), values_by_file={"z": 1, "a": 2, "m": 3})
    assert list(entry.values_by_file.keys()) == ["a", "m", "z"]


def test_diff_entry_path_str_simple():
    entry = DiffEntry(path=("key",), values_by_file={})
    assert entry.path_str == "key"


def test_diff_entry_path_str_nested():
    entry = DiffEntry(path=("key1", "key2"), values_by_file={})
    assert entry.path_str == "key1.key2"


def test_diff_entry_path_str_with_index():
    entry = DiffEntry(path=("list", 0, "key"), values_by_file={})
    assert entry.path_str == "list[0].key"


def test_diff_entry_has_differences_true():
    entry = DiffEntry(path=("key",), values_by_file={"f1": "a", "f2": "b"})
    assert entry.has_differences is True


def test_diff_entry_has_differences_false():
    entry = DiffEntry(path=("key",), values_by_file={"f1": "a", "f2": "a"})
    assert entry.has_differences is False


def test_diff_entry_has_differences_empty():
    entry = DiffEntry(path=("key",), values_by_file={})
    assert entry.has_differences is False


def test_diff_entry_with_missing_sentinel():
    entry = DiffEntry(path=("key",), values_by_file={"f1": "a", "f2": MISSING})
    assert entry.has_differences is True


# =============================================================================
# Tests for ComparisonResult
# =============================================================================


def test_comparison_result_creation():
    result = ComparisonResult(
        file_paths=["file1.json", "file2.json"],
        differences_by_type={"env": [], "sys": []},
    )
    assert result.file_paths == ["file1.json", "file2.json"]
    assert result.total_differences == 0
    assert result.has_differences is False


def test_comparison_result_with_differences():
    diff = DiffEntry(path=("key",), values_by_file={"f1": "a", "f2": "b"})
    result = ComparisonResult(
        file_paths=["file1.json", "file2.json"],
        differences_by_type={"env": [diff], "sys": []},
    )
    assert result.total_differences == 1
    assert result.has_differences is True


def test_comparison_result_multiple_types():
    diff1 = DiffEntry(path=("key1",), values_by_file={"f1": "a", "f2": "b"})
    diff2 = DiffEntry(path=("key2",), values_by_file={"f1": 1, "f2": 2})
    result = ComparisonResult(
        file_paths=["file1.json", "file2.json"],
        differences_by_type={"env": [diff1], "sys": [diff2]},
    )
    assert result.total_differences == 2


# =============================================================================
# Tests for ConfigFlattener
# =============================================================================


def test_flattener_simple_dict():
    data = {"a": 1, "b": 2}
    result = ConfigFlattener.flatten(data)
    assert result == {("a",): 1, ("b",): 2}


def test_flattener_nested_dict():
    data = {"a": {"b": {"c": 1}}}
    result = ConfigFlattener.flatten(data)
    assert result == {("a", "b", "c"): 1}


def test_flattener_with_list():
    data = {"a": [1, 2, 3]}
    result = ConfigFlattener.flatten(data)
    assert result == {("a", 0): 1, ("a", 1): 2, ("a", 2): 3}


def test_flattener_complex_structure():
    data = {"a": [{"b": 1}, {"c": 2}]}
    result = ConfigFlattener.flatten(data)
    assert result == {("a", 0, "b"): 1, ("a", 1, "c"): 2}


def test_flattener_leaf_keys():
    data = {"npuDeviceIds": [1, 2, 3], "other": {"nested": "value"}}
    result = ConfigFlattener.flatten(data)
    # npuDeviceIds should be treated as a leaf
    assert ("npuDeviceIds",) in result
    assert result[("npuDeviceIds",)] == [1, 2, 3]


def test_flattener_empty_dict():
    result = ConfigFlattener.flatten({})
    assert result == {}


def test_unflattener_simple():
    flat = {("a",): 1, ("b",): 2}
    result = ConfigFlattener.unflatten(flat)
    assert result == {"a": 1, "b": 2}


def test_unflattener_nested():
    flat = {("a", "b", "c"): 1}
    result = ConfigFlattener.unflatten(flat)
    assert result == {"a": {"b": {"c": 1}}}


def test_unflattener_with_list():
    flat = {("a", 0): 1, ("a", 1): 2}
    result = ConfigFlattener.unflatten(flat)
    # The implementation creates dicts for list items
    assert result["a"][0] == {0: 1}
    assert result["a"][1] == {1: 2}


def test_unflattener_empty():
    result = ConfigFlattener.unflatten({})
    assert result == {}


def test_unflattener_complex():
    flat = {("a", 0, "b"): 1, ("a", 1, "c"): 2}
    result = ConfigFlattener.unflatten(flat)
    # The implementation creates nested structure with integer keys as dict keys
    assert result["a"][0][0]["b"] == 1
    assert result["a"][1][1]["c"] == 2


# =============================================================================
# Tests for ConfigComparator
# =============================================================================


def test_comparator_init_requires_two_files():
    with pytest.raises(ValueError, match="At least 2 files"):
        ConfigComparator(["file1.json"])


def test_comparator_init_success():
    comparator = ConfigComparator(["file1.json", "file2.json"])
    assert comparator.file_paths == ["file1.json", "file2.json"]


def test_comparator_load_configs():
    comparator = ConfigComparator(["file1.json", "file2.json"])
    with patch("builtins.open", mock_open(read_data='{"key": "value"}')):
        comparator.load_configs()
    assert "file1.json" in comparator.configs
    assert "file2.json" in comparator.configs
    assert comparator.configs["file1.json"] == {"key": "value"}


def test_comparator_compare_no_differences():
    comparator = ConfigComparator(["file1.json", "file2.json"])
    comparator.configs = {
        "file1.json": {"env": {"key": "value"}},
        "file2.json": {"env": {"key": "value"}},
    }
    result = comparator.compare()
    assert result.has_differences is False
    assert result.total_differences == 0


def test_comparator_compare_with_differences():
    comparator = ConfigComparator(["file1.json", "file2.json"])
    comparator.configs = {
        "file1.json": {"env": {"key": "value1"}},
        "file2.json": {"env": {"key": "value2"}},
    }
    result = comparator.compare()
    assert result.has_differences is True
    assert result.total_differences == 1


def test_comparator_compare_missing_key():
    comparator = ConfigComparator(["file1.json", "file2.json"])
    comparator.configs = {
        "file1.json": {"env": {"key": "value"}},
        "file2.json": {"env": {}},
    }
    result = comparator.compare()
    assert result.has_differences is True


# =============================================================================
# Tests for DiffFormatter
# =============================================================================


def test_formatter_format_summary():
    result = ComparisonResult(
        file_paths=["file1.json", "file2.json"],
        differences_by_type={},
    )
    formatter = DiffFormatter(result)
    summary = formatter.format_summary()
    assert "Files compared: 2" in summary
    assert "file1.json" in summary
    assert "file2.json" in summary


def test_formatter_format_differences_none():
    result = ComparisonResult(
        file_paths=["file1.json", "file2.json"],
        differences_by_type={},
    )
    formatter = DiffFormatter(result)
    output = formatter.format_differences()
    assert "ALL FILES IDENTICAL" in output


def test_formatter_format_differences_with_diffs():
    diff = DiffEntry(
        path=("key",), values_by_file={"file1.json": "a", "file2.json": "b"}
    )
    result = ComparisonResult(
        file_paths=["file1.json", "file2.json"],
        differences_by_type={"env": [diff]},
    )
    formatter = DiffFormatter(result)
    output = formatter.format_differences()
    assert "env DIFFERENCES" in output
    assert "key" in output


def test_formatter_format_diff_entry():
    diff = DiffEntry(
        path=("key",), values_by_file={"file1.json": "a", "file2.json": "b"}
    )
    result = ComparisonResult(
        file_paths=["file1.json", "file2.json"], differences_by_type={}
    )
    formatter = DiffFormatter(result)
    output = formatter._format_diff_entry(diff)
    assert "Path: key" in output
    assert "file1" in output
    assert "file2" in output


def test_formatter_shorten_text():
    result = ComparisonResult(file_paths=[], differences_by_type={})
    formatter = DiffFormatter(result, max_display_len=10)
    shortened = formatter._shorten("this is a very long text")
    assert "[truncated]" in shortened


def test_formatter_no_shorten_when_verbose():
    result = ComparisonResult(file_paths=[], differences_by_type={})
    formatter = DiffFormatter(result, verbose=True, max_display_len=10)
    text = "this is a very long text"
    shortened = formatter._shorten(text)
    assert "[truncated]" not in shortened


def test_formatter_section_header():
    result = ComparisonResult(file_paths=[], differences_by_type={})
    formatter = DiffFormatter(result)
    header = formatter._section_header("TEST")
    assert "TEST" in header
    assert "=" in header


# =============================================================================
# Tests for Compare command
# =============================================================================


def test_compare_execute_success_no_differences():
    compare = Compare()
    args = Mock(dumped_path=["file1.json", "file2.json"])
    with patch("msprechecker.commands.compare.ConfigComparator") as mock_comp:
        mock_comp.return_value.compare.return_value = Mock(
            has_differences=False,
            total_differences=0,
            file_paths=["file1.json", "file2.json"],
            differences_by_type={},
        )
        result = compare.execute(args)
        assert result == 0


def test_compare_execute_success_with_differences():
    compare = Compare()
    args = Mock(dumped_path=["file1.json", "file2.json"])
    diff = DiffEntry(path=("key",), values_by_file={"f1": "a", "f2": "b"})
    with patch("msprechecker.commands.compare.ConfigComparator") as mock_comp:
        mock_comp.return_value.compare.return_value = Mock(
            has_differences=True,
            total_differences=1,
            file_paths=["file1.json", "file2.json"],
            differences_by_type={"env": [diff]},
        )
        result = compare.execute(args)
        assert result == 1


def test_compare_execute_value_error():
    compare = Compare()
    args = Mock(dumped_path=["file1.json"])  # Only one file
    result = compare.execute(args)
    assert result == 1


def test_compare_execute_file_not_found():
    compare = Compare()
    args = Mock(dumped_path=["file1.json", "file2.json"])
    with patch("msprechecker.commands.compare.ConfigComparator") as mock_comp:
        mock_comp.return_value.load_configs.side_effect = FileNotFoundError()
        result = compare.execute(args)
        assert result == 1


def test_compare_execute_json_error():
    compare = Compare()
    args = Mock(dumped_path=["file1.json", "file2.json"])
    with patch("msprechecker.commands.compare.ConfigComparator") as mock_comp:
        mock_comp.return_value.load_configs.side_effect = json.JSONDecodeError(
            "test", "", 0
        )
        result = compare.execute(args)
        assert result == 1
