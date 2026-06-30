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

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from msprechecker.utils.ascend import Framework, RankTable, RankTableParseError, parse_rank_table

# =============================================================================
# Tests for parse_rank_table
# =============================================================================


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_given_mindie_framework_then_calls_mindie_parser(mock_resolve, mock_file):
    """Test parse_rank_table with MINDIE framework."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": "192.168.2.1", "device_id": 0, "rank_id": 0}],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)
    with patch.object(Path, "is_file", return_value=True):
        # Test the framework dispatch - should not raise for valid data
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_given_vllm_framework_then_calls_vllm_parser(mock_resolve, mock_file):
    """Test parse_rank_table with VLLM framework."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "prefill_device_list": [
            {
                "server_id": "192.168.1.1",
                "device_ip": "192.168.2.1",
                "device_id": 0,
                "cluster_id": 1,
            }
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)
    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.VLLM)
        assert isinstance(result, RankTable)


def test_parse_rank_table_given_unknown_framework_then_raises_value_error():
    """Test parse_rank_table with UNKNOWN framework."""
    with pytest.raises(ValueError, match="No rank table parser"):
        parse_rank_table(Path("/fake/path"), Framework.UNKNOWN)


def test_parse_rank_table_given_sglang_framework_then_raises_value_error():
    """Test parse_rank_table with SGLANG framework."""
    with pytest.raises(ValueError, match="No rank table parser"):
        parse_rank_table(Path("/fake/path"), Framework.SGLANG)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_valid_data(mock_resolve, mock_file):
    """Test parse_rank_table with MINDIE framework and valid data."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": "192.168.2.1", "device_id": 0, "rank_id": 0}],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)
        assert result.server_count == 1


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_with_valid_data(mock_resolve, mock_file):
    """Test parse_rank_table with VLLM framework and valid data."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "prefill_device_list": [
            {
                "server_id": "192.168.1.1",
                "device_ip": "192.168.2.1",
                "device_id": 0,
                "cluster_id": 1,
            }
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.VLLM)
        assert isinstance(result, RankTable)
        assert result.server_count == 1


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_invalid_server_id(mock_resolve, mock_file):
    """Test parse_rank_table with invalid server_id - all invalid leads to no devices."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "invalid_ip",
                "device": [{"device_ip": "192.168.2.1", "device_id": 0, "rank_id": 0}],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="No devices found in rank table"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_invalid_device_ip(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": "invalid_ip", "device_id": 0, "rank_id": 0}],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_invalid_device_id(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": "192.168.2.1", "device_id": "invalid", "rank_id": 0}],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_with_invalid_rank_id(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": "192.168.2.1", "device_id": 0, "rank_id": "invalid"}],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.MINDIE)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_with_invalid_server_id(mock_resolve, mock_file):
    """Test parse_rank_table with invalid server_id - all invalid leads to no devices."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "prefill_device_list": [
            {
                "server_id": "invalid_ip",
                "device_ip": "192.168.2.1",
                "device_id": 0,
                "cluster_id": 1,
            }
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="No devices found in rank table"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.VLLM)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_with_invalid_cluster_id(mock_resolve, mock_file):
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "prefill_device_list": [
            {
                "server_id": "192.168.1.1",
                "device_ip": "192.168.2.1",
                "device_id": 0,
                "cluster_id": "invalid",
            }
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True):
        result = parse_rank_table(Path("/fake/path"), Framework.VLLM)
        assert isinstance(result, RankTable)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_exceeds_host_limit(mock_resolve, mock_file):
    """Test parse_rank_table with exceeds host limit."""
    mock_resolve.return_value = Path("/fake/path")
    # Create data with more hosts than allowed
    data = {
        "server_list": [{"server_id": f"192.168.{i}.1", "device": []} for i in range(1001)],
        "server_count": 1001,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="Host count exceeds limit"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_exceeds_device_limit(mock_resolve, mock_file):
    """Test parse_rank_table with exceeds device limit."""
    mock_resolve.return_value = Path("/fake/path")
    # Create data with more devices than allowed per host
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": f"192.168.2.{i}", "device_id": i, "rank_id": i} for i in range(33)],
            }
        ],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="Device count for host .* exceeds limit"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_exceeds_total_limit(mock_resolve, mock_file):
    """Test parse_rank_table with exceeds total limit."""
    mock_resolve.return_value = Path("/fake/path")
    # Create data with more devices than total allowed
    data = {
        "prefill_device_list": [
            {
                "server_id": f"192.168.{i}.1",
                "device_ip": f"192.168.{i}.2",
                "device_id": i,
                "cluster_id": i + 1,
            }
            for i in range(32001)  # Exceeds _HOST_LIMIT * _DEVICE_LIMIT_PER_HOST
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="length exceeds limit"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.VLLM)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_exceeds_host_limit(mock_resolve, mock_file):
    """Test parse_rank_table with exceeds host limit."""
    mock_resolve.return_value = Path("/fake/path")
    # Create data with more hosts than allowed (using valid IPs)
    devices = []
    for i in range(1001):
        # Use valid IPs in the 10.0.x.x range
        octet2 = i // 256
        octet3 = i % 256
        devices.append(
            {
                "server_id": f"10.0.{octet2}.{octet3}",
                "device_ip": f"10.1.{octet2}.{octet3}",
                "device_id": 0,
                "cluster_id": i + 1,
            }
        )

    data = {
        "prefill_device_list": devices,
        "decode_device_list": [],
        "server_count": 1001,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match=r"Host count exceeds limit \d+"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.VLLM)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_missing_device_list(mock_resolve, mock_file):
    """Test parse_rank_table with missing device list keys."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="Expected 'prefill_device_list' and 'decode_device_list'"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.VLLM)


@patch("builtins.open", side_effect=OSError("Failed to open"))
@patch("pathlib.Path.resolve")
def test_parse_rank_table_json_load_error(mock_resolve, mock_file):
    """Test parse_rank_table with file open error."""
    mock_resolve.return_value = Path("/fake/path")
    with patch.object(Path, "is_file", return_value=True), pytest.raises(RankTableParseError):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_missing_server_list(mock_resolve, mock_file):
    """Test parse_rank_table with missing server_list in mindie format."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_count": 1,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="'server_list' not found in rank table"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_missing_server_count(mock_resolve, mock_file):
    """Test parse_rank_table with missing server_count in mindie format."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": "192.168.2.1", "device_id": 0, "rank_id": 0}],
            }
        ],
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="'server_count' not found in rank table"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_invalid_version(mock_resolve, mock_file):
    """Test parse_rank_table with invalid version in mindie format."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [
            {
                "server_id": "192.168.1.1",
                "device": [{"device_ip": "192.168.2.1", "device_id": 0, "rank_id": 0}],
            }
        ],
        "server_count": 1,
        "version": "invalid_version",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True), pytest.raises(RankTableParseError, match="Invalid version"):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_invalid_version(mock_resolve, mock_file):
    """Test parse_rank_table with invalid version in vllm format."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "prefill_device_list": [
            {
                "server_id": "192.168.1.1",
                "device_ip": "192.168.2.1",
                "device_id": 0,
                "cluster_id": 1,
            }
        ],
        "decode_device_list": [],
        "server_count": 1,
        "version": "invalid_version",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True), pytest.raises(RankTableParseError, match="Invalid version"):
        parse_rank_table(Path("/fake/path"), Framework.VLLM)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_mindie_empty_server_list(mock_resolve, mock_file):
    """Test parse_rank_table with empty server_list in mindie format."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "server_list": [],
        "server_count": 0,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="No devices found in rank table"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.MINDIE)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_vllm_empty_device_lists(mock_resolve, mock_file):
    """Test parse_rank_table with empty device lists in vllm format."""
    mock_resolve.return_value = Path("/fake/path")
    data = {
        "prefill_device_list": [],
        "decode_device_list": [],
        "server_count": 0,
        "version": "1.0",
    }
    mock_file.return_value.read.return_value = json.dumps(data)

    with (
        patch.object(Path, "is_file", return_value=True),
        pytest.raises(RankTableParseError, match="No devices found in rank table"),
    ):
        parse_rank_table(Path("/fake/path"), Framework.VLLM)


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.resolve")
def test_parse_rank_table_error_message_uses_plain_path(mock_resolve, mock_file):
    """Rank table errors should show plain paths, not PosixPath repr."""
    rank_path = Path("/fake/rank_table.json")
    mock_resolve.return_value = rank_path
    data = {"server_count": 1, "version": "1.0"}
    mock_file.return_value.read.return_value = json.dumps(data)

    with patch.object(Path, "is_file", return_value=True), pytest.raises(RankTableParseError) as exc_info:
        parse_rank_table(rank_path, Framework.MINDIE)

    msg = str(exc_info.value)
    assert "/fake/rank_table.json" in msg
    assert "PosixPath" not in msg


@patch("builtins.open", side_effect=OSError("Failed to open"))
@patch("pathlib.Path.resolve")
def test_parse_rank_table_json_load_error_uses_plain_path(mock_resolve, mock_file):
    """JSON load errors should show plain paths, not PosixPath repr."""
    rank_path = Path("/fake/rank_table.json")
    mock_resolve.return_value = rank_path
    with patch.object(Path, "is_file", return_value=True), pytest.raises(RankTableParseError) as exc_info:
        parse_rank_table(rank_path, Framework.MINDIE)

    msg = str(exc_info.value)
    assert "/fake/rank_table.json" in msg
    assert "PosixPath" not in msg
