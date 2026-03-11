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
from unittest.mock import Mock, mock_open, patch

from msprechecker.commands.dump import Dump, setup_dump
from msprechecker.util import Framework


# =============================================================================
# Tests for setup_dump
# =============================================================================


def test_setup_dump_creates_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    result = setup_dump(subparsers)
    assert result is not None


def test_setup_dump_parser_has_required_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_dump(subparsers)
    # Should not raise when parsing with --output-path
    args = parser.parse_args(["dump", "--output-path", "/tmp/test.json"])
    assert args.output_path == "/tmp/test.json"


# =============================================================================
# Tests for Dump class
# =============================================================================


def test_dump_execute_success():
    dump = Dump()
    args = Mock(
        output_path="/tmp/test.json",
        configs=None,
        weight_dir=None,
        rank_table_path=None,
        ascend_only=False,
    )
    with patch(
        "msprechecker.commands.dump.detect_framework", return_value=Framework.MINDIE
    ), patch("msprechecker.commands.dump.Collector") as mock_collector, patch(
        "builtins.open", mock_open()
    ):
        mock_collector.return_value.collect.return_value = {"key": "value"}
        result = dump.execute(args)
        assert result == 0


def test_dump_execute_with_configs():
    dump = Dump()
    args = Mock(
        output_path="/tmp/test.json",
        configs=["cfg1:/path/cfg1.json", "cfg2:/path/cfg2.json"],
        weight_dir=None,
        rank_table_path=None,
        ascend_only=False,
    )
    with patch(
        "msprechecker.commands.dump.detect_framework", return_value=Framework.MINDIE
    ), patch("os.path.isfile", return_value=True), patch(
        "msprechecker.commands.dump.Collector"
    ) as mock_collector, patch("builtins.open", mock_open()):
        mock_collector.return_value.collect.return_value = {"key": "value"}
        result = dump.execute(args)
        assert result == 0


def test_dump_execute_with_weight_dir():
    dump = Dump()
    args = Mock(
        output_path="/tmp/test.json",
        configs=None,
        weight_dir="/path/to/weights",
        rank_table_path=None,
        ascend_only=False,
    )
    with patch(
        "msprechecker.commands.dump.detect_framework", return_value=Framework.MINDIE
    ), patch("msprechecker.commands.dump.Collector") as mock_collector, patch(
        "builtins.open", mock_open()
    ):
        mock_collector.return_value.collect.return_value = {"key": "value"}
        result = dump.execute(args)
        assert result == 0


def test_dump_execute_with_rank_table():
    dump = Dump()
    args = Mock(
        output_path="/tmp/test.json",
        configs=None,
        weight_dir=None,
        rank_table_path="/path/rank_table.json",
        ascend_only=False,
    )
    from msprechecker.util import RankTable

    with patch(
        "msprechecker.commands.dump.detect_framework", return_value=Framework.MINDIE
    ), patch("msprechecker.commands.dump.parse_rank_table") as mock_parse, patch(
        "msprechecker.commands.dump.Collector"
    ) as mock_collector, patch(
        "msprechecker.commands.dump.Network"
    ) as mock_network, patch("builtins.open", mock_open()):
        mock_parse.return_value = RankTable(
            host_to_devices={}, server_count=0, version="1.0"
        )
        mock_collector.return_value.collect.return_value = {"key": "value"}
        result = dump.execute(args)
        assert result == 0
        # Verify Network was called with the rank table
        mock_network.assert_called_once()


def test_dump_execute_write_error():
    dump = Dump()
    args = Mock(
        output_path="/tmp/test.json",
        configs=None,
        weight_dir=None,
        rank_table_path=None,
        ascend_only=False,
    )
    with patch(
        "msprechecker.commands.dump.detect_framework", return_value=Framework.MINDIE
    ), patch("msprechecker.commands.dump.Collector") as mock_collector:
        mock_collector.return_value.collect.side_effect = OSError("Write error")
        result = dump.execute(args)
        assert result == 1
