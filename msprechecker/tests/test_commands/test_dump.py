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

# pylint: disable=duplicate-code

import argparse
from unittest.mock import MagicMock


from msprechecker.commands.dump import Dump
from msprechecker.util import Framework


def test_dump_success_message_plain_path(tmp_path, monkeypatch, capsys):
    args = argparse.Namespace(
        configs=None,
        weight_dir=None,
        rank_table_path=None,
        output_path=tmp_path / "out" / "dump.json",
    )
    monkeypatch.setattr("msprechecker.commands.dump.detect_framework", lambda: Framework.MINDIE)
    monkeypatch.setattr("msprechecker.commands.dump.is_in_container", lambda: False)

    mock_collector = MagicMock()
    mock_collector.collect.return_value = {"sys": {}}
    monkeypatch.setattr("msprechecker.commands.dump.Dump._build_collector", lambda *a, **k: mock_collector)

    assert Dump.execute(args) == 0
    captured = capsys.readouterr()
    expected = str(args.output_path)
    assert expected in captured.out
    assert "PosixPath" not in captured.out
