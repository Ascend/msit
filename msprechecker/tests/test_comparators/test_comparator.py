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

import json
import logging

from msprechecker.comparators import Comparator
from msprechecker.reporters.strategy import CompareErrorDisplay


def _sample_data(host: str = "host-a", extra=None) -> dict:
    data = {"sys": {"host": host}, "env": {"VAR": "val"}}
    if extra:
        data.update(extra)
    return data


def test_compare_identical_files(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    data = _sample_data()
    handler = Comparator().compare({f1: data, f2: data})
    assert handler.empty()


def test_compare_diff_exit_handler_not_empty(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    handler = Comparator().compare({f1: _sample_data("host-a"), f2: _sample_data("host-b")})
    assert not handler.empty()


def test_compare_missing_key(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    d1 = {"sys": {"host": "a", "extra": 1}}
    d2 = {"sys": {"host": "a"}}
    handler = Comparator().compare({f1: d1, f2: d2})
    assert not handler.empty()


def test_compare_three_files(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f3 = tmp_path / "c.json"
    handler = Comparator().compare(
        {
            f1: _sample_data("host-a"),
            f2: _sample_data("host-b"),
            f3: _sample_data("host-c"),
        }
    )
    assert not handler.empty()
    assert len(handler.errors) >= 1


def test_compare_npu_device_ids(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    d1 = {"cfg": {"BackendConfig": {"npuDeviceIds": [0, 1]}}}
    d2 = {"cfg": {"BackendConfig": {"npuDeviceIds": [0, 2]}}}
    handler = Comparator().compare({f1: d1, f2: d2})
    assert not handler.empty()
    for error in handler:
        json.dumps(error.values)


def test_compare_display_json_dumps_no_crash(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    handler = Comparator().compare({f1: _sample_data("x"), f2: _sample_data("y")})
    display = CompareErrorDisplay()
    display.display(handler)
    for error in handler:
        dumped = json.dumps(error.values)
        assert "PosixPath" not in dumped
        assert "host" in dumped


def test_compare_display_identical_shows_no_difference(tmp_path, caplog):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    data = _sample_data()
    handler = Comparator().compare({f1: data, f2: data})
    display = CompareErrorDisplay()
    reporter_logger = logging.getLogger("msprechecker.reporters.strategy")
    reporter_logger.addHandler(caplog.handler)
    try:
        display.display(handler)
    finally:
        reporter_logger.removeHandler(caplog.handler)
    assert any("no difference" in record.message.lower() for record in caplog.records)
