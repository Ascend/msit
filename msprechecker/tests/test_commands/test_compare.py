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
import json
import logging

from msprechecker.commands.coordinator import CompareStrategy
from msprechecker.utils.path_io import readable_file


def _write_dump(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return readable_file(str(path))


def _sample_data(host: str = "host-a") -> dict:
    return {"sys": {"host": host}, "env": {"VAR": "val"}}


def test_compare_strategy_identical_exit_0(tmp_path, capsys):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    data = _sample_data()
    _write_dump(f1, data)
    _write_dump(f2, data)
    args = argparse.Namespace(dumped_path=[f1.resolve(), f2.resolve()])
    assert CompareStrategy.execute(args) == 0
    err = capsys.readouterr().err
    assert "no difference" in err.lower()


def test_compare_strategy_diff_exit_1(tmp_path, capsys):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    _write_dump(f1, _sample_data("host-a"))
    _write_dump(f2, _sample_data("host-b"))
    args = argparse.Namespace(dumped_path=[f1.resolve(), f2.resolve()])
    assert CompareStrategy.execute(args) == 1
    err = capsys.readouterr().err
    assert "PosixPath" not in err
    assert "DIFF REPORT" in err


def test_compare_strategy_single_file_exit_1(tmp_path, caplog):
    f1 = tmp_path / "only.json"
    _write_dump(f1, _sample_data())
    args = argparse.Namespace(dumped_path=[f1.resolve()])
    with caplog.at_level(logging.INFO):
        assert CompareStrategy.execute(args) == 1


def test_compare_strategy_invalid_json_exit_1(tmp_path, caplog):
    f1 = tmp_path / "bad.json"
    f2 = tmp_path / "good.json"
    f1.write_text("{not json", encoding="utf-8")
    _write_dump(f2, _sample_data())
    args = argparse.Namespace(dumped_path=[f1.resolve(), f2.resolve()])
    with caplog.at_level(logging.ERROR):
        assert CompareStrategy.execute(args) == 1


def test_compare_strategy_three_files_exit_1(tmp_path):
    paths = [tmp_path / f"f{i}.json" for i in range(3)]
    for i, path in enumerate(paths):
        _write_dump(path, _sample_data(f"host-{i}"))
    args = argparse.Namespace(dumped_path=[p.resolve() for p in paths])
    assert CompareStrategy.execute(args) == 1
