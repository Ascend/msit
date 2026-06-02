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

import pytest
from msprechecker.utils.path_io import as_arg_type
from msprechecker.utils.path_io import existing_dir
from msprechecker.utils.path_io import has_suffix
from msprechecker.utils.path_io import is_file
from msprechecker.utils.path_io import iter_regular_files
from msprechecker.utils.path_io import normalize_user_path
from msprechecker.utils.path_io import readable_file


def test_normalize_tilde_path(tmp_path, monkeypatch):
    cfg = tmp_path / "test_cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    result = normalize_user_path("~/test_cfg.json")
    assert result == cfg.resolve()


def test_normalize_dotdot_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "sub" / "cfg.json"
    cfg.parent.mkdir()
    cfg.write_text("{}", encoding="utf-8")
    result = normalize_user_path("sub/../sub/cfg.json")
    assert result == cfg.resolve()


def test_readable_file_rejects_missing(tmp_path):
    missing = tmp_path / "nope.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=readable_file)
    with pytest.raises(SystemExit):
        parser.parse_args(["--cfg", str(missing)])


def test_readable_file_symlink_input(tmp_path):
    real_file = tmp_path / "real.json"
    real_file.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "link.json"
    symlink.symlink_to(real_file)
    assert readable_file(str(symlink)) == real_file.resolve()


def test_existing_dir_not_dir(tmp_path):
    not_dir = tmp_path / "file.txt"
    not_dir.write_text("x", encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError):
        existing_dir(str(not_dir))


def test_has_suffix_compose_ok(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    custom = as_arg_type(is_file, has_suffix(".txt"))
    assert custom(str(f)) == f.resolve()


def test_has_suffix_compose_reject(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}", encoding="utf-8")
    custom = as_arg_type(is_file, has_suffix(".txt"))
    with pytest.raises(argparse.ArgumentTypeError):
        custom(str(f))


def test_iter_regular_files_skip_symlink(tmp_path):
    root = tmp_path / "weights"
    root.mkdir()
    real_file = root / "model.safetensors"
    real_file.write_bytes(b"x")
    external = tmp_path / "external.safetensors"
    external.write_bytes(b"y")
    link_dir = root / "linked"
    link_dir.symlink_to(tmp_path / "outside")
    (tmp_path / "outside").mkdir(exist_ok=True)
    (tmp_path / "outside" / "hidden.safetensors").write_bytes(b"z")

    result = list(iter_regular_files(root, suffix=".safetensors"))
    assert result == [real_file.resolve()]


def test_iter_regular_files_root_symlink_dir(tmp_path):
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    real_file = real_dir / "w.safetensors"
    real_file.write_bytes(b"a")
    link_dir = tmp_path / "link_dir"
    link_dir.symlink_to(real_dir)

    result = list(iter_regular_files(link_dir, suffix=".safetensors"))
    assert len(result) == 1
    assert result[0].resolve() == real_file.resolve()


def test_iter_regular_files_size_at_limit(tmp_path):
    root = tmp_path / "weights"
    root.mkdir()
    max_bytes = 1024
    at_limit = root / "big.safetensors"
    at_limit.write_bytes(b"\0" * max_bytes)

    result = list(iter_regular_files(root, suffix=".safetensors", max_bytes=max_bytes))
    assert result == [at_limit.resolve()]


def test_iter_regular_files_size_over_limit(tmp_path):
    root = tmp_path / "weights"
    root.mkdir()
    max_bytes = 1024
    ok_file = root / "ok.safetensors"
    ok_file.write_bytes(b"\0" * max_bytes)
    too_big = root / "too.safetensors"
    too_big.write_bytes(b"\0" * (max_bytes + 1))

    result = list(iter_regular_files(root, suffix=".safetensors", max_bytes=max_bytes))
    assert result == [ok_file.resolve()]
