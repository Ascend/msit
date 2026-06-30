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
import os
from pathlib import Path

import pytest
from msprechecker.utils.path_io import (
    as_arg_type,
    existing_dir,
    has_suffix,
    is_file,
    iter_regular_files,
    normalize_user_path,
    readable_file,
    to_user_path,
    validate_trusted_executable,
)

_NON_ROOT_UID = 1000


def _make_executable(path: Path, *, mode: int = 0o550, uid: int | None = None) -> None:
    path.write_bytes(b"")
    path.chmod(mode)
    if uid is not None:
        try:
            os.chown(path, uid, -1)
        except PermissionError:
            pytest.skip(f"cannot chown to uid {uid}")


def test_validate_trusted_executable_current_user(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: _NON_ROOT_UID)
    exe = tmp_path / "tool"
    _make_executable(exe, mode=0o550, uid=_NON_ROOT_UID)
    result = validate_trusted_executable(exe)
    assert result == exe.resolve()


def test_validate_trusted_executable_root_owned(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: _NON_ROOT_UID)
    exe = tmp_path / "tool"
    _make_executable(exe, mode=0o550, uid=_NON_ROOT_UID)
    try:
        os.chown(exe, 0, -1)
    except PermissionError:
        pytest.skip("cannot chown to root")
    result = validate_trusted_executable(exe)
    assert result == exe.resolve()


def test_validate_trusted_executable_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: _NON_ROOT_UID)
    real = tmp_path / "real"
    _make_executable(real, mode=0o550, uid=_NON_ROOT_UID)
    link = tmp_path / "link"
    link.symlink_to(real)
    assert validate_trusted_executable(link) is None


def test_validate_trusted_executable_rejects_symlink_in_path(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: _NON_ROOT_UID)
    realdir = tmp_path / "realdir"
    realdir.mkdir()
    exe = realdir / "tool"
    _make_executable(exe, mode=0o550, uid=_NON_ROOT_UID)
    linkdir = tmp_path / "linkdir"
    linkdir.symlink_to(realdir)
    assert validate_trusted_executable(linkdir / "tool") is None


def test_validate_trusted_executable_rejects_group_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: _NON_ROOT_UID)
    exe = tmp_path / "tool"
    _make_executable(exe, mode=0o770, uid=_NON_ROOT_UID)
    assert validate_trusted_executable(exe) is None


def test_validate_trusted_executable_rejects_other_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: _NON_ROOT_UID)
    exe = tmp_path / "tool"
    _make_executable(exe, mode=0o702, uid=_NON_ROOT_UID)
    assert validate_trusted_executable(exe) is None


def test_validate_trusted_executable_rejects_oversized(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: _NON_ROOT_UID)
    monkeypatch.setattr(
        "msprechecker.utils.path_io.TRUSTED_EXEC_MAX_BYTES",
        16,
    )
    exe = tmp_path / "tool"
    _make_executable(exe, mode=0o640, uid=_NON_ROOT_UID)
    exe.write_bytes(b"\0" * 16)
    exe.chmod(0o550)
    assert validate_trusted_executable(exe) is None


def test_validate_trusted_executable_root_euid_bypass(tmp_path, monkeypatch):
    exe = tmp_path / "tool"
    _make_executable(exe, mode=0o550)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    result = validate_trusted_executable(exe)
    assert result == exe.resolve()


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


def test_to_user_path_str():
    assert to_user_path("/foo/bar") == "/foo/bar"


def test_to_user_path_path():
    assert to_user_path(Path("/foo/bar")) == "/foo/bar"


def test_readable_file_error_no_posixpath(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        readable_file(str(missing))
    assert "PosixPath" not in str(exc.value)


def test_has_suffix_error_no_posixpath(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}", encoding="utf-8")
    custom = as_arg_type(is_file, has_suffix(".txt"))
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        custom(str(f))
    assert "PosixPath" not in str(exc.value)


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
