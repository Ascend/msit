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

from __future__ import annotations

import argparse
import os
import stat
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

PathCheck = Callable[[Path], Path]

DEFAULT_MAX_FILE_BYTES = 10 * 1024**3
TRUSTED_EXEC_MAX_BYTES = 50 * 1024**3


def normalize_user_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _path_is_file(path: Path) -> bool:
    return path.is_file()


def _path_is_dir(path: Path) -> bool:
    return path.is_dir()


def _path_access(path: Path, mode: int) -> bool:
    return os.access(path, mode)


def _path_has_suffix(path: Path, suffix: str) -> bool:
    return path.suffix == suffix


def check(predicate: Callable[[Path], bool], message: str) -> PathCheck:
    def _check(path: Path) -> Path:
        if not predicate(path):
            raise argparse.ArgumentTypeError(message.format(path=path))
        return path

    return _check


def as_arg_type(*checks: PathCheck) -> Callable[[str], Path]:
    def _parse(value: str) -> Path:
        path = normalize_user_path(value)
        for fn in checks:
            path = fn(path)
        return path

    return _parse


is_file = check(_path_is_file, "{path!r} is not a file")
is_dir = check(_path_is_dir, "{path!r} is not a directory")
is_readable = check(partial(_path_access, mode=os.R_OK), "{path!r} is not readable")

readable_file = as_arg_type(is_file, is_readable)
existing_dir = as_arg_type(is_dir)


def validate_trusted_executable(path: Path) -> Path | None:
    try:
        if os.geteuid() == 0:
            if not path.is_file():
                return None
            resolved = path.resolve()
            if not os.access(resolved, os.X_OK):
                return None
            return resolved

        accumulated = Path(path.anchor) if path.is_absolute() else Path(".")
        start = 1 if path.is_absolute() else 0
        for part in path.parts[start:]:
            if part in ("", "."):
                continue
            accumulated = accumulated / part
            if stat.S_ISLNK(accumulated.lstat().st_mode):
                return None

        final = path.lstat()
        mode = final.st_mode
        if (
            not stat.S_ISREG(mode)
            or mode & (stat.S_IWGRP | stat.S_IWOTH)
            or final.st_uid not in (0, os.geteuid())
            or final.st_size >= TRUSTED_EXEC_MAX_BYTES
        ):
            return None

        resolved = path.resolve()
        if not os.access(resolved, os.X_OK):
            return None
        return resolved
    except OSError:
        return None


def has_suffix(suffix: str) -> PathCheck:
    message = f"{{path!r}} must end with {suffix!r}"
    return check(partial(_path_has_suffix, suffix=suffix), message)


def iter_regular_files(root: Path, *, suffix: str = "", max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> Iterator[Path]:
    """Root 须为入口已 normalize 的 Path；允许 root 为软链接目录，子项不跟随软链接。"""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            if not current.is_dir():
                continue
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file() and entry.suffix == suffix and entry.stat().st_size <= max_bytes:
                    yield entry
            except OSError:
                continue
