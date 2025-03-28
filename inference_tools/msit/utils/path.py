# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from pathlib import Path
from shutil import disk_usage
from stat import S_IMODE, S_IRUSR, S_IWGRP, S_IWOTH, S_IWUSR, S_IXUSR

from msit.utils.constants import MsgConst, PathConst
from msit.utils.exceptions import MsitException
from msit.utils.log import logger

_MAX_PATH_LENGTH = 4096
_MAX_LAST_NAME_LENGTH = 255
_VALID_PATH_PATTERN = r"^(?!.*\.\.)[a-zA-Z0-9_./-]+$"

_MODE_READ = ["r", "rb"]
_MODE_WRITE = ["w", "wb", "a", "ab", "a+"]
_MODE_EXEC = ["e"]
_MODE = _MODE_READ + _MODE_WRITE + _MODE_EXEC
_MAX_DIR_DEPTH = 32
AUTHORITY_DIR = 0o750
AUTHORITY_FILE = 0o640
SOFT_LINK_LEVEL_IGNORE = 0
SOFT_LINK_LEVEL_WARNING = 1
SOFT_LINK_LEVEL_STRICT = 2
_SOFT_LINK_LEVELS = [SOFT_LINK_LEVEL_IGNORE, SOFT_LINK_LEVEL_WARNING, SOFT_LINK_LEVEL_STRICT]


def is_file(path: str):
    return os.path.isfile(path)


def is_dir(path: str):
    return os.path.isdir(path)


def get_basename_from_path(path: str):
    return os.path.basename(path.rstrip("/"))


def get_file_size(path: str):
    return os.path.getsize(path)


def get_name_and_ext(model_path):
    basename = get_basename_from_path(model_path)
    # Always returns (name, ext).
    return os.path.splitext(basename)


def join_path(*args, max_depth=_MAX_DIR_DEPTH):
    if not isinstance(max_depth, int) or max_depth <= 0:
        raise MsitException(MsgConst.INVALID_DATA_TYPE, "max_depth must be a positive integer.")

    def flatten(items, depth=0):
        if depth > max_depth:
            raise MsitException(MsgConst.RISK_ALERT, f"Maximum recursion depth {max_depth} exceeded")
        for item in items:
            if isinstance(item, str):
                yield item
            elif isinstance(item, (list, tuple)):
                yield from flatten(item, depth + 1)
            else:
                pass

    return os.path.join(*flatten(args))


def is_saved_model_scene(model_path):
    saved_model_pb = join_path(model_path, "saved_model.pb")
    if not is_file(saved_model_pb):
        return False
    variables_dir = join_path(model_path, "variables")
    return is_dir(variables_dir)


def convert_bytes(bytes_size: int):
    if bytes_size < 1024:
        return f"{bytes_size} Bytes"
    elif bytes_size < 1_048_576:  # 1024 * 1024
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1_073_741_824:  # 1024 * 1024 * 1024
        return f"{bytes_size / (1_048_576):.2f} MB"
    else:
        return f"{bytes_size / (1_073_741_824):.2f} GB"


class MsitPath:
    def __init__(
        self,
        path: str,
        path_type: str,
        mode: str,
        size_limitation: int = None,
        suffix: str = None,
        max_dir_depth: int = _MAX_DIR_DEPTH,
    ):
        self.path = path
        self.path_type = self._check_path_type(path_type)
        self.mode = self._check_mode(mode)
        self.size_limitation = self._check_positive_int(size_limitation) if size_limitation else None
        self.suffix = suffix
        self.max_dir_depth = self._check_positive_int(max_dir_depth)

    @staticmethod
    def _check_path_type(path_type):
        if path_type not in [PathConst.FILE, PathConst.DIR]:
            raise MsitException(
                MsgConst.INVALID_ARGU,
                f"The path type must be one of {[PathConst.FILE, PathConst.DIR]}, " f"currently: {path_type}.",
            )
        return path_type

    @staticmethod
    def _check_mode(mode):
        if mode not in _MODE:
            raise MsitException(MsgConst.INVALID_ARGU, f"Mode must be one of {_MODE}, currently: {mode}.")
        return mode

    @staticmethod
    def _check_positive_int(value):
        if not isinstance(value, int) or value <= 0:
            raise MsitException(
                MsgConst.INVALID_ARGU, f"The value must be an integer greater than 0, currently: {value}."
            )
        return value

    @staticmethod
    def _check_path_exist(path):
        if not os.path.exists(path):
            raise MsitException(MsgConst.INVALID_ARGU, f"Path not found: {path}.")

    @staticmethod
    def _check_soft_link(path, level):
        if not os.path.islink(path):
            return path
        real_path = os.path.realpath(path)
        if level not in _SOFT_LINK_LEVELS:
            raise MsitException(
                MsgConst.INVALID_ARGU, f"The validation level of symbolic links must be one of {_SOFT_LINK_LEVELS}."
            )
        if level == SOFT_LINK_LEVEL_STRICT:
            MsitException(MsgConst.RISK_ALERT, f"Path {path} is a symlink. Usage prohibited.")
        elif level == SOFT_LINK_LEVEL_WARNING:
            logger.warning(f"Found a symlink, path {path} points to {real_path}.")
        elif level == SOFT_LINK_LEVEL_IGNORE:
            pass
        return real_path

    @staticmethod
    def _check_write_permission_for_group_others(path, permission):
        if bool(permission & (S_IWGRP | S_IWOTH)):
            raise MsitException(
                MsgConst.RISK_ALERT,
                f"The path {path} is writable by group and others. "
                "Permissions for files (or directories) should not exceed 0o755 (rwxr-xr-x).",
            )

    @classmethod
    def _check_permission(cls, path, mode):
        path_stat = os.stat(path)
        owner_id = path_stat.st_uid
        current_uid = os.geteuid()
        if owner_id not in {current_uid, 0}:
            raise MsitException(MsgConst.RISK_ALERT, f"The owner of {path} must be root or the current user.")
        permission = S_IMODE(path_stat.st_mode)
        if current_uid == 0:
            logger.warning(f"Running as root: Skipping permission checks for {path}, but this is a potential risk.")
        else:
            cls._check_write_permission_for_group_others(path, permission)
            if mode in _MODE_READ and not (permission & S_IRUSR):
                raise MsitException(
                    MsgConst.NO_PERMISSION, f"The current user is not authorized to read the path: {path}."
                )
            if mode in _MODE_WRITE and not (permission & S_IWUSR):
                raise MsitException(
                    MsgConst.NO_PERMISSION, f"The current user is not authorized to write the path: {path}."
                )
            if mode in _MODE_EXEC and not (permission & S_IXUSR):
                raise MsitException(
                    MsgConst.NO_PERMISSION, f"The current user is not authorized to execute the path: {path}."
                )

    def check(self, path_exist=True, soft_link_level=SOFT_LINK_LEVEL_STRICT):
        self.path = os.path.abspath(os.path.normpath(self.path))
        if self.mode in _MODE_WRITE and not path_exist:
            parent_dir = os.path.abspath(join_path(self.path, os.pardir))
            self._check_path_exist(parent_dir)
            parent_dir = self._check_soft_link(parent_dir, soft_link_level)
            if not is_dir(parent_dir):
                raise MsitException(MsgConst.INVALID_ARGU, f"The parent directory {parent_dir} is not valid.")
            self._check_special_chars()
            self._check_path_length()
            self._check_permission(parent_dir, self.mode)
        else:
            self._check_path_exist(self.path)
            self.path = self._check_soft_link(self.path, soft_link_level)
            self._check_special_chars()
            self._check_path_length()
            if self.path_type == PathConst.FILE:
                if not is_file(self.path):
                    raise MsitException(MsgConst.INVALID_ARGU, f"The path {self.path} is not a file.")
                self._check_file_suffix()
                self._check_file_size()
            elif self.path_type == PathConst.DIR:
                if not is_dir(self.path):
                    raise MsitException(MsgConst.INVALID_ARGU, f"The path {self.path} is not a directory.")
                self._check_dir_size()
            self._check_permission(self.path, self.mode)
        if self.path_type == PathConst.DIR and not self.path.endswith("/"):
            self.path += "/"
        return self.path

    def _check_special_chars(self):
        if not re.match(_VALID_PATH_PATTERN, self.path):
            raise MsitException(MsgConst.INVALID_ARGU, f"Path {self.path} contains special characters.")

    def _check_path_length(self):
        if len(self.path) > _MAX_PATH_LENGTH:
            raise MsitException(
                MsgConst.RISK_ALERT, f"Current path length ({len(self.path)}) exceeds the limit ({_MAX_PATH_LENGTH})."
            )
        dir_depth = 0
        for dir_name in self.path.split("/"):
            dir_depth += 1
            if dir_depth > _MAX_DIR_DEPTH:
                raise MsitException(MsgConst.RISK_ALERT, f"Exceeded max directory depth ({_MAX_DIR_DEPTH}).")
            if len(dir_name) > _MAX_LAST_NAME_LENGTH:
                raise MsitException(
                    MsgConst.RISK_ALERT,
                    f"Current {self.path_type} length ({len(dir_name)}) exceeds the limit ({_MAX_LAST_NAME_LENGTH}).",
                )

    def _check_file_suffix(self):
        if self.suffix and not self.path.endswith(self.suffix):
            raise MsitException(MsgConst.INVALID_ARGU, f"{self.path} is not a {self.suffix} file.")

    def _check_file_size(self):
        if self.size_limitation and os.path.getsize(self.path) > self.size_limitation:
            raise MsitException(
                MsgConst.RISK_ALERT, f"File size exceeds the limit ({convert_bytes(self.size_limitation)})."
            )

    def _check_dir_size(self):
        if self.size_limitation and get_dir_size(self.path, self.max_dir_depth) > self.size_limitation:
            raise MsitException(
                MsgConst.RISK_ALERT, f"Directory size exceeds the limit ({convert_bytes(self.size_limitation)})."
            )


def get_dir_size(dir_path, max_dir_depth=_MAX_DIR_DEPTH):
    total_size = 0
    for root, _, files in os.walk(dir_path):
        # fmt: off
        current_depth = root[len(dir_path):].count(os.sep)
        # fmt: on
        if current_depth > max_dir_depth:
            raise MsitException(
                MsgConst.RISK_ALERT,
                f"Calculated size of {dir_path}, but exceeded max depth ({max_dir_depth}). Current size: {total_size}.",
            )
        for file_name in files:
            total_size += os.path.getsize(join_path(root, file_name))
    return total_size


def make_dir(dir_path):
    dir_path = MsitPath(dir_path, PathConst.DIR, "w").check(path_exist=False)
    new_dir = Path(dir_path)
    try:
        new_dir.mkdir(mode=AUTHORITY_DIR, exist_ok=True, parents=False)
    except OSError as e:
        raise MsitException(
            MsgConst.IO_FAILURE,
            f"Failed to create {dir_path}, please Check if the parent directory of the current "
            f"path exists, and verify permissions or disk space.",
        ) from e


def change_permission(path, permission):
    if not os.path.exists(path) or os.path.islink(path):
        return
    try:
        os.chmod(path, permission)
    except PermissionError as e:
        raise MsitException(MsgConst.NO_PERMISSION, f"Failed to set permissions ({permission}) for {path}.") from e


def is_enough_disk_space(path, required_space):
    return disk_usage(path).free < required_space


class DirSafeHandler:
    @staticmethod
    def ensure_dir_exists(path: str):
        if not is_dir(path):
            make_dir(path)

    @staticmethod
    def get_or_raise(path: str, error_msg: str):
        if path:
            return path
        else:
            raise MsitException(MsgConst.PATH_NOT_FOUND, error_msg)

    @staticmethod
    def join_and_create(parent: str, child: str):
        path = join_path(parent, child)
        DirSafeHandler.ensure_dir_exists(path)
        return path
