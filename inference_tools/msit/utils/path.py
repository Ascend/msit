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
from re import match
from pathlib import Path
from stat import S_IMODE, S_IRUSR, S_IWGRP, S_IWOTH, S_IWUSR, S_IXUSR

from msit.common.log import logger
from msit.common.exceptions import MsitException
from msit.common.constants import PathConst, MsgConst
from msit.utils.toolkits import convert_bytes


class MsitPath:
    def __init__(self, path: str, path_type, mode, size_limitation=None, suffix=None, max_dir_depth=5):
        self.path = path
        self.path_type = self._check_path_type(path_type)
        self.mode = self._check_mode(mode)
        self.size_limitation = self._check_positive_int(size_limitation) if size_limitation else None
        self.suffix = suffix
        self.max_dir_depth = self._check_positive_int(max_dir_depth)

    @property
    def is_file(self):
        return os.path.isfile(self.path)

    @property
    def is_dir(self):
        return os.path.isdir(self.path)

    @staticmethod
    def _check_path_type(path_type):
        if path_type not in [PathConst.FILE, PathConst.DIR]:
            raise MsitException(MsgConst.INVALID_ARGU, \
                                f"The path type must be one of {[PathConst.FILE, PathConst.DIR]}, "
                                f"currently: {path_type}.")
        return path_type

    @staticmethod
    def _check_mode(mode):
        if mode not in PathConst.MODE:
            raise MsitException(MsgConst.INVALID_ARGU, f"Mode must be one of {PathConst.MODE}, currently: {mode}.")
        return mode

    @staticmethod
    def _check_positive_int(value):
        if not isinstance(value, int) or value <= 0:
            raise MsitException(MsgConst.INVALID_ARGU, \
                                f"The value must be an integer greater than 0, currently: {value}.")
        return value

    @staticmethod
    def _check_path_exist(path):
        if not os.path.exists(path):
            raise MsitException(MsgConst.INVALID_ARGU, f"Path not found: {path}.")

    @staticmethod
    def _check_write_permission_for_group_others(path, permission):
        if bool(permission & (S_IWGRP | S_IWOTH)): 
            raise MsitException(MsgConst.RISK_ALERT, \
                                f"The path {path} is writable by group and others. "
                                f"Permissions for files (or directories) should not exceed 0o755 (rwxr-xr-x).")

    @classmethod
    def _check_permission(cls, path, mode):
        path_stat = os.stat(path)
        owner_id = path_stat.st_uid
        group_owner_id = path_stat.st_gid
        if owner_id != os.geteuid() or group_owner_id not in os.getgroups():
            raise MsitException(MsgConst.RISK_ALERT, f"The current user isn't the owner or group owner of {path}.")
        permission = S_IMODE(path_stat.st_mode)
        cls._check_write_permission_for_group_others(path, permission)
        if mode in PathConst.MODE_READ:
            if not bool(permission & S_IRUSR):
                raise MsitException(MsgConst.NO_PERMISSION, \
                                    f"The current user is not authorized to read the path: {path}.")
        if mode in PathConst.MODE_WRITE:
            if not bool(permission & S_IWUSR):
                raise MsitException(MsgConst.NO_PERMISSION, \
                                    f"The current user is not authorized to write the path: {path}.")
        if mode == PathConst.MODE_EXEC:
            if not bool(permission & S_IXUSR):
                raise MsitException(MsgConst.NO_PERMISSION, \
                                    f"The current user is not authorized to execute the path: {path}.")

    def check(self):
        self.path = os.path.abspath(os.path.normpath(self.path))

        if self.mode in PathConst.MODE_WRITE:
            parent_dir = os.path.dirname(self.path)
            self._check_path_exist(parent_dir)
            if not os.path.isdir(parent_dir):
                raise MsitException(MsgConst.INVALID_ARGU, f"The parent directory {parent_dir} is not valid.")
            self._check_permission(parent_dir, self.mode)
        else:
            self._check_path_exist(self.path)
            if self.path_type == PathConst.FILE:
                if not self.is_file:
                    raise MsitException(MsgConst.INVALID_ARGU, f"The path {self.path} is not a file.")
                self._check_file_size()
            elif self.path_type == PathConst.DIR:
                if not self.is_dir:
                    raise MsitException(MsgConst.INVALID_ARGU, f"The path {self.path} is not a directory.")
                self._check_dir_size()
            self._check_permission(self.path, self.mode)

        self.path = self._check_soft_link()
        self._check_path_length()
        self._check_special_chars()

        if self.path_type == PathConst.FILE:
            self._check_file_suffix()
        elif self.path_type == PathConst.DIR:
            self.path += "/"
        return self.path

    def _check_soft_link(self):
        if os.path.islink(self.path):
            real_path = os.path.realpath(self.path)
            logger.info(f"Found a symlink, path {self.path} points to {real_path}.")
            return real_path
        return self.path

    def _check_path_length(self):
        if len(self.path) > PathConst.MAX_PATH_LENGTH:
            raise MsitException(MsgConst.RISK_ALERT, \
                                f"Current path length ({len(self.path)}) exceeds "
                                f"the limit ({PathConst.MAX_PATH_LENGTH}).")
        dir_depth = 0
        for dir_name in self.path.split("/"):
            dir_depth += 1
            if dir_depth > PathConst.MAX_DIR_DEPTH:
                raise MsitException(MsgConst.RISK_ALERT, f"Exceeded max directory depth ({PathConst.MAX_DIR_DEPTH}).")
            if len(dir_name) > PathConst.MAX_LAST_NAME_LENGTH:
                raise MsitException(MsgConst.RISK_ALERT, \
                                    f"Current {self.path_type} length ({len(dir_name)}) "
                                    f"exceeds the limit ({PathConst.MAX_LAST_NAME_LENGTH}).")

    def _check_special_chars(self):
        if not match(PathConst.VALID_PATH_PATTERN, self.path):
            raise MsitException(MsgConst.INVALID_ARGU, "Path contains special characters.")

    def _check_file_suffix(self):
        if self.suffix and not self.path.endswith(self.suffix):
            raise MsitException(MsgConst.INVALID_ARGU, f"{self.path} is not a {self.suffix} file.")

    def _check_file_size(self):
        if self.size_limitation and os.path.getsize(self.path) > self.size_limitation:
            raise MsitException(MsgConst.RISK_ALERT, \
                                f"File size exceeds the limit ({convert_bytes(self.size_limitation)}).")

    def _check_dir_size(self):
        if self.size_limitation and get_dir_size(self.path, self.max_dir_depth) > self.size_limitation:
            raise MsitException(MsgConst.RISK_ALERT, \
                                f"Directory size exceeds the limit ({convert_bytes(self.size_limitation)}).")


def get_dir_size(dir_path, max_dir_depth=5):
    total_size = 0
    for root, _, files in os.walk(dir_path):
        current_depth = root[len(dir_path):].count(os.sep)
        if current_depth > max_dir_depth:
            logger.warning(
                f"Calculated size of {dir_path}, but exceeded max depth ({max_dir_depth}). Current size: {total_size}."
            )
            return total_size
        for file_name in files:
            total_size += os.path.getsize(os.path.join(root, file_name))
    return total_size


def make_dir(dir_path):
    dir_path = MsitPath(dir_path, PathConst.DIR, "w").check()
    new_dir = Path(dir_path)
    try:
        new_dir.mkdir(mode=PathConst.AUTHORITY_DIR, exist_ok=True, parents=False)
    except OSError as e:
        raise MsitException(MsgConst.IO_FAILURE, \
                            f"Failed to create {dir_path}, please Check if the parent directory of the current "
                            f"path exists, and verify permissions or disk space.") from e


def change_permission(path, permission):
    if not os.path.exists(path) or os.path.islink(path):
        return
    try:
        os.chmod(path, permission)
    except PermissionError as e:
        raise MsitException(MsgConst.NO_PERMISSION, f"Failed to set permissions ({permission}) for {path}.") from e
