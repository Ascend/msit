# -*- coding: utf-8 -*-
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

import re
import os
import stat

from .base import BasePathConstraint


class IsNameTooLong(BasePathConstraint):
    def __init__(self, *, description="long name"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        try:
            os.stat(path)
        except OSError as e:
            if getattr(e, 'winerror', None) == 206:
                return True
            if getattr(e, 'errno', None) == 36:
                return True
            if "too long" in str(e).lower():
                return True

        return False


class Exists(BasePathConstraint):
    def __init__(self, *, description="an existing path"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        return self._get_path_stat(path) is not None


class IsFile(BasePathConstraint):
    def __init__(self, *, description="a regular file"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        stat_result = self._get_path_stat(path)
        return stat_result is not None and stat.S_ISREG(stat_result.st_mode)


class IsDir(BasePathConstraint):
    def __init__(self, *, description="a directory"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        stat_result = self._get_path_stat(path)
        return stat_result is not None and stat.S_ISDIR(stat_result.st_mode)


class HasSoftLink(BasePathConstraint):
    """Check if path contains a soft link; if path does not exist, raise FileNotFoundError."""
    def __init__(self, *, description="a soft link"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        norm_path = os.path.normpath(os.path.abspath(path))
        components = norm_path.split(os.sep)

        current = os.path.sep
        for part in components[1:]:
            if not part:
                continue
            current = os.path.join(current, part)
            current_stat = self._get_path_stat(current)
            if current_stat is None:
                raise FileNotFoundError(f"Path component {current!r} does not exist.")
            if stat.S_ISLNK(current_stat.st_mode):
                return True

        final_stat = self._get_path_stat(norm_path)
        if final_stat is None:
            raise FileNotFoundError(f"Path {norm_path!r} does not exist.")

        return stat.S_ISLNK(final_stat.st_mode)


class IsReadable(BasePathConstraint):
    def __init__(self, *, description="a readable path"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        return os.access(path, os.R_OK)


class IsWritable(BasePathConstraint):
    def __init__(self, *, description="a writable path"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        return os.access(path, os.W_OK)


class HasWritableParentDir(BasePathConstraint):
    def __init__(self, *, description="a writable parent directory"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        path = os.path.abspath(path)
        dir_name = os.path.dirname(path)
        return os.access(dir_name, os.W_OK)


class IsExecutable(BasePathConstraint):
    def __init__(self, *, description="an executable path"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        return os.access(path, os.X_OK)


class IsWritableToGroupOrOthers(BasePathConstraint):
    def __init__(self, *, description="writable to group or others"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        stat_result = self._get_path_stat(path)
        if stat_result is None:
            raise FileNotFoundError(f"The path {path!r} does not exist or cannot be accessed.")

        return bool(stat_result.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


class IsConsistentToCurrentUser(BasePathConstraint):
    def __init__(self, *, description="consistent with the current user"):
        super().__init__(description=description)

    def _is_satisfied_by(self, path):
        stat_result = self._get_path_stat(path)
        if stat_result is None:
            raise FileNotFoundError(f"The path {path!r} does not exist or cannot be accessed.")

        return stat_result.st_uid == os.geteuid() or stat_result.st_uid == 0 # inconsistent only with other normal user


class IsSizeReasonable(BasePathConstraint):
    def __init__(self, *, size_limit=None, require_confirm=True, description="reasonable on its size"):
        super().__init__(description=description)
        self.size_limit = size_limit
        self.require_confirm = require_confirm

    def _is_satisfied_by(self, path):
        stat_result = self._get_path_stat(path)
        if stat_result is None:
            raise FileNotFoundError(f"The path {path!r} does not exist or cannot be accessed.")

        if not stat.S_ISREG(stat_result.st_mode):
            raise OSError(f"The path {path!r} is not a regular file.")

        return self._check_size(path, stat_result.st_size)

    def _check_size(self, path, file_size):
        if self.size_limit is None:
            from ..utils.constants import EXT_SIZE_MAPPING

            ext = os.path.splitext(path)[1]
            self.size_limit = EXT_SIZE_MAPPING.get(ext, max(EXT_SIZE_MAPPING.values()))

        is_reasonable = file_size < self.size_limit

        if not is_reasonable and self.require_confirm:
            return self._prompt_user_confirmation(path, file_size)

        return is_reasonable

    def _prompt_user_confirmation(self, path, file_size):
        prompt = (
            f"The file {path!r} is larger than expected ({file_size} Bytes). This could be a potential security threat."
            " Attempting to read such a file could potentially impact system performance.\n"
            "Please confirm your awareness of the risks associated with this action (y/n): "
        )
        confirm_pattern = re.compile(r'y(?:es)?', re.IGNORECASE)

        try:
            user_action = input(prompt)
        except Exception:
            return False

        return bool(confirm_pattern.match(user_action))
