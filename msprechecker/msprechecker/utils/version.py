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

import re
from functools import total_ordering

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version


def get_pkg_version(pkg_name):
    try:
        version_str = version(pkg_name)
    except PackageNotFoundError:
        try:
            from importlib import import_module

            version_str = import_module(pkg_name).__version__
        except Exception:
            version_str = None

    return version_str


@total_ordering
class Version:
    VERSION_REGEX = re.compile(
        r"(?P<major>\d{1,4})\.(?P<minor>\d{1,4})"
        r"(?:\.(?P<patch>\d{1,4})|\.rc(?P<rc>\d{1,4})(?:\.b(?P<beta>\d{1,4}))?|\.rc(?P<rc_only>\d{1,4}))?"
    )

    def __init__(self, version_str):
        m = self._parse_version_str(version_str)
        self._major = m.group("major")
        self._minor = m.group("minor")
        self._patch = m.group("patch")
        self._rc = m.group("rc") or m.group("rc_only")
        self._beta = m.group("beta")

    def __repr__(self):
        base = f"{self._major}.{self._minor}"
        if self._patch is not None:
            return f"{base}.{self._patch}"
        if self._rc is not None:
            return f"{base}.rc{self._rc}" + (f".b{self._beta}" if self._beta else "")
        return base

    def __eq__(self, other):
        return self.cmp_tuple() == self._other_version(other)

    def __lt__(self, other):
        return self.cmp_tuple() < self._other_version(other)

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def patch(self):
        return self._patch

    @property
    def rc(self):
        return self._rc

    @property
    def beta(self):
        return self._beta

    @staticmethod
    def _parse_version_str(version_str):
        m = Version.VERSION_REGEX.match(version_str)
        if m:
            return m
        ver = get_pkg_version(version_str)
        if ver:
            return Version._parse_version_str(ver)

        raise ValueError(f"Invalid version string: {version_str}")

    @staticmethod
    def _other_version(other):
        if isinstance(other, Version):
            return other.cmp_tuple()

        return Version(other).cmp_tuple()

    def cmp_tuple(self):
        return (
            int(self._major),
            int(self._minor),
            int(self._patch) if self._patch else 0,
            int(self._rc) if self._rc else float("inf"),
            int(self._beta) if self._beta else float("inf"),
        )
