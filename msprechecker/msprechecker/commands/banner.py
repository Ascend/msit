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

import logging
import os
import platform
import shutil
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from ..core.strategy import Ascend, Lscpu
from ..util import get_npu_count, get_npu_memory, get_npu_type, get_pkg_version


logger = logging.getLogger(__name__)


class InfoSection(ABC):
    @abstractmethod
    def get_info(self) -> str:
        pass


class PlatformInfoSection(InfoSection):
    def get_info(self) -> str:
        return f"Platform: {platform.platform()}"


class PythonInfoSection(InfoSection):
    def __init__(self, packages):
        self.packages = packages

    def get_info(self) -> str:
        python_info = f"Python {platform.python_version()}"
        for package in self.packages:
            package_ver = get_pkg_version(package)
            if package_ver:
                python_info += f", {package}-{package_ver}"
            else:
                python_info += f", {package} [not installed]"
        return python_info


class CpuInfoSection(InfoSection):
    def get_info(self) -> str:
        data = Lscpu().execute()
        model_name = (
            data.get("Model name", "Unknown") if isinstance(data, dict) else "Unknown"
        )
        return f"CPU: {model_name} ({os.cpu_count()} cores)"


class NpuInfoSection(InfoSection):
    def get_info(self) -> str:
        npu_type = get_npu_type()
        npu_count = get_npu_count()
        npu_memory = get_npu_memory()  # MB

        if npu_memory is None:
            return f"NPU: {npu_type.value} ({npu_count})"
        return f"NPU: {npu_type.value} ({npu_count} x {npu_memory // 1024}G)"


class AscendInfoSection(InfoSection):
    def get_info(self) -> str:
        data = Ascend().execute()
        if not data:
            return "Ascend: not found"
        return self._format_ascend_info(data)

    def _format_ascend_info(self, data) -> str:
        prefix = "Ascend: "
        indent = " " * len(prefix)
        lines = []
        for idx, (comp, info) in enumerate(data.items()):
            line_prefix = prefix if idx == 0 else indent
            lines.append(f"{line_prefix}{comp}: {self._format_version(info)}")
        return "\n".join(lines)

    def _format_version(self, info: Dict[str, str]) -> str:
        if not info:
            return "not found"

        version = None
        timestamp = None
        commit_id = None
        for key in info:
            key_lower = key.lower()
            if version is None and "version" in key_lower:
                version = info[key]
            # mindie version key name is Ascend-mindie, lower to ascend-mindie
            elif version is None and "ascend-mindie" in key_lower:
                version = info[key]
            elif commit_id is None and "commit" in key_lower:
                commit_id = info[key]
            # only matches time or timestamp, skip runtime
            elif timestamp is None and key_lower in {"time", "timestamp"}:
                timestamp = info[key]

        if version is None:
            version = "not found"

        comp_ver = f"{version} ({timestamp})" if timestamp else version

        if commit_id:
            comp_ver += f" -- {commit_id}"

        return comp_ver


class BannerPresenter:
    TITLE = "MindStudio Prechecker Tool"
    PYTHON_INFO_PACKAGES = [
        "msprechecker",
        "torch",
        "torch_npu",
        "transformers",
    ]

    def __init__(
        self,
        *,
        sections: Optional[List[InfoSection]] = None,
        python_packages: Optional[List[str]] = None,
    ):
        packages = python_packages or self.PYTHON_INFO_PACKAGES

        # detect if it's mindie 3.0.0 by verifying if it is installed
        if get_pkg_version("mindie-motor") is not None:
            packages.extend(("mindie-llm", "mindie-motor", "atb-llm"))

        self.sections = sections or [
            PlatformInfoSection(),
            PythonInfoSection(packages),
            CpuInfoSection(),
            NpuInfoSection(),
            AscendInfoSection(),
        ]

    def add_section(self, section: InfoSection):
        self.sections.append(section)

    def render(self) -> str:
        """Return the full banner as a string (enables testing and logging)."""
        cols, _ = shutil.get_terminal_size()
        lines = [f" {self.TITLE} ".center(cols, "=")]
        for section in self.sections:
            lines.append(section.get_info())
        lines.append("-" * cols)
        return "\n".join(lines)

    def print_banner(self):
        print(self.render())
