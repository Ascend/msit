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

from ..core.strategy import Ascend, Lscpu
from ..utils import get_npu_count, get_npu_type, get_pkg_version


logger = logging.getLogger(__name__)


class InfoSection(ABC):
    @abstractmethod
    def get_info(self):
        pass


class PlatformInfoSection(InfoSection):
    def get_info(self):
        return f"Platform: {platform.platform()}"


class PythonInfoSection(InfoSection):
    def __init__(self, packages):
        self.packages = packages

    def get_info(self):
        python_info = f"Python {platform.python_version()}"
        for package in self.packages:
            package_ver = get_pkg_version(package)
            if package_ver:
                python_info += f", {package}-{package_ver}"
            else:
                python_info += f", {package} [not installed]"
        return python_info


class CpuInfoSection(InfoSection):
    def get_info(self):
        data = Lscpu().execute()
        model_name = (
            data.get("model_name", "Unknown Type") if isinstance(data, dict) else None
        )
        return f"CPU: {model_name} ({os.cpu_count()} cores)"


class NpuInfoSection(InfoSection):
    def get_info(self):
        npu_type, npu_device_nums = get_npu_type()
        npu_type = npu_type.display if npu_type else "Unknown Type"
        npu_device_nums = npu_device_nums or 0
        npu_count = get_npu_count()
        return f"NPU: {npu_type} ({npu_device_nums} devices {npu_count} chips)"


class AscendInfoSection(InfoSection):
    def get_info(self):
        data = Ascend().execute()
        if not data:
            return "Ascend: not found"
        return self._format_ascend_info(data)

    def _format_ascend_info(self, data):
        prefix = "Ascend: "
        indent = " " * len(prefix)
        items = list(data.items())
        lines = []
        for idx, (comp, info) in enumerate(items):
            line_prefix = prefix if idx == 0 else indent
            lines.append(f"{line_prefix}{comp}: {self._format_version(info)}")
        return "\n".join(lines)

    def _format_version(self, info):
        comp_ver = info.get("version", "not found")
        timestamp = info.get("timestamp")
        if timestamp:
            comp_ver += f" ({timestamp})"
        commit_id = info.get("commit")
        if commit_id:
            comp_ver += f" -- {commit_id}"
        return comp_ver


class BannerPresenter:
    PYTHON_INFO_PACKAGES = ["msprechecker", "torch", "torch_npu", "transformers"]

    def __init__(self, *, sections=None):
        self.sections = sections or [
            PlatformInfoSection(),
            PythonInfoSection(self.PYTHON_INFO_PACKAGES),
            CpuInfoSection(),
            NpuInfoSection(),
            AscendInfoSection(),
        ]

    def add_section(self, section: InfoSection):
        self.sections.append(section)

    def print_banner(self):
        cols, _ = shutil.get_terminal_size()

        title = "MindStudio Prechecker Tool"
        logger.info(f" {title} ".center(cols, "="))

        for section in self.sections:
            logger.info(section.get_info())

        logger.info("-" * cols)
