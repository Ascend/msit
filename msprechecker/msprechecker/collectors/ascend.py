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

import os
from dataclasses import dataclass, field

from msprechecker.utils.path_io import to_user_path

from .base import BaseCollector


@dataclass
class VersionComponent:
    name: str
    env_var: str = ""
    default_home: str = ""
    version_file: str = ""
    ver_keys: tuple = field(default_factory=tuple)
    ts_key: str = ""
    commit_key: str = ""

    def get_version_file(self):
        home = os.environ.get(self.env_var, "") if self.env_var else ""
        base_path = home or self.default_home
        if self.version_file.startswith("/"):
            return os.path.normpath(self.version_file)
        return os.path.normpath(os.path.join(base_path, self.version_file))


class AscendCollector(BaseCollector):
    COMPONENTS = [
        VersionComponent(name="driver", version_file="/usr/local/Ascend/driver/version.info", ver_keys=("version",)),
        VersionComponent(
            name="toolkit",
            env_var="ASCEND_TOOLKIT_HOME",
            default_home="/usr/local/Ascend/ascend-toolkit/latest/",
            version_file="toolkit/version.info",
            ver_keys=("version", "version_dir"),
            ts_key="timestamp",
        ),
        VersionComponent(
            name="opp_kernel",
            env_var="ASCEND_TOOLKIT_HOME",
            default_home="/usr/local/Ascend/ascend-toolkit/latest/",
            version_file="opp_kernel/version.info",
            ver_keys=("version", "version_dir"),
            ts_key="timestamp",
        ),
        VersionComponent(
            name="mindstudio_toolkit",
            env_var="ASCEND_TOOLKIT_HOME",
            default_home="/usr/local/Ascend/ascend-toolkit/latest/",
            version_file="mindstudio-toolkit/version.info",
            ver_keys=("version",),
        ),
        VersionComponent(
            name="atb",
            env_var="ATB_HOME_PATH",
            default_home="/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0",
            version_file="../../version.info",
            ver_keys=("ascend-cann-atb version",),
            commit_key="commit id",
        ),
        VersionComponent(
            name="mindie",
            env_var="MINDIE_LLM_HOME_PATH",
            default_home="/usr/local/Ascend/mindie/latest/mindie-llm",
            version_file="../version.info",
            ver_keys=("ascend-mindie",),
            ts_key="timestamp",
        ),
        VersionComponent(
            name="atb-models",
            env_var="ATB_SPEED_HOME_PATH",
            default_home="/usr/local/Ascend/atb-models",
            version_file="version.info",
            ver_keys=("atb-models version",),
            ts_key="time",
            commit_key="commit id",
        ),
    ]

    def __init__(self, error_handler=None):
        super().__init__(error_handler)
        self.error_handler.type = "ascend"

    @staticmethod
    def _parse_version_file(file_path, *, ver_keys, ts_key="", commit_key=""):
        result = {}

        if not os.path.isfile(file_path):
            return result

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                AscendCollector._parse_line(line, result, ver_keys, ts_key, commit_key)

        return result

    @staticmethod
    def _parse_line(line, result, ver_keys, ts_key, commit_key):
        line = line.strip()
        if not line:
            return
        parts = line.split("=", 1) if "=" in line else line.split(":", 1)
        if len(parts) != 2:
            return
        key, value = parts[0].strip().lower(), parts[1].strip()
        for ver_key in ver_keys:
            if ver_key == key:
                if "version" not in result:
                    result["version"] = value
                else:
                    result["version"] += f" ({value})"
        if ts_key and ts_key == key:
            result["timestamp"] = value
        if commit_key and commit_key == key:
            result["commit"] = value

    def _collect_data(self):
        results = {}
        for comp in self.COMPONENTS:
            file_path = comp.get_version_file()
            result = self._get_component_version(comp, file_path)
            results[comp.name] = result
        return results

    def _get_component_version(self, comp, file_path):
        result = {}
        try:
            result = self._parse_version_file(
                file_path, ver_keys=comp.ver_keys, ts_key=comp.ts_key, commit_key=comp.commit_key
            )
        except Exception as e:
            self.error_handler.add_error(
                reason=str(e),
                filename=__file__,
                function="_get_component_version",
                lineno=145,
                what=f"尝试处理文件失败：{to_user_path(file_path)}",
            )
        return result
