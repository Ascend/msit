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
from pathlib import Path
from typing import Optional, Dict, Any

from .base_strategy import CollectStrategyGroup, CollectStrategy
from ..utils import Utils, LOGGER


class Ascend(CollectStrategyGroup):
    def __init__(
            self,
            name: str = "ascend",
            strategies=None,
    ):
        super().__init__(
            name,
            strategies
            or [
                Driver(),
                Toolkit(),
                OppKernel(),
                TB(),
                MindIE(),
                TBSpeed(),
                NPUSmi(),
            ],
        )


class NPUSmi(CollectStrategy):
    COMPONENTS = {
        "Name": ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"],
        "HB" + "M Capacity": ["npu-smi", "info", "-t", "memory", "-i", "0"],
        "Total Count": ["npu-smi", "info", "-l"],
        "Chip Count": ["npu-smi", "info", "-l"],
        "PCI Device ID": ["npu-smi", "info", "-t", "board", "-i", "0"]
    }

    SPECIFICATIONS = {
        "910B1": {
            64: {
                "SPECIFICATION": "Atlas800T-A2-32G-400T",
                "performance": 400
            },
        },
        "910B2": {
            64: {
                "SPECIFICATION": "Atlas800T-A2-64G-376T",
                "performance": 376
            },
        },
        "910B3": {
            64: {
                "SPECIFICATION": "Atlas800T-A2-64G-313T",
                "performance": 313
            },
        },
        "910B4": {
            32: {
                "SPECIFICATION": "Atlas800I-A2-32G-280T",
                "performance": 280
            },
            64: {
                "SPECIFICATION": "Atlas800I-A2-64G-280T",
                "performance": 280
            },
        },
        "9362": {
            64: {
                "SPECIFICATION": "Atlas800T-A3-64G-560T",
                "performance": 560
            }
        },
        "9382": {
            64: {
                "SPECIFICATION": "Atlas800T-A3-64G-752T",
                "performance": 752
            }
        },
        "9392": {
            64: {
                "SPECIFICATION": "Atlas800T-A3-64G-800T",
                "performance": 800
            }
        }
    }

    def __init__(self, name: str = "npu-smi"):
        super().__init__(name)

    def _get_performance(self, data):
        chip_name = data.get("Name")
        memory = int(data.get("HB" + "M Capacity")) // 1024
        return self.SPECIFICATIONS[chip_name][memory]['performance']

    def _collect_single_data(self, cmd, keyword):
        output = Utils.collect_data(cmd, {"LD_LIBRARY_PATH":
                                              "/usr/local/Ascend/driver/lib64/common/:"
                                              "/usr/local/Ascend/driver/lib64/driver/:"})
        if output == "--":
            Utils.log_error_and_exit(
                f"Failed to execute command: {' '.join(cmd)}")
            return {}
        return {keyword: Utils.grep_lines(output, keyword)}

    def _check_npu_performance(self, current):
        target_performance = self._get_performance(self._target)
        current_performance = self._get_performance(current)
        if target_performance > current_performance:
            LOGGER.warning("Current NPU performance {}T is less than the performance {}T in dump file, "
                           "which may cause performance issue".format(current_performance, target_performance))
        elif target_performance < current_performance:
            LOGGER.warning("Current NPU performance {} is greater than the cpus {} in dump file, "
                           "which may cause performance issue when enable some features".format(current_performance,
                                                                                                target_performance))

    def _check_memory(self, current):
        target_memory = int(self._target.get("HB" + "M Capacity"))
        current_memory = int(current.get("HB" + "M Capacity"))
        if target_memory > current_memory:
            Utils.log_error_and_exit("Current memory {} is less than the memory {} in dump file, "
                                     "which may cause OOM".format(current_memory, target_memory))

    def _check_npus(self, current):
        target_npus = int(self._target.get("Total Count"))
        current_npus = int(current.get("Total Count"))
        if target_npus > current_npus:
            Utils.log_error_and_exit("Current NPU count {} is less than the total count {} in dump file"
                                     "".format(current_npus, target_npus))

    def execute(self):
        output = {}
        for key, cmd in self.COMPONENTS.items():
            output.update(self._collect_single_data(cmd, key))
        return output

    def sync(self, target_data: dict) -> Any:
        super().sync(target_data)
        current = self.execute()
        self._check_npu_performance(current)
        self._check_memory(current)


class _Ascend(CollectStrategy):
    """Base collect strategy for Ascend component version files.

    Subclass via ``_AscendComponent`` for zero-boilerplate components.
    """

    def __init__(
            self,
            name: str,
            *,
            version_path: str,
            default_home: str = "",
            home_environ: str = "",
    ):
        super().__init__(name)
        self._version_path = version_path
        self._default_home = default_home
        self._home_environ = home_environ

    def _resolve_home(self) -> str:
        """Return a validated home path from the environment, or empty string."""
        if not self._home_environ:
            return ""

        home_path = os.getenv(self._home_environ, "")
        if not home_path:
            return ""

        return home_path

    def _resolve_full_path(self, home_path: str) -> Path:
        """Combine home and version_path into an absolute, normalised path."""
        vp = Path(self._version_path)
        if vp.is_absolute():
            return vp.resolve()

        base = Path(home_path or self._default_home)
        return (base / vp).resolve()

    @staticmethod
    def _parse_version_file(path: Path) -> Dict[str, str]:
        """Parse ``KEY=VALUE`` or ``KEY: VALUE`` lines from *path*."""
        results: dict[str, str] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("=", 1) if "=" in line else line.split(":", 1)
                if len(parts) != 2:
                    LOGGER.debug("Unexpected format in line: %r", line)
                    continue
                results[parts[0].strip()] = parts[1].strip()
        return results

    def execute(self) -> Any:
        home_path = self._resolve_home()
        full_path = self._resolve_full_path(home_path)

        if not full_path.is_file():
            LOGGER.debug("Version file not found at: %r", str(full_path))
            return {}

        try:
            results = self._parse_version_file(full_path)
        except OSError:
            LOGGER.exception("Failed to read version file %r", str(full_path))
            return {}

        if not results:
            LOGGER.debug("Version file yielded no data: %r", str(full_path))
            return {}

        return results

    def sync(self, target_data: dict) -> Any:
        pass


class _AscendComponent(_Ascend):
    """Declare ``_DEFAULT_*`` class variables; get a concrete component for free.

    Example::

        class Toolkit(_AscendComponent):
            _DEFAULT_NAME = "toolkit"
            _DEFAULT_VERSION_PATH = "toolkit/version.info"
            _DEFAULT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
            _DEFAULT_ENVIRON = "ASCEND_TOOLKIT_HOME"
    """

    _DEFAULT_NAME: str
    _DEFAULT_VERSION_PATH: str
    _DEFAULT_HOME: str = ""
    _DEFAULT_ENVIRON: str = ""

    def __init__(
            self,
            name: Optional[str] = None,
            *,
            version_path: Optional[str] = None,
            default_home: Optional[str] = None,
            home_environ: Optional[str] = None,
    ):
        super().__init__(
            name if name is not None else self._DEFAULT_NAME,
            version_path=version_path
            if version_path is not None
            else self._DEFAULT_VERSION_PATH,
            default_home=default_home
            if default_home is not None
            else self._DEFAULT_HOME,
            home_environ=home_environ
            if home_environ is not None
            else self._DEFAULT_ENVIRON,
        )


class TB(_AscendComponent):
    # version.info lives two levels above the ATB CXX ABI directory.
    # Using an explicit absolute path avoids silent drift if _DEFAULT_HOME changes.
    _DEFAULT_NAME = "atb"
    _DEFAULT_VERSION_PATH = "/usr/local/Ascend/nnal/atb/latest/version.info"
    _DEFAULT_HOME = "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0"
    _DEFAULT_ENVIRON = "ATB_HOME_PATH"


class MindIE(_AscendComponent):
    # version.info lives one level above the MindIE-LLM directory.
    _DEFAULT_NAME = "mindie"
    _DEFAULT_VERSION_PATH = "/usr/local/Ascend/mindie/latest/version.info"
    _DEFAULT_HOME = "/usr/local/Ascend/mindie/latest/mindie-llm"
    _DEFAULT_ENVIRON = "MINDIE_LLM_HOME_PATH"


class TBSpeed(_AscendComponent):
    _DEFAULT_NAME = "atb-models"
    _DEFAULT_VERSION_PATH = "version.info"
    _DEFAULT_HOME = "/usr/local/Ascend/atb-models"
    _DEFAULT_ENVIRON = "ATB_SPEED_HOME_PATH"


class OppKernel(_AscendComponent):
    _DEFAULT_NAME = "opp_kernel"
    _DEFAULT_VERSION_PATH = "opp_kernel/version.info"
    _DEFAULT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
    _DEFAULT_ENVIRON = "ASCEND_TOOLKIT_HOME"


class Toolkit(_AscendComponent):
    _DEFAULT_NAME = "toolkit"
    _DEFAULT_VERSION_PATH = "toolkit/version.info"
    _DEFAULT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
    _DEFAULT_ENVIRON = "ASCEND_TOOLKIT_HOME"


class Driver(_AscendComponent):
    # Driver has no home directory concept; version_path is always absolute.
    _DEFAULT_NAME = "driver"
    _DEFAULT_VERSION_PATH = "/usr/local/Ascend/driver/version.info"
