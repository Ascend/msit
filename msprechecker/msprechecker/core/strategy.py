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

import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from msprechecker.util import get_pkg_version


logger = logging.getLogger(__name__)


class CollectStrategy(ABC):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def execute(self) -> Any:
        pass


class CollectStrategyGroup(CollectStrategy):
    def __init__(
        self,
        name: str,
        strategies: Optional[List[CollectStrategy]] = None,
    ) -> None:
        super().__init__(name)
        self._strategies: List[CollectStrategy] = []

        if strategies is not None:
            try:
                strategies = list(strategies)
            except TypeError:
                logger.error(
                    "strategies must be an iterable. Got %s instead", strategies
                )
                raise

            for strategy in strategies:
                self.add(strategy)

    def add(self, strategy: CollectStrategy) -> "CollectStrategyGroup":
        if not isinstance(strategy, CollectStrategy):
            raise TypeError("collect_strategy must be an instance of CollectStrategy")
        if any(s.name == strategy.name for s in self._strategies):
            raise ValueError(
                f"A strategy with name {strategy.name!r} already exists in this group"
            )
        self._strategies.append(strategy)
        return self

    def execute(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for strategy in self._strategies:
            try:
                results[strategy.name] = strategy.execute()
            except Exception:
                logger.exception("Strategy %r failed", strategy.name)
                results[strategy.name] = None
        return results


class Env(CollectStrategy):
    ENV_FILTERS = [
        "ASCEND",
        "MINDIE",
        "ATB_",
        "HCCL_",
        "MIES",
        "RANKTABLE",
        "GE_",
        "TORCH",
        "ACL_",
        "NPU_",
        "LCCL_",
        "LCAL_",
        "OPS",
        "INF_",
    ]

    def __init__(self, name: str = "env", ascend_only: bool = False):
        super().__init__(name)
        self._ascend_only = ascend_only

    def execute(self):
        env_items = os.environ.items()

        if self._ascend_only:
            return {
                k: v
                for k, v in env_items
                if any(item in k for item in self.ENV_FILTERS)
            }
        return dict(env_items)


class Lscpu(CollectStrategy):
    def __init__(self, name="lscpu"):
        super().__init__(name)
        self._output = None

    @staticmethod
    def _parse_output(output: str):
        if not output:
            return None

        info = {}
        for line in output.splitlines():
            if ":" not in line:
                continue

            key, value = [x.strip() for x in line.split(":", 1)]
            # Silently skip duplicate keys; first occurrence wins
            if key not in info:
                info[key] = value

        return info or None

    def execute(self):
        lscpu_path = shutil.which("lscpu")
        if not lscpu_path:
            logger.warning("lscpu command not found in system PATH")
            return None

        if self._output is None:
            try:
                self._output = subprocess.check_output(
                    [lscpu_path], stderr=subprocess.DEVNULL, text=True
                )
            except Exception:
                logger.exception("Failed to execute lscpu command:")
                return None

        return self._parse_output(self._output)


class CPUHighPerformance(CollectStrategy):
    def __init__(self, name: str = "cpu_high_performance"):
        super().__init__(name)
        self._dmidecode_output = None
        self._cpupower_output = None
        self._lshw_output = None

    @staticmethod
    def _check_via_psutil():
        """
        Last-resort check: compares current CPU frequency to the reported maximum.
        NOTE: On modern CPUs with dynamic frequency scaling (e.g. Intel Speed Shift),
        the current frequency may drop during idle even in 'performance' governor mode.
        This method can produce false negatives; treat result as advisory only.
        """
        import psutil

        cpu_freq = psutil.cpu_freq()
        if not cpu_freq:
            logger.debug("Unable to get CPU frequency information via psutil")
            return False
        return cpu_freq.current == cpu_freq.max

    def _check_via_dmidecode(self):
        dmidecode_path = shutil.which("dmidecode")
        if dmidecode_path is None:
            logger.debug("dmidecode command not found in system PATH")
            return False

        if self._dmidecode_output is None:
            cmd = shlex.split(f"{dmidecode_path} -t processor")
            try:
                self._dmidecode_output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True
                )
            except Exception:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Failed to execute dmidecode command")
                return False

        return self._parse_dmidecode_output()

    def _parse_dmidecode_output(self):
        max_pattern, cur_pattern = (
            re.compile(r"Max Speed:\s*([^\n]+)", re.IGNORECASE),
            re.compile(r"Current Speed:\s*([^\n]+)", re.IGNORECASE),
        )
        max_speeds = []
        current_speeds = []
        for line in self._dmidecode_output.splitlines():
            m_max = max_pattern.search(line)
            m_cur = cur_pattern.search(line)
            if m_max:
                max_speeds.append(m_max.group(1).strip())
            if m_cur:
                current_speeds.append(m_cur.group(1).strip())

        return bool(max_speeds and current_speeds and max_speeds == current_speeds)

    def _check_via_cpupower(self):
        cpupower_path = shutil.which("cpupower")
        if cpupower_path is None:
            logger.debug("cpupower command not found in system PATH")
            return False

        if self._cpupower_output is None:
            cmd = shlex.split(f"{cpupower_path} frequency-info")
            try:
                self._cpupower_output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True
                )
            except Exception:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Failed to execute cpupower command")
                return False

        return self._parse_cpupower_output()

    def _parse_cpupower_output(self):
        limit_pattern, cur_pattern = (
            re.compile(
                r"hardware limits:\s*[\d\.]+\s*[GMK]?Hz\s*-\s*([\d\.]+\s*[GMK]?Hz)",
                re.IGNORECASE,
            ),
            re.compile(r"current CPU frequency:\s*([\d\.]+\s*[GMK]?Hz)", re.IGNORECASE),
        )

        max_limit_match = limit_pattern.search(self._cpupower_output)
        cur_freq_match = cur_pattern.search(self._cpupower_output)

        if max_limit_match and cur_freq_match:
            max_limit = max_limit_match.group(1).strip()
            cur_freq = cur_freq_match.group(1).strip()
            return max_limit == cur_freq
        return False

    def _check_via_lshw(self):
        lshw_path = shutil.which("lshw")
        if lshw_path is None:
            logger.debug("lshw command not found in system PATH")
            return False

        if self._lshw_output is None:
            cmd = shlex.split(f"{lshw_path} -c cpu")
            try:
                self._lshw_output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True
                )
            except Exception:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Failed to execute lshw command")
                return False

        return self._parse_lshw_output()

    def _parse_lshw_output(self):
        size_pattern, capacity_pattern = (
            re.compile(r"size:\s*([^\n]+)", re.IGNORECASE),
            re.compile(r"capacity:\s*([^\n]+)", re.IGNORECASE),
        )

        sizes = []
        capacities = []
        for line in self._lshw_output.splitlines():
            m_size = size_pattern.search(line)
            m_capacity = capacity_pattern.search(line)
            if m_size:
                sizes.append(m_size.group(1).strip())
            if m_capacity:
                capacities.append(m_capacity.group(1).strip())
        return bool(sizes and capacities and sizes == capacities)

    def _check_via_scaling_governor(self):
        cpu_count = os.cpu_count()
        if cpu_count is None:
            logger.debug("Unable to determine CPU count")
            return False

        scaling_governor_pattern = (
            "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor"
        )
        for core_id in range(cpu_count):
            gov_path = scaling_governor_pattern.format(core_id)
            if not os.path.isfile(gov_path):
                logger.debug("Scaling governor file not found for CPU core %s", core_id)
                return False

            try:
                with open(gov_path, encoding="utf-8") as f:
                    if f.read().strip() != "performance":
                        logger.debug(
                            "CPU core %s scaling governor is not set to performance mode",
                            core_id,
                        )
                        return False
            except Exception:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception(
                        "Failed to read scaling governor file for CPU core %s", core_id
                    )
                return False
        return True

    def execute(self):
        # Check order: most reliable → least reliable.
        # scaling_governor: direct kernel sysfs read, most authoritative.
        # dmidecode: reads BIOS-reported speeds, reliable but requires root on some systems.
        # cpupower: userspace tool, requires cpupower package.
        # lshw: hardware lister, broad compatibility.
        # psutil (last resort): instantaneous frequency; may yield false negatives on
        #   CPUs with dynamic scaling even when governor is set to 'performance'.
        return (
            self._check_via_scaling_governor()
            or self._check_via_dmidecode()
            or self._check_via_cpupower()
            or self._check_via_lshw()
            or self._check_via_psutil()
        )


class VirtualMachine(CollectStrategy):
    def __init__(self, name: str = "virtual_machine"):
        super().__init__(name)

    def execute(self):
        cpu_info_path = "/proc/cpuinfo"

        if not os.path.isfile(cpu_info_path):
            logger.debug("/proc/cpuinfo file not found")
            return False

        try:
            with open(cpu_info_path, encoding="utf-8") as f:
                return any("hypervisor" in line for line in f)
        except Exception:
            logger.exception("Failed to read /proc/cpuinfo file")
            return False


class TransparentHugepage(CollectStrategy):
    def __init__(self, name: str = "transparent_hugepage"):
        super().__init__(name)

    def execute(self):
        transparent_hugepage_path = "/sys/kernel/mm/transparent_hugepage/enabled"

        if not os.path.isfile(transparent_hugepage_path):
            logger.debug("Transparent hugepage configuration file not found")
            return None

        try:
            with open(transparent_hugepage_path, encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            logger.exception("Failed to read transparent hugepage configuration")
            return None


class Kernel(CollectStrategy):
    def __init__(self, name: str = "kernel"):
        super().__init__(name)

    def execute(self):
        return dict(platform.uname()._asdict())


class PageSize(CollectStrategy):
    def __init__(self, name: str = "page_size"):
        super().__init__(name)

    def execute(self):
        try:
            return os.sysconf("SC_PAGESIZE")
        except Exception:
            logger.exception("Failed to get system page size")
            return None


class JeMalloc(CollectStrategy):
    def __init__(self, name: str = "jemalloc"):
        super().__init__(name)

    def _check_via_apt(self) -> bool:
        """Check if jemalloc is installed via apt."""
        apt_path = shutil.which("apt")
        if apt_path is None:
            return False

        try:
            result_apt = subprocess.run(
                [apt_path, "list", "--installed", "libjemalloc*"],
                capture_output=True,
                text=True,
                check=False,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

        return result_apt.returncode == 0 and "libjemalloc" in result_apt.stdout

    def _check_via_yum(self) -> bool:
        """Check if jemalloc is installed via yum."""
        yum_path = shutil.which("yum")
        if yum_path is None:
            return False

        try:
            result_yum = subprocess.run(
                [yum_path, "list", "installed", "jemalloc*"],
                capture_output=True,
                text=True,
                check=False,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

        return result_yum.returncode == 0 and "jemalloc" in result_yum.stdout

    def execute(self) -> bool:
        return self._check_via_apt() or self._check_via_yum()


class Sys(CollectStrategyGroup):
    def __init__(
        self,
        name="sys",
        strategies=None,
    ):
        super().__init__(
            name,
            strategies
            or [
                Lscpu(),
                CPUHighPerformance(),
                VirtualMachine(),
                TransparentHugepage(),
                Kernel(),
                PageSize(),
                JeMalloc(),
            ],
        )


class Config(CollectStrategy):
    def __init__(self, name, *, config_path):
        super().__init__(name)
        self._config_path = config_path
        self._processor = {
            ".json": self._process_json,
            ".yaml": self._process_yaml,
            ".yml": self._process_yaml,
            ".sh": self._process_shell,
        }

    def _process_json(self, content):
        logger.debug("Processing JSON configuration file: %r", self._config_path)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.exception(
                "Failed to parse JSON configuration file %r", self._config_path
            )
            return content

    def _process_yaml(self, content):
        logger.debug("Processing YAML configuration file: %s", self._config_path)
        import yaml

        try:
            if "---" in content:
                return list(yaml.safe_load_all(content))
            return yaml.safe_load(content)
        except yaml.YAMLError:
            logger.exception(
                "Failed to parse YAML configuration file %r", self._config_path
            )
            return content

    def _process_shell(self, content):
        logger.debug("Processing shell configuration file: %s", self._config_path)
        return content

    def _read_file(self) -> Optional[str]:
        if not self._config_path:
            logger.warning("Configuration path is empty or not provided")
            return None
        if not os.path.isfile(self._config_path):
            logger.warning("Configuration file %r not found", self._config_path)
            return None
        try:
            with open(self._config_path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            logger.exception("Failed to read configuration file %r", self._config_path)
            return None

    def _parse(self, content: str):
        ext = os.path.splitext(self._config_path)[-1]
        processor = self._processor.get(ext)
        if processor is None:
            logger.warning(
                "Unsupported configuration file format: %r", self._config_path
            )
            return content
        return processor(content)

    def execute(self):
        content = self._read_file()
        if content is None:
            return None
        return self._parse(content)


class _Ascend(CollectStrategy):
    """
    Base strategy using Class Attributes for defaults.
    This allows subclasses to define only what changes.
    """

    NAME = ""
    HOME_ENVIRON = ""
    DEFAULT_HOME = ""
    RELATIVE_VERSION_PATH = ""

    def __init__(
        self,
        name: str = "",
        *,
        home_environ: str = "",
        default_home: str = "",
        relative_version_path: str = "",
    ):
        """
        Args:
            name: The name of the component.
            home_environ: The environment variable that contains the home path of the component.
            default_home: The default home path of the component.
            relative_version_path: The relative path of the version file from the home path.
        """
        super().__init__(name or self.NAME)
        self._home_environ = home_environ or self.HOME_ENVIRON
        self._default_home = default_home or self.DEFAULT_HOME
        self._relative_version_path = (
            relative_version_path or self.RELATIVE_VERSION_PATH
        )

    def _resolve_home(self) -> Path:
        """Resolve the home path from the environment variable or the default home path."""
        if not self._home_environ:
            logger.debug(
                "No environment variable provided, using default home path %r",
                self._default_home,
            )
            home_path = self._default_home
        elif self._home_environ not in os.environ:
            logger.debug(
                "Environment variable %r not found, using default home path %r",
                self._home_environ,
                self._default_home,
            )
            home_path = self._default_home
        else:
            logger.debug(
                "Environment variable %r found, using value %r",
                self._home_environ,
                os.environ[self._home_environ],
            )
            home_path = os.environ[self._home_environ]

        return Path(home_path).resolve()

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
                    logger.debug("Unexpected format in line: %r", line)
                    continue
                results[parts[0].strip()] = parts[1].strip()
        return results

    def execute(self) -> Optional[Dict[str, str]]:
        home_path = self._resolve_home()
        if not home_path.is_dir():
            logger.debug("Home path %r is not a directory", home_path)
            return None

        version_path = home_path / self._relative_version_path
        if not version_path.is_file():
            logger.debug("Version file not found at: %r", version_path)
            return None

        try:
            results = self._parse_version_file(version_path)
        except OSError:
            logger.debug("Failed to read version file %r", str(version_path))
            return None

        if not results:
            logger.debug("Version file yielded no data: %r", str(version_path))
            return None

        return results


class Driver(_Ascend):
    NAME = "driver"
    RELATIVE_VERSION_PATH = "version.info"
    DEFAULT_HOME = "/usr/local/Ascend/driver"


class Toolkit(_Ascend):
    NAME = "toolkit"
    HOME_ENVIRON = "ASCEND_TOOLKIT_HOME"
    RELATIVE_VERSION_PATH = "compiler/version.info"
    DEFAULT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"


class Opp(_Ascend):
    NAME = "opp"
    HOME_ENVIRON = "ASCEND_TOOLKIT_HOME"
    RELATIVE_VERSION_PATH = "opp/version.info"
    DEFAULT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"


class TB(_Ascend):
    NAME = "atb"
    HOME_ENVIRON = "ATB_HOME_PATH"
    RELATIVE_VERSION_PATH = "../../version.info"

    def __init__(
        self,
        name: str = "",
        *,
        home_environ: str = "",
        default_home: str = "",
        relative_version_path: str = "",
    ):
        super().__init__(
            name,
            home_environ=home_environ,
            default_home=self._get_abi_path(),
            relative_version_path=relative_version_path,
        )

    @staticmethod
    def _get_abi_path() -> str:
        try:
            import torch

            abi = 1 if torch.compiled_with_cxx11_abi() else 0
        except (ImportError, AttributeError):
            abi = 0
        return "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_{}".format(abi)


class MindIE(_Ascend):
    NAME = "mindie"
    HOME_ENVIRON = "MINDIE_LLM_HOME_PATH"
    RELATIVE_VERSION_PATH = "../version.info"
    DEFAULT_HOME = "/usr/local/Ascend/mindie/latest/mindie-llm"


class TBSpeed(_Ascend):
    NAME = "atb-models"
    HOME_ENVIRON = "ATB_SPEED_HOME_PATH"
    RELATIVE_VERSION_PATH = "version.info"
    DEFAULT_HOME = "/usr/local/Ascend/atb-models"


class Ascend(CollectStrategyGroup):
    def __init__(
        self,
        name: str = "ascend",
        strategies=None,
    ):
        super().__init__(
            name,
            strategies or [Driver(), Toolkit(), Opp(), TB()],
        )
        if get_pkg_version("mindie-motor") is None: # motor is packaged since 3.0.0
            self._strategies.extend((MindIE(), TBSpeed()))
