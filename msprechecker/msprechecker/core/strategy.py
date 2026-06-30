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

from __future__ import annotations

# pylint: disable=too-many-lines,relative-beyond-top-level

import hashlib
import ipaddress
import json
import logging
import operator
import os
import platform
import re
import shlex
import shutil
import subprocess  # nosec B404
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import reduce
from pathlib import Path
from typing import Any, Optional

from ..util import get_npu_count, get_pkg_version
from ..utils.ascend import RankTable
from ..utils.path_io import to_user_path

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
        strategies: Optional[list[CollectStrategy]] = None,
    ) -> None:
        super().__init__(name)
        self._strategies: list[CollectStrategy] = []

        if strategies is not None:
            try:
                strategies = list(strategies)
            except TypeError:
                logger.error("strategies must be an iterable. Got %s instead", strategies)
                raise

            for strategy in strategies:
                self.add(strategy)

    def add(self, strategy: CollectStrategy) -> "CollectStrategyGroup":
        if not isinstance(strategy, CollectStrategy):
            raise TypeError("collect_strategy must be an instance of CollectStrategy")
        if any(s.name == strategy.name for s in self._strategies):
            raise ValueError(f"A strategy with name {strategy.name!r} already exists in this group")
        self._strategies.append(strategy)
        return self

    def execute(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
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
            return {k: v for k, v in env_items if any(item in k for item in self.ENV_FILTERS)}
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
                self._output = subprocess.check_output([lscpu_path], stderr=subprocess.DEVNULL, text=True)  # nosec B603
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
                self._dmidecode_output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)  # nosec B603
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
                self._cpupower_output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)  # nosec B603
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
                self._lshw_output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)  # nosec B603
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

        scaling_governor_pattern = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor"
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
                    logger.exception("Failed to read scaling governor file for CPU core %s", core_id)
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
            result_apt = subprocess.run(  # nosec B603
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
            result_yum = subprocess.run(  # nosec B603
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
        logger.debug("Processing JSON configuration file: %s", to_user_path(self._config_path))
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.exception("Failed to parse JSON configuration file %s", to_user_path(self._config_path))
            return content

    def _process_yaml(self, content):
        logger.debug("Processing YAML configuration file: %s", self._config_path)
        import yaml

        try:
            if "---" in content:
                return list(yaml.safe_load_all(content))
            return yaml.safe_load(content)
        except yaml.YAMLError:
            logger.exception("Failed to parse YAML configuration file %s", to_user_path(self._config_path))
            return content

    def _process_shell(self, content):
        logger.debug("Processing shell configuration file: %s", self._config_path)
        return content

    def _read_file(self) -> Optional[str]:
        if not self._config_path:
            logger.warning("Configuration path is empty or not provided")
            return None
        if not os.path.isfile(self._config_path):
            logger.warning("Configuration file %s not found", to_user_path(self._config_path))
            return None
        try:
            with open(self._config_path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            logger.exception("Failed to read configuration file %s", to_user_path(self._config_path))
            return None

    def _parse(self, content: str):
        ext = os.path.splitext(self._config_path)[-1]
        processor = self._processor.get(ext)
        if processor is None:
            logger.warning("Unsupported configuration file format: %s", to_user_path(self._config_path))
            return content
        return processor(content)

    def execute(self):
        content = self._read_file()
        if content is None:
            return None
        return self._parse(content)


class Weight(CollectStrategy):
    """Collect strategy that computes SHA-256 hashes of safetensor weight files."""

    # Matches the shard index in filenames like ``model-00003-of-00010.safetensors``
    _TENSOR_ID_RE = re.compile(r"(\d{5})-of-\d{5}")

    def __init__(
        self,
        name: str = "weight",
        *,
        weight_dir: str = "",
        tensor_suffix: str = ".safetensors",
        max_size: int = 10 * 1024**3,  # 10 GiB – skip files larger than this
        chunk_size: int = 256 * 1024**2,  # 256 MiB read buffer
        max_hash_workers: int = 4,
    ):
        super().__init__(name)
        self._weight_dir = weight_dir
        self._tensor_suffix = tensor_suffix
        self._max_size = max_size
        self._chunk_size = chunk_size
        self._max_hash_workers = max_hash_workers

    def _validate_weight_dir(self) -> bool:
        if not os.path.isdir(self._weight_dir):
            logger.warning(
                "Expected %r to be a directory. Weight strategy skipped",
                self._weight_dir,
            )
            return False
        return True

    def _is_valid_tensor_file(self, path: str) -> bool:
        if os.path.islink(path):
            logger.warning("Expected %r to be a regular file. Weight strategy skipped", path)
            return False

        if not os.path.isfile(path) or not path.endswith(self._tensor_suffix):
            return False

        file_size = os.path.getsize(path)
        if file_size > self._max_size:
            logger.warning(
                "Tensor file %r (%d bytes) exceeds max_size (%d bytes), skipping",
                path,
                file_size,
                self._max_size,
            )
            return False
        return True

    def _filter_valid_tensor_files(self) -> list[str]:
        result = []
        for filename in os.listdir(self._weight_dir):
            full_path = os.path.join(self._weight_dir, filename)
            if self._is_valid_tensor_file(full_path):
                result.append(full_path)
        return result

    def _get_tensor_id(self, tensor_file: str) -> str:
        """Return the zero-padded shard index, or the basename as fallback."""
        m = self._TENSOR_ID_RE.search(os.path.basename(tensor_file))
        return m.group(1) if m else os.path.basename(tensor_file)

    def _calculate_hash256(self, tensor_file: str) -> str:
        sha256 = hashlib.sha256()
        with open(tensor_file, "rb") as f:
            while True:
                chunk = f.read(self._chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    def _parallel_hash_calculation(self, tensor_files: list[str]) -> dict[str, Optional[str]]:
        max_workers = min(len(tensor_files), self._max_hash_workers)
        results: dict[str, Optional[str]] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(self._calculate_hash256, f): self._get_tensor_id(f) for f in tensor_files}
            for future in as_completed(future_to_id):
                tensor_id = future_to_id[future]
                try:
                    results[tensor_id] = future.result()
                except Exception:
                    logger.exception("Failed to calculate hash for tensor id %r", tensor_id)
                    results[tensor_id] = None

        return results

    def execute(self) -> Optional[dict[str, Optional[str]]]:
        if not self._validate_weight_dir():
            return None

        tensor_files = self._filter_valid_tensor_files()
        if not tensor_files:
            logger.warning("No valid tensor files found in %r", self._weight_dir)
            return None

        return self._parallel_hash_calculation(tensor_files)


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
        self._relative_version_path = relative_version_path or self.RELATIVE_VERSION_PATH

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
    def _parse_version_file(path: Path) -> dict[str, str]:
        """Parse ``KEY=VALUE`` or ``KEY: VALUE`` lines from *path*."""
        results: dict[str, str] = {}
        with open(path, encoding="utf-8") as f:
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

    def execute(self) -> Optional[dict[str, str]]:
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

    def __init__(  # pylint: disable=useless-parent-delegation
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
        except RuntimeError:
            logger.warning("Unexpected Error occurred while importing torch, default to abi 0")
            abi = 0
        return f"/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_{abi}"


class MindIE(_Ascend):
    NAME = "mindie"
    HOME_ENVIRON = "MIES_INSTALL_PATH"  # use mindie-service to locate
    RELATIVE_VERSION_PATH = "../version.info"
    DEFAULT_HOME = "/usr/local/Ascend/mindie/latest/mindie-service"


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
        if get_pkg_version("mindie-motor") is None:  # motor is packaged since 3.0.0
            self._strategies.extend((MindIE(), TBSpeed()))


class Ping(CollectStrategy):
    def __init__(self, name: str = "ping", *, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
        if not isinstance(ip, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            raise TypeError(f"IP address must be an instance of ipaddress.IPv4Address or ipaddress.IPv6Address: {ip!r}")

        super().__init__(name)
        self._ip = ip
        self._ping_path = shutil.which("ping")

    def _ping_ip(self) -> Optional[str]:
        """Ping the IP address and return the output."""
        cmd = f"{self._ping_path} -c 3 -q -W 2 {self._ip}"
        try:
            return subprocess.check_output(  # nosec B603
                shlex.split(cmd),
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )
        except Exception:
            if logger.isEnabledFor(logging.WARNING):
                logger.exception("Failed to execute ping command for IP %s", self._ip)
            return None

    def execute(self) -> Optional[str]:
        """Ping the IP address and return the output.

        Returns:
            Optional[str]: The output of the ping command, or None if the command failed.
        """
        if self._ping_path is None:
            logger.warning("ping command not found in system PATH")
            return None

        return self._ping_ip()


class HccnTool(CollectStrategy):
    """Base class for all hccn_tool-based collect strategies."""

    HCCN_TOOL_PATH = "/usr/local/Ascend/driver/tools/hccn_tool"

    def __init__(
        self,
        name: str,
        *,
        device_ids: list[int],
        max_workers: int = 8,
        timeout: float = 3,
    ):
        super().__init__(name)
        self._device_ids = device_ids
        self._max_workers = max_workers
        self._timeout = timeout
        self._bin_path = shutil.which("hccn_tool") or self.HCCN_TOOL_PATH

    def _run(self, cmd: list[str]) -> Optional[str]:
        """Execute a single hccn_tool command and return its stdout and stderr."""
        try:
            return subprocess.check_output(  # nosec B603
                cmd,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self._timeout,
            )
        except Exception:
            if logger.isEnabledFor(logging.WARNING):
                logger.exception("Failed to execute hccn_tool command: %s", cmd)
            return None

    @abstractmethod
    def execute(self) -> list[Any]:
        """Execute the hccn_tool command for each device and return a list of results."""


class _SingleOption(HccnTool):
    """
    Runs:  hccn_tool -i <device_id> <option> -g

    Returns:
        List[Optional[str]]: A list of results, one per device_id, or None if the command failed.
    """

    OPTION: str
    DEFAULT_NAME: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "OPTION", None), str):
            raise TypeError(f"{cls.__name__} must define a string class variable 'OPTION'")
        if not isinstance(getattr(cls, "DEFAULT_NAME", None), str):
            raise TypeError(f"{cls.__name__} must define a string class variable 'DEFAULT_NAME'")

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        device_ids: list[int],
        max_workers: int = 8,
        timeout: float = 3,
    ):
        super().__init__(
            name=name or self.DEFAULT_NAME,
            device_ids=device_ids,
            max_workers=max_workers,
            timeout=timeout,
        )

    def _build_cmd(self, device_id: int) -> list[str]:
        return [self._bin_path, "-i", str(device_id), self.OPTION, "-g"]

    def execute(self) -> list[Optional[str]]:
        cmds = [self._build_cmd(device_id) for device_id in self._device_ids]
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self._run, cmds))


class _OptionIp(HccnTool):
    """
    Runs:  hccn_tool -i <device_id> <option> -g address <peer_ip>
           for every (device_id, peer_ip) combination.

    Returns:
        List[Dict[str, Optional[str]]]: A list of results, one per device_id, or None if the command failed.
        — one dict per device_id; keys are peer IPs, values are raw output.
    """

    OPTION: str
    DEFAULT_NAME: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "OPTION", None), str):
            raise TypeError(f"{cls.__name__} must define a string class variable 'OPTION'")
        if not isinstance(getattr(cls, "DEFAULT_NAME", None), str):
            raise TypeError(f"{cls.__name__} must define a string class variable 'DEFAULT_NAME'")

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        device_ids: list[int],
        device_ips: list[str],
        max_workers: int = 8,
        timeout: float = 3,
    ):
        super().__init__(
            name=name or self.DEFAULT_NAME,
            device_ids=device_ids,
            max_workers=max_workers,
            timeout=timeout,
        )
        self._device_ips = device_ips

    def _build_cmd(self, device_id: int, ip: str) -> list[str]:
        return [self._bin_path, "-i", str(device_id), self.OPTION, "-g", "address", ip]

    def _probe_device(self, device_id: int) -> dict[str, Optional[str]]:
        """Run probes for all peer IPs from a single device — sequentially."""
        return {ip: self._run(self._build_cmd(device_id, ip)) for ip in self._device_ips}

    def execute(self) -> list[dict[str, Optional[str]]]:
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self._probe_device, self._device_ids))


class Vnic(_SingleOption):
    OPTION = "-vnic"
    DEFAULT_NAME = "vnic"


class Link(_SingleOption):
    OPTION = "-link"
    DEFAULT_NAME = "link"


class Tls(_SingleOption):
    OPTION = "-tls"
    DEFAULT_NAME = "tls"


class HcclPing(_OptionIp):
    OPTION = "-ping"
    DEFAULT_NAME = "hccl_ping"


class HccsPing(_OptionIp):
    OPTION = "-hccs_ping"
    DEFAULT_NAME = "hccs_ping"


class Network(CollectStrategyGroup):
    HCCS_PING_VERSION = "1.2"

    def __init__(
        self,
        name: str = "network",
        *,
        rank_table: RankTable,
        npu_count: Optional[int] = None,
    ):
        if not isinstance(rank_table, RankTable):
            raise TypeError("rank_table must be an instance of RankTable")

        npu_count = npu_count if npu_count is not None else get_npu_count()
        if npu_count == 0:
            raise ValueError("No NPU devices found in the system")

        # Collect all unique device IPs from rank table
        all_device_ips = list(
            dict.fromkeys(
                device_info.device_ip
                for device_info_list in rank_table.host_to_devices.values()
                for device_info in device_info_list
            )
        )

        if not all_device_ips:
            raise ValueError("No device IP addresses found in the rank table")

        device_ids = list(range(npu_count))

        # Determine which ping class to use based on rank table version
        ping_cls = HccsPing if rank_table.version == self.HCCS_PING_VERSION else HcclPing

        # Build strategies list
        # Each Ping needs a unique name based on the IP
        strategies = [Ping(name=f"ping_{ip}", ip=ip) for ip in all_device_ips]
        strategies.extend(
            [
                Vnic(device_ids=device_ids),
                Link(device_ids=device_ids),
                Tls(device_ids=device_ids),
                ping_cls(device_ids=device_ids, device_ips=all_device_ips),
            ]
        )

        super().__init__(name, strategies=strategies)


class Stress(CollectStrategy):
    def __init__(
        self,
        name: str,
        *,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        epochs: int = 5,
    ):
        self._torch = None
        try:
            import torch

            self._torch = torch
        except ImportError:
            logger.warning("Failed to import torch")

        for param_name, param_value in [
            ("batch_size", batch_size),
            ("seq_len", seq_len),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("epochs", epochs),
        ]:
            if not isinstance(param_value, int) or param_value <= 0:
                raise ValueError(f"{param_name} must be a positive integer, got {param_value!r}")

        super().__init__(name)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._epochs = epochs

    @property
    @abstractmethod
    def device_type(self) -> str:
        pass

    @abstractmethod
    def _get_free_memory(self, device: str) -> float:
        pass

    def _calculate_tensor_memory(self, shape: tuple[int, ...]) -> int:
        if not isinstance(shape, tuple):
            shape = (shape,)

        return reduce(operator.mul, shape, 1) * 4  # float32 = 4 bytes

    def _check_memory_for_matmul(self, device_pos: str) -> bool:
        mat_a_mem = self._calculate_tensor_memory((self._batch_size, self._seq_len, self._hidden_size))
        mat_b_mem = self._calculate_tensor_memory((self._batch_size, self._hidden_size, self._intermediate_size))
        # addbmm output shape is (seq_len, intermediate_size); unchanged
        mat_c_mem = self._calculate_tensor_memory((self._seq_len, self._intermediate_size))
        total_required = mat_a_mem + mat_b_mem + mat_c_mem

        free_memory = self._get_free_memory(device_pos)
        safety_margin = 0.2
        available_with_margin = free_memory * (1 - safety_margin)
        has_enough_mem = total_required <= available_with_margin
        logger.debug(
            "Device %s - Required memory: %d bytes, Free memory: %d bytes, Available with margin: %d bytes",
            device_pos,
            total_required,
            free_memory,
            available_with_margin,
        )

        if not has_enough_mem:
            logger.warning("Insufficient memory on device %s for matmul operation", device_pos)
            return False

        return True

    def _matmul_stress_test(self, device_id: int) -> float:
        """Run matrix-multiply stress on one device and return elapsed ms."""
        device_pos = f"{self.device_type}:{device_id}"

        if not self._check_memory_for_matmul(device_pos):
            return 0.0

        start_time = time.perf_counter()
        for _ in range(self._epochs):
            mat_a = self._torch.randn(self._batch_size, self._seq_len, self._hidden_size).to(device_pos)
            mat_b = self._torch.randn(self._batch_size, self._hidden_size, self._intermediate_size).to(device_pos)
            mat_c = self._torch.zeros(self._seq_len, self._intermediate_size).to(device_pos)
            self._torch.addbmm(mat_c, mat_a, mat_b)

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000

    def execute(self) -> Optional[dict[int, Optional[float]]]:
        """Run the stress test and return the elapsed time in milliseconds for each device.

        Returns:
            Optional[Dict[int, Optional[float]]]: A dictionary of device IDs and their elapsed time in milliseconds,
                                                  or None if torch is not available.
        """
        if not self._torch:
            logger.warning("torch is not available, skip the stress test")
            return None

        output = {}
        cpu_count = os.cpu_count() or 1
        self._torch.set_num_threads(cpu_count)
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            future_to_id = {executor.submit(self._matmul_stress_test, cpu_id): cpu_id for cpu_id in range(cpu_count)}
            for future in as_completed(future_to_id):
                cpu_id = future_to_id[future]
                try:
                    elapsed_ms = future.result()
                    logger.debug(
                        "Stress test completed on device %s:%s in %.2f ms",
                        self.device_type,
                        cpu_id,
                        elapsed_ms,
                    )
                    output[cpu_id] = elapsed_ms
                except Exception:
                    if logger.isEnabledFor(logging.WARNING):
                        logger.exception(
                            "Stress test failed on device %s:%s",
                            self.device_type,
                            cpu_id,
                        )
                    output[cpu_id] = None

        return output


class CPU(Stress):
    def __init__(
        self,
        name: str = "cpu",
        *,
        batch_size: int = 1,
        seq_len: int = 512,
        hidden_size: int = 1024,
        intermediate_size: int = 64,
        epochs: int = 5,
    ):
        super().__init__(
            name,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            epochs=epochs,
        )

    @property
    def device_type(self) -> str:
        return "cpu"

    def _get_free_memory(self, device: str) -> float:
        import psutil

        memory_available = psutil.virtual_memory().available
        logger.debug("Available CPU memory: %d bytes", memory_available)
        return memory_available


class NPU(Stress):
    def __init__(
        self,
        name: str = "npu",
        *,
        batch_size: int = 1,
        seq_len: int = 4096,
        hidden_size: int = 8192,
        intermediate_size: int = 3584,
        epochs: int = 5,
    ):
        super().__init__(
            name,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            epochs=epochs,
        )

        self._torch_npu = None
        try:
            import torch_npu

            self._torch_npu = torch_npu
        except ImportError:
            logger.warning("Failed to import torch_npu")

    @property
    def device_type(self) -> str:
        return "npu"

    def _get_free_memory(self, device: str) -> float:
        if self._torch_npu is None:
            logger.warning("torch_npu is not available")
            return 0

        if not self._torch_npu.npu.is_available():
            logger.warning("NPU device is not available: %s", device)
            return 0

        total_memory = self._torch_npu.npu.get_device_properties(device).total_memory
        used_memory = self._torch_npu.npu.memory_allocated(device)
        logger.debug(
            "NPU device %s - Total memory: %d bytes, Used memory: %d bytes, Free memory: %d bytes",
            device,
            total_memory,
            used_memory,
            total_memory - used_memory,
        )
        return total_memory - used_memory

    def execute(self) -> Optional[dict[int, Optional[float]]]:
        if not self._torch_npu:
            logger.warning("torch_npu is not available, skip the stress test")
            return None

        return super().execute()
