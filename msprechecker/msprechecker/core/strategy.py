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

import hashlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..util import get_npu_count, is_valid_ip, RankTable


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
                logger.exception(
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
            with open(cpu_info_path) as f:
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
            with open(transparent_hugepage_path) as f:
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
        try:
            result_apt = subprocess.run(
                ["/usr/bin/apt", "list", "--installed", "libjemalloc*"],
                capture_output=True,
                text=True,
                check=False,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

        return result_apt.returncode == 0 and "libjemalloc" in result_apt.stdout

    def _check_via_yum(self) -> bool:
        """Check if jemalloc is installed via yum."""
        try:
            result_yum = subprocess.run(
                ["/usr/bin/yum", "list", "installed", "jemalloc*"],
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
            with open(self._config_path) as f:
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
            logger.warning(
                "Expected %r to be a regular file. Weight strategy skipped", path
            )
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

    def _filter_valid_tensor_files(self) -> List[str]:
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

    def _parallel_hash_calculation(
        self, tensor_files: List[str]
    ) -> Dict[str, Optional[str]]:
        max_workers = min(len(tensor_files), self._max_hash_workers)
        results: Dict[str, Optional[str]] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self._calculate_hash256, f): self._get_tensor_id(f)
                for f in tensor_files
            }
            for future in as_completed(future_to_id):
                tensor_id = future_to_id[future]
                try:
                    results[tensor_id] = future.result()
                except Exception:
                    logger.exception(
                        "Failed to calculate hash for tensor id %r", tensor_id
                    )
                    results[tensor_id] = None

        return results

    def execute(self) -> Optional[Dict[str, Optional[str]]]:
        if not self._validate_weight_dir():
            return None

        tensor_files = self._filter_valid_tensor_files()
        if not tensor_files:
            logger.warning("No valid tensor files found in %r", self._weight_dir)
            return None

        return self._parallel_hash_calculation(tensor_files)


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

        if not self._default_home:
            # No reference root to validate against; trust the env var.
            return home_path

        # Use pathlib.relative_to() instead of startswith() to avoid the
        # "/usr/local/Ascend-evil" false-positive bypass.
        allowed_root = Path(self._default_home).parent.resolve()
        candidate = Path(home_path).resolve()
        try:
            candidate.relative_to(allowed_root)
            return home_path
        except ValueError:
            logger.warning(
                "Environment variable %r points to %r which is outside "
                "the expected root %r, falling back to default",
                self._home_environ,
                home_path,
                str(allowed_root),
            )
            return ""

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
                    logger.debug("Unexpected format in line: %r", line)
                    continue
                results[parts[0].strip()] = parts[1].strip()
        return results

    def execute(self) -> Any:
        home_path = self._resolve_home()
        full_path = self._resolve_full_path(home_path)

        if not full_path.is_file():
            logger.debug("Version file not found at: %r", str(full_path))
            return None

        try:
            results = self._parse_version_file(full_path)
        except OSError:
            logger.exception("Failed to read version file %r", str(full_path))
            return None

        if not results:
            logger.debug("Version file yielded no data: %r", str(full_path))
            return None

        return results


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


class Driver(_AscendComponent):
    # Driver has no home directory concept; version_path is always absolute.
    _DEFAULT_NAME = "driver"
    _DEFAULT_VERSION_PATH = "/usr/local/Ascend/driver/version.info"


class Toolkit(_AscendComponent):
    _DEFAULT_NAME = "toolkit"
    _DEFAULT_VERSION_PATH = "toolkit/version.info"
    _DEFAULT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
    _DEFAULT_ENVIRON = "ASCEND_TOOLKIT_HOME"


class OppKernel(_AscendComponent):
    _DEFAULT_NAME = "opp_kernel"
    _DEFAULT_VERSION_PATH = "opp_kernel/version.info"
    _DEFAULT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
    _DEFAULT_ENVIRON = "ASCEND_TOOLKIT_HOME"


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
            ],
        )


class Ping(CollectStrategy):
    def __init__(self, name: str = "ping", *, ip: str) -> None:
        if not isinstance(ip, str):
            raise TypeError(f"IP address must be a string: {ip!r}")
        if not is_valid_ip(ip):
            raise ValueError(f"IP address format is invalid: {ip!r}")

        super().__init__(name)
        self._ip = ip
        self._ping_path = shutil.which("ping")

    def _ping_ip(self) -> Optional[str]:
        """Ping the IP address and return the output."""
        cmd = f"{self._ping_path} -c 3 -q -W 2 {self._ip}"
        try:
            return subprocess.check_output(
                shlex.split(cmd),
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )
        except Exception:
            logger.warning(
                "Failed to execute ping command for IP %s", self._ip, exc_info=True
            )
            return None

    def execute(self) -> Optional[str]:
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
        device_ids: List[int],
        max_workers: int = 8,
        timeout: float = 3,
    ):
        super().__init__(name)
        self._device_ids = device_ids
        self._max_workers = max_workers
        self._timeout = timeout
        self._bin_path = shutil.which("hccn_tool") or self.HCCN_TOOL_PATH

    def _run(self, cmd: List[str]) -> Optional[str]:
        """Execute a single hccn_tool command and return its stdout and stderr."""
        try:
            return subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self._timeout,
            )
        except Exception:
            if logger.isEnabledFor(logging.WARNING):
                logger.exception("Failed to execute command: %s", cmd)
            return None

    @abstractmethod
    def execute(self) -> List[Any]:
        """Execute the hccn_tool command for each device and return a list of results."""


class _SingleOption(HccnTool):
    """
    Runs:  hccn_tool -i <device_id> <option> -g
    Returns: List[Optional[str]]  — one entry per device_id
    """

    option: str
    default_name: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "option", None), str):
            raise TypeError(
                f"{cls.__name__} must define a string class variable 'option'"
            )
        if not isinstance(getattr(cls, "default_name", None), str):
            raise TypeError(
                f"{cls.__name__} must define a string class variable 'default_name'"
            )

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        device_ids: List[int],
        max_workers: int = 8,
        timeout: float = 3,
    ):
        super().__init__(
            name=name or self.default_name,
            device_ids=device_ids,
            max_workers=max_workers,
            timeout=timeout,
        )

    def _build_cmd(self, device_id: int) -> List[str]:
        return [self._bin_path, "-i", str(device_id), self.option, "-g"]

    def execute(self) -> List[Optional[str]]:
        cmds = [self._build_cmd(device_id) for device_id in self._device_ids]
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self._run, cmds))


class _OptionIp(HccnTool):
    """
    Runs:  hccn_tool -i <device_id> <option> -g address <peer_ip>
           for every (device_id, peer_ip) combination.

    Returns:
        List[Dict[str, Optional[str]]]
        — one dict per device_id; keys are peer IPs, values are raw output.
    """

    option: str
    default_name: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "option", None), str):
            raise TypeError(
                f"{cls.__name__} must define a string class variable 'option'"
            )
        if not isinstance(getattr(cls, "default_name", None), str):
            raise TypeError(
                f"{cls.__name__} must define a string class variable 'default_name'"
            )

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        device_ids: List[int],
        device_ips: List[str],
        max_workers: int = 8,
        timeout: float = 3,
    ):
        super().__init__(
            name=name or self.default_name,
            device_ids=device_ids,
            max_workers=max_workers,
            timeout=timeout,
        )
        self._device_ips = device_ips

    def _build_cmd(self, device_id: int, ip: str) -> List[str]:
        return [self._bin_path, "-i", str(device_id), self.option, "-g", "address", ip]

    def _probe_device(self, device_id: int) -> Dict[str, Optional[str]]:
        """Run probes for all peer IPs from a single device — sequentially."""
        return {
            ip: self._run(self._build_cmd(device_id, ip)) for ip in self._device_ips
        }

    def execute(self) -> List[Dict[str, Optional[str]]]:
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self._probe_device, self._device_ids))


class Vnic(_SingleOption):
    option = "-vnic"
    default_name = "vnic"


class Link(_SingleOption):
    option = "-link"
    default_name = "link"


class Tls(_SingleOption):
    option = "-tls"
    default_name = "tls"


class HcclPing(_OptionIp):
    option = "-ping"
    default_name = "hccl_ping"


class HccsPing(_OptionIp):
    option = "-hccs_ping"
    default_name = "hccs_ping"


class Network(CollectStrategyGroup):
    _HCCS_PING_VERSION = "1.2"

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
        ping_cls = (
            HccsPing if rank_table.version == self._HCCS_PING_VERSION else HcclPing
        )

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
        self, name, *, batch_size, seq_len, hidden_size, intermediate_size, epochs=5
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
                raise ValueError(
                    f"{param_name} must be a positive integer, got {param_value!r}"
                )

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
    def _get_free_memory(self, device) -> float:
        pass

    def _calculate_tensor_memory(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        import operator

        # Use functools.reduce for Python 3.7 compatibility (math.prod added in 3.8)
        from functools import reduce

        return reduce(operator.mul, shape, 1) * 4  # float32 = 4 bytes

    def _check_memory_for_matmul(self, device_pos):
        mat_a_mem = self._calculate_tensor_memory(
            (self._batch_size, self._seq_len, self._hidden_size)
        )
        mat_b_mem = self._calculate_tensor_memory(
            (self._batch_size, self._hidden_size, self._intermediate_size)
        )
        # addbmm output shape is (seq_len, intermediate_size); unchanged
        mat_c_mem = self._calculate_tensor_memory(
            (self._seq_len, self._intermediate_size)
        )
        total_required = mat_a_mem + mat_b_mem + mat_c_mem

        free_memory = self._get_free_memory(device_pos)
        safety_margin = 0.2
        available_with_margin = free_memory * (1 - safety_margin)
        has_enough_mem = total_required <= available_with_margin
        logger.debug(
            "Device %s - Required memory: %d bytes, Free memory: %d bytes, "
            "Available with margin: %d bytes",
            device_pos,
            total_required,
            free_memory,
            available_with_margin,
        )

        if not has_enough_mem:
            logger.warning(
                "Insufficient memory on device %s for matmul operation", device_pos
            )
            return False

        return True

    def _matmul_stress_test(self, device_id):
        """Run matrix-multiply stress on one device and return elapsed ms."""
        device_pos = f"{self.device_type}:{device_id}"

        if not self._check_memory_for_matmul(device_pos):
            return 0.0

        start_time = time.perf_counter()
        for _ in range(self._epochs):
            mat_a = self._torch.randn(
                self._batch_size, self._seq_len, self._hidden_size
            ).to(device_pos)
            mat_b = self._torch.randn(
                self._batch_size, self._hidden_size, self._intermediate_size
            ).to(device_pos)
            mat_c = self._torch.zeros(self._seq_len, self._intermediate_size).to(
                device_pos
            )
            self._torch.addbmm(mat_c, mat_a, mat_b)

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000

    def execute(self):
        if not self._torch:
            logger.error("torch is not available, skip the stress test")
            return None

        output = {}
        cpu_count = os.cpu_count() or 1
        self._torch.set_num_threads(cpu_count)
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            future_to_id = {
                executor.submit(self._matmul_stress_test, cpu_id): cpu_id
                for cpu_id in range(cpu_count)
            }
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
        batch_size=1,
        seq_len=512,
        hidden_size=1024,
        intermediate_size=64,
        epochs=5,
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

    def _get_free_memory(self, device) -> float:
        import psutil

        memory_available = psutil.virtual_memory().available
        logger.debug("Available CPU memory: %d bytes", memory_available)
        return memory_available


class NPU(Stress):
    def __init__(
        self,
        name: str = "npu",
        *,
        batch_size=1,
        seq_len=4096,
        hidden_size=8192,
        intermediate_size=3584,
        epochs=5,
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

    def _get_free_memory(self, device):
        if self._torch_npu is None:
            logger.warning("torch_npu is not available")
            return 0

        if not self._torch_npu.npu.is_available():
            logger.warning("NPU device is not available: %s", device)
            return 0

        total_memory = self._torch_npu.npu.get_device_properties(device).total_memory
        used_memory = self._torch_npu.npu.memory_allocated(device)
        logger.debug(
            "NPU device %s - Total memory: %d bytes, Used memory: %d bytes, "
            "Free memory: %d bytes",
            device,
            total_memory,
            used_memory,
            total_memory - used_memory,
        )
        return total_memory - used_memory

    def execute(self):
        if not self._torch_npu:
            logger.warning("torch_npu is not available, skip the stress test")
            return None

        return super().execute()
