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
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import as_completed, ProcessPoolExecutor

from typing import Any, Dict, List, Optional

from ..utils.ascend import get_npu_count, RankTable, search_weight_dir_mindie, search_weight_dir_vllm, search_weight_dir_sglang, get_framework, Framework
from ..utils.helper import is_valid_ip


logger = logging.getLogger(__name__)


class CollectStrategy(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def execute(self) -> Any:
        pass


class CollectStrategyGroup(CollectStrategy):
    def __init__(self, name: str, strategies: Optional[List[CollectStrategy]] = None):
        self._strategies = []
        if strategies is not None:
            try:
                self._strategies = list(strategies)
            except TypeError:
                logger.error(
                    "strategies must be an iterable. Got %s instead", strategies
                )
                raise

        if not all(
            isinstance(strategy, (CollectStrategy, CollectStrategyGroup))
            for strategy in self._strategies
        ):
            raise TypeError(
                "All collect_strategies must be instances of CollectStrategy or CollectStrategyGroup"
            )

        super().__init__(name)

    def add(self, strategy: CollectStrategy) -> "CollectStrategyGroup":
        if not isinstance(strategy, (CollectStrategy, CollectStrategyGroup)):
            raise TypeError(
                "collect_strategy must be an instance of CollectStrategy or CollectStrategyGroup"
            )

        self._strategies.append(strategy)
        return self

    def execute(self) -> Dict[str, Dict[str, Any]]:
        return {strategy.name: strategy.execute() for strategy in self._strategies}


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
            except Exception as e:
                logger.warning("Failed to execute lscpu command: %s", str(e))
                return None

        return self._parse_output(self._output)


class CPUHighPerformance(CollectStrategy):
    def __init__(self, name: str = "cpu_high_performance"):
        super().__init__(name)
        self._dmidecode_output = None
        self._cpupower_output = None
        self._lshw_output = None

    @staticmethod
    def _check_by_psutil():
        import psutil

        cpu_freq = psutil.cpu_freq()
        if not cpu_freq:
            logger.debug("Unable to get CPU frequency information via psutil")
            return False
        return cpu_freq.current == cpu_freq.max

    def _check_by_dmidecode(self):
        dmidecode_path = shutil.which("dmidecode")
        if dmidecode_path is None:
            logger.debug("dmidecode command not found in system PATH")
            return False

        if not self._dmidecode_output:
            cmd = shlex.split(f"{dmidecode_path} -t processor")
            try:
                self._dmidecode_output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True
                )
            except Exception as e:
                logger.debug(f"Failed to execute dmidecode command: {e}")
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

    def _check_by_cpupower(self):
        cpupower_path = shutil.which("cpupower")
        if cpupower_path is None:
            logger.debug("cpupower command not found in system PATH")
            return False

        if not self._cpupower_output:
            cmd = shlex.split(f"{cpupower_path} frequency-info")
            try:
                self._cpupower_output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True
                )
            except Exception as e:
                logger.debug(f"Failed to execute cpupower command: {e}")
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

    def _check_by_lshw(self):
        lshw_path = shutil.which("lshw")
        if lshw_path is None:
            logger.debug("lshw command not found in system PATH")
            return False

        if not self._lshw_output:
            cmd = shlex.split(f"{lshw_path} -c cpu")
            try:
                self._lshw_output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True
                )
            except Exception as e:
                logger.debug(f"Failed to execute lshw command: {e}")
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

    def _check_by_scaling_governor(self):
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
                logger.debug(f"Scaling governor file not found for CPU core {core_id}")
                return False

            try:
                with open(gov_path, encoding="utf-8") as f:
                    if f.read().strip() != "performance":
                        logger.debug(
                            f"CPU core {core_id} scaling governor is not set to performance mode"
                        )
                        return False
            except Exception as e:
                logger.debug(
                    f"Failed to read scaling governor file for CPU core {core_id}: {e}"
                )
                return False
        return True

    def execute(self):
        return (
            self._check_by_dmidecode()
            or self._check_by_scaling_governor()
            or self._check_by_cpupower()
            or self._check_by_psutil()
            or self._check_by_lshw()
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
        except Exception as e:
            logger.warning(f"Failed to read /proc/cpuinfo file: {e}")
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
        except Exception as e:
            logger.warning(f"Failed to read transparent hugepage configuration: {e}")
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
        except Exception as e:
            logger.warning(f"Failed to get system page size: {e}")
            return None


class Sys(CollectStrategyGroup):
    def __init__(
        self,
        name="sys",
        strategies=[
            Lscpu(),
            CPUHighPerformance(),
            VirtualMachine(),
            TransparentHugepage(),
            Kernel(),
            PageSize(),
        ],
    ):
        super().__init__(name, strategies)


class Config(CollectStrategy):
    def __init__(self, name, *, config_path):
        super().__init__(name)
        self._config_path = config_path
        self._framework = get_framework()

    def _process_json(self, content):
        logger.debug('Processing JSON configuration file: %r', self._config_path)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON configuration file %r: %s", self._config_path, str(e))
            return content

    def _process_yaml(self, content):
        logger.debug('Processing YAML configuration file: %s', self._config_path)
        import yaml
        try:
            if "---" in content:
                return list(yaml.safe_load_all(content))
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.warning("Failed to parse YAML configuration file %r: %s", self._config_path, str(e))
            return content

    def _process_shell(self, content):
        logger.debug('Processing shell configuration file: %s', self._config_path)
        return content

    def execute(self):
        if not self._config_path:
            logger.warning("Configuration path is empty or not provided")
            return None

        from msguard import Rule

        rule = Rule.input_file_read
        if not rule.is_satisfied_by(self._config_path):
            logger.warning("Expected %r to be %s", self._config_path, rule)
            return None

        with open(self._config_path) as f:
            content = f.read()

        extension_to_process_method = {
            '.json': self._process_json,
            '.yaml': self._process_yaml,
            '.yml': self._process_yaml,
            '.sh': self._process_shell
        }

        if '.' not in self._config_path:
            logger.warning(
                "Unsupported configuration file format: %r", self._config_path
            )
            return content

        ext = os.path.splitext(self._config_path)[-1]
        if ext not in extension_to_process_method:
            logger.warning(
                "Unsupported configuration file format: %r", self._config_path
            )
            return content

        return extension_to_process_method[ext](content)


class Weight(CollectStrategy):
    def __init__(
        self,
        name: str = "weight",
        *,
        weight_dir: str = "",
        tensor_suffix=".safetensors",
        max_size: int = 10 * 1024**3,
        chunk_size: int = 256 * 1024**2,
    ):
        super().__init__(name)
        self._weight_dir = weight_dir
        self._tensor_suffix = tensor_suffix
        self.max_size = max_size
        self._chunk_size = chunk_size

    def _calculate_hash256(self, tensor_file):
        sha256_hash = hashlib.sha256()
        with open(tensor_file, "rb") as f:
            while True:
                data = f.read(self._chunk_size)
                if not data:
                    break
                sha256_hash.update(data)
        return sha256_hash.hexdigest()

    def _validate_weight_dir(self):
        from msguard import Rule

        rule = Rule.input_dir_traverse
        if not rule.is_satisfied_by(self._weight_dir):
            logger.warning("Expected %r to be %s", self._weight_dir, rule)
            return False
        return True

    def _is_valid_tensor_file(self, tensor_file, weight_rule):
        if not os.path.isfile(tensor_file) or not tensor_file.endswith(
            self._tensor_suffix
        ):
            return False
        if not weight_rule.is_satisfied_by(tensor_file):
            logger.warning("Expected %r to be %s. Skipped", tensor_file, weight_rule)
            return False

        return True

    def _filter_valid_tensor_files(self):
        from msguard import Path, where

        weight_rule = where(
            os.getuid() == 0,
            Path.is_file(),
            Path.is_file()
            & ~Path.has_soft_link()
            & Path.is_readable()
            & ~Path.is_writable_to_group_or_others()
            & Path.is_consistent_to_current_user()
            & Path.is_size_reasonable(size_limit=self.max_size),
            description="current user is root",
        )

        return [
            os.path.join(self._weight_dir, f)
            for f in os.listdir(self._weight_dir)
            if self._is_valid_tensor_file(os.path.join(self._weight_dir, f), weight_rule)
        ]

    def _parallel_hash_calculation(self, tensor_files):
        if not tensor_files:
            logger.warning("No valid tensor files found in the specified directory")
            return None

        max_workers = min(len(tensor_files), os.cpu_count() or 1)
        tensor_id_pattern = re.compile(
            r"(\d{5})-of-\d{5}" + re.escape(self._tensor_suffix)
        )

        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {}
            for tensor_file in tensor_files:
                tensor_basename = os.path.basename(tensor_file)
                m = tensor_id_pattern.search(tensor_basename)
                tensor_id = m.group(1) if m else tensor_basename
                future = executor.submit(self._calculate_hash256, tensor_file)
                future_to_id[future] = tensor_id

            for future in as_completed(future_to_id):
                tensor_id = future_to_id[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.warning(
                        "Failed to calculate hash for tensor id %r: %s", tensor_id, e
                    )
                    result = None
                results[tensor_id] = result

        return results

    def execute(self):
        if not self._validate_weight_dir():
            return None
        tensor_files = self._filter_valid_tensor_files()
        if not tensor_files:
            logger.warning("No valid tensor files found in the specified directory")
            return None
        return self._parallel_hash_calculation(tensor_files)


class _Ascend(CollectStrategy):
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

    def execute(self) -> Any:
        home_path = os.getenv(self._home_environ) if self._home_environ else ""
        base_path = home_path or self._default_home
        if self._version_path.startswith("/"):
            full_path = os.path.normpath(self._version_path)
        else:
            full_path = os.path.normpath(os.path.join(base_path, self._version_path))

        if not os.path.isfile(full_path):
            logger.debug("The version file is not found at: %r", full_path)
            return None

        results = {}
        with open(full_path) as f:
            for line in f:
                line = line.strip()

                if not line:
                    logger.debug("The version file is empty: %r", full_path)
                    return None

                parts = line.split("=", 1) if "=" in line else line.split(":", 1)
                if len(parts) != 2:
                    logger.debug("Unexpected format in line: %r", line)
                    continue
                results[parts[0].strip()] = parts[1].strip()

        return results if results else None


class Driver(_Ascend):
    def __init__(
        self,
        name: str = "driver",
        *,
        version_path: str = "/usr/local/Ascend/driver/version.info",
        default_home: str = "",
        home_environ: str = "",
    ):
        super().__init__(
            name,
            version_path=version_path,
            default_home=default_home,
            home_environ=home_environ,
        )


class Toolkit(_Ascend):
    def __init__(
        self,
        name: str = "toolkit",
        *,
        version_path: str = "toolkit/version.info",
        default_home: str = "/usr/local/Ascend/ascend-toolkit/latest",
        home_environ: str = "ASCEND_TOOLKIT_HOME",
    ):
        super().__init__(
            name,
            version_path=version_path,
            default_home=default_home,
            home_environ=home_environ,
        )


class OppKernel(_Ascend):
    def __init__(
        self,
        name: str = "opp_kernel",
        *,
        version_path: str = "opp_kernel/version.info",
        default_home: str = "/usr/local/Ascend/ascend-toolkit/latest",
        home_environ: str = "ASCEND_TOOLKIT_HOME",
    ):
        super().__init__(
            name,
            version_path=version_path,
            default_home=default_home,
            home_environ=home_environ,
        )


class TB(_Ascend):
    def __init__(
        self,
        name: str = "atb",
        *,
        version_path: str = "../../version.info",
        default_home: str = "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_0",
        home_environ: str = "ATB_HOME_PATH",
    ):
        super().__init__(
            name,
            version_path=version_path,
            default_home=default_home,
            home_environ=home_environ,
        )


class MindIE(_Ascend):
    def __init__(
        self,
        name: str = "mindie",
        *,
        version_path: str = "../version.info",
        default_home: str = "/usr/local/Ascend/mindie/latest/mindie-llm",
        home_environ: str = "MINDIE_LLM_HOME_PATH",
    ):
        super().__init__(
            name,
            version_path=version_path,
            default_home=default_home,
            home_environ=home_environ,
        )


class TBSpeed(_Ascend):
    def __init__(
        self,
        name: str = "atb-models",
        *,
        version_path: str = "version.info",
        default_home: str = "/usr/local/Ascend/atb-models",
        home_environ: str = "ATB_SPEED_HOME_PATH",
    ):
        super().__init__(
            name,
            version_path=version_path,
            default_home=default_home,
            home_environ=home_environ,
        )


class Ascend(CollectStrategyGroup):
    def __init__(
        self,
        name: str = "ascend",
        strategies=[
            Driver(),
            Toolkit(),
            OppKernel(),
            TB(),
            MindIE(),
            TBSpeed(),
        ],
    ):
        super().__init__(name, strategies)


class Ping(CollectStrategy):
    def __init__(self, name="ping", *, ip: str = ""):
        if not isinstance(ip, str):
            raise TypeError("IP address must be a string: %r" % ip)

        if not is_valid_ip(ip):
            raise ValueError("IP address format is invalid: %r" % ip)

        super().__init__(name)
        self._ip = ip

    def execute(self) -> Any:
        if not self._ip:
            logger.warning("IP address is empty or not provided")
            return None

        ping_path = shutil.which("ping")
        if ping_path is None:
            logger.warning("ping command not found in system PATH")
            return None

        cmd = f"{ping_path} -c 3 -q -W 2 {self._ip}"

        try:
            output = subprocess.check_output(
                shlex.split(cmd), stderr=subprocess.STDOUT, text=True, timeout=5
            )
            return output
        except Exception as e:
            logger.warning(
                "Failed to execute ping command for IP %s: %s", self._ip, str(e)
            )
            return None


class HccnTool(CollectStrategy):
    def __init__(self, name: str, *, device_id: int, timeout=None):
        super().__init__(name)
        self._device_id = device_id
        self._timeout = timeout or 3

        self._output = None
        self._hccn_tool_path = '/usr/local/Ascend/driver/tools/hccn_tool'

    @property
    @abstractmethod
    def cmd(self) -> str:
        pass

    def execute(self):
        from msguard import Rule

        rule = Rule.input_file_exec
        if not rule.is_satisfied_by(self._hccn_tool_path):
            logger.warning("Expected %r to be %s", self._hccn_tool_path, rule)
            return None

        if self._output is None:
            try:
                self._output = subprocess.check_output(
                    shlex.split(self.cmd), stderr=subprocess.DEVNULL, text=True,
                    timeout=self._timeout
                )
            except Exception as e:
                logger.warning(
                    "Failed to execute hccn_tool command for device %s: %s",
                    self._device_id,
                    str(e),
                )
                self._output = "100% packet loss"

        return {self._device_id: self._output}


class Vnic(HccnTool):
    def __init__(self, name: str = "vnic", *, device_id: int):
        super().__init__(name, device_id=device_id)

    @property
    def cmd(self):
        return f"{self._hccn_tool_path} -i {self._device_id} -vnic -g"


class Link(HccnTool):
    def __init__(self, name: str = "link", *, device_id: int):
        super().__init__(name, device_id=device_id)

    @property
    def cmd(self):
        return f"{self._hccn_tool_path} -i {self._device_id} -link -g"


class Tls(HccnTool):
    def __init__(self, name: str = "tls", *, device_id: int):
        super().__init__(name, device_id=device_id)

    @property
    def cmd(self):
        return f"{self._hccn_tool_path} -i {self._device_id} -tls -g"


class HcclPing(HccnTool):
    def __init__(self, name: str = "hccl_ping", *, device_id: int, device_ip: str):
        if not isinstance(device_ip, str):
            raise TypeError("IP address must be a string: %r" % device_ip)

        if not is_valid_ip(device_ip):
            raise ValueError("IP address format is invalid: %r" % device_ip)

        super().__init__(name, device_id=device_id)
        self._device_ip = device_ip

    @property
    def cmd(self):
        return f"{self._hccn_tool_path} -i {self._device_id} -ping -g address {self._device_ip}"


class HccsPing(HccnTool):
    def __init__(self, name: str = "hccl_ping", *, device_id: int, device_ip: str):
        if not isinstance(device_ip, str):
            raise TypeError("IP address must be a string: %r" % device_ip)

        if not is_valid_ip(device_ip):
            raise ValueError("IP address format is invalid: %r" % device_ip)

        super().__init__(name, device_id=device_id)
        self._device_ip = device_ip

    @property
    def cmd(self):
        return f"{self._hccn_tool_path} -i {self._device_id} -hccs_ping -g address {self._device_ip}"


class Network(CollectStrategyGroup):
    def __init__(
        self,
        name: str = "network",
        strategies=None,
        *,
        rank_table: RankTable,
        npu_count=None,
    ):
        if not isinstance(rank_table, RankTable):
            raise TypeError("rank_table must be an instance of RankTable")

        npu_count = npu_count if npu_count else get_npu_count()
        if npu_count == 0:
            raise ValueError("No NPU devices found in the system")

        all_device_ips = (
            device_info.device_ip
            for device_info_list in rank_table.host_to_devices.values()
            for device_info in device_info_list
        )
        if not all_device_ips:
            raise ValueError("No device IP addresses found in the rank table")

        ping_cls = (
            HccsPing if getattr(rank_table, "version", "1.0") == "1.2" else HcclPing
        )
        strategies = [Ping(ip=host_ip) for host_ip in rank_table.host_to_devices]
        for device_id in range(npu_count):
            strategies.extend(
                (
                    Vnic(device_id=device_id),
                    Link(device_id=device_id),
                    Tls(device_id=device_id),
                )
            )
            for device_ip in all_device_ips:
                strategies.append(ping_cls(device_id=device_id, device_ip=device_ip))

        super().__init__(name, strategies=strategies)
        self._rank_table = rank_table


class Stress(CollectStrategy):
    def __init__(self, name, *, batch_size, seq_len, hidden_size, intermediate_size):
        self._torch = None
        try:
            import torch

            self._torch = torch
        except ImportError:
            logger.warning()

        super().__init__(name)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size

    @staticmethod
    def _calculate_tensor_memory(shape):
        return math.prod(shape) * 4  # default to float32

    @abstractmethod
    def _get_free_memory(self, device):
        pass

    def _check_memory_for_matmul(self, device_pos):
        mat_a_mem = self._calculate_tensor_memory(
            (self._batch_size, self._seq_len, self._hidden_size)
        )
        mat_b_mem = self._calculate_tensor_memory(
            (self._batch_size, self._hidden_size, self._intermediate_size)
        )
        mat_c_mem = self._calculate_tensor_memory(
            (self._seq_len, self._intermediate_size)
        )
        total_required = mat_a_mem + mat_b_mem + mat_c_mem

        free_memory = self._get_free_memory(device_pos)
        safety_margin = 0.2

        available_with_margin = free_memory * (1 - safety_margin)
        has_enough_mem = total_required <= available_with_margin

        if not has_enough_mem:
            logger.warning()
            return False

        return True

    def _matmul_stress_test(self, device_type, device_id):
        device_pos = f"{device_type}:{device_id}"

        if not self._check_memory_for_matmul(device_pos):
            return

        # 执行多次矩阵运算：mat_c + mat_a × mat_b
        for _ in range(10):
            mat_a = self.torch.randn(
                self._batch_size, self._seq_len, self._hidden_size
            ).to(device_pos)
            mat_b = self.torch.randn(
                self._batch_size, self._hidden_size, self._intermediate_size
            ).to(device_pos)
            mat_c = self.torch.randn(self._seq_len, self._intermediate_size).to(
                device_pos
            )
            self.torch.addbmm(mat_c, mat_a, mat_b)

    def execute(self):
        if not self.torch:
            return None

        cpu_ids = os.cpu_count()
        self.torch.set_num_threads(cpu_ids)

        output = dict.fromkeys(range(cpu_ids), 0)
        for cpu_id in SimpleProgressBar(range(cpu_ids)):
            start_time = time.time()
            self._matmul_stress_test(cpu_id)
            end_time = time.time()
            cpu_time = (end_time - start_time) * 1000

            output[cpu_id] = cpu_time

        return output
