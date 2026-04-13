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
import platform
import re
import shlex
import shutil
import subprocess

from .base_strategy import CollectStrategyGroup, CollectStrategy
from ..utils import LOGGER, Utils


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

    def sync(self, target_data: dict):
        super().sync(target_data)
        LOGGER.info("Sys check passed.")


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

    def compare_architecture(self, current: dict):
        target_architecture = self._target.get("Architecture")
        current_architecture = current.get("Architecture")
        if target_architecture != current_architecture:
            Utils.log_error_and_exit("Current architecture {} is not equal to the architecture {} in dump file"
                                     "".format(current_architecture, target_architecture))

    def compare_cpu(self, current: dict):
        target_cpu = int(self._target.get("CPU(s)"))
        current_cpu = int(current.get("CPU(s)"))
        if target_cpu > current_cpu:
            LOGGER.warning("Current cpus {} is less than the cpus {} in dump file, "
                           "which may cause performance issue".format(current_cpu, target_cpu))
        elif target_cpu < current_cpu:
            LOGGER.warning("Current cpus {} is greater than the cpus {} in dump file, "
                           "which may cause performance issue when enable some features".format(current_cpu,
                                                                                                target_cpu))

    def compare_hyper_threading(self, current: dict):
        current_threads = int(current.get("Thread(s) per core", 0))
        if current_threads > 1:
            LOGGER.warning("Hyper-threading is enabled. It is recommended to disable it for better performance while "
                           "using CPU binding strategy.")

    def execute(self):
        lscpu_path = shutil.which("lscpu")
        if not lscpu_path:
            LOGGER.warning("lscpu command not found in system PATH")
            return None

        if self._output is None:
            try:
                self._output = subprocess.check_output(
                    [lscpu_path], stderr=subprocess.DEVNULL, text=True, env={"LANG": "en_US.UTF-8"}
                )
            except Exception as e:
                LOGGER.warning("Failed to execute lscpu command: %s", str(e))
                return None

        return self._parse_output(self._output)

    def sync(self, target_data: dict):
        super().sync(target_data)
        current = self.execute()
        self.compare_architecture(current)
        self.compare_cpu(current)
        self.compare_hyper_threading(current)


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
            LOGGER.debug("Unable to get CPU frequency information via psutil")
            return False
        return cpu_freq.current == cpu_freq.max

    def _check_via_dmidecode(self):
        dmidecode_path = shutil.which("dmidecode")
        if dmidecode_path is None:
            LOGGER.debug("dmidecode command not found in system PATH")
            return False

        if self._dmidecode_output is None:
            cmd = shlex.split(f"{dmidecode_path} -t processor")
            try:
                self._dmidecode_output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True
                )
            except Exception as e:
                LOGGER.debug(f"Failed to execute dmidecode command: {e}")
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
            LOGGER.debug("cpupower command not found in system PATH")
            return False

        if self._cpupower_output is None:
            cmd = shlex.split(f"{cpupower_path} frequency-info")
            try:
                self._cpupower_output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True
                )
            except Exception as e:
                LOGGER.debug(f"Failed to execute cpupower command: {e}")
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
            LOGGER.debug("lshw command not found in system PATH")
            return False

        if self._lshw_output is None:
            cmd = shlex.split(f"{lshw_path} -c cpu")
            try:
                self._lshw_output = subprocess.check_output(
                    cmd, stderr=subprocess.DEVNULL, text=True
                )
            except Exception as e:
                LOGGER.debug(f"Failed to execute lshw command: {e}")
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
            LOGGER.debug("Unable to determine CPU count")
            return False

        scaling_governor_pattern = (
            "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor"
        )
        for core_id in range(cpu_count):
            gov_path = scaling_governor_pattern.format(core_id)
            if not os.path.isfile(gov_path):
                LOGGER.debug(f"Scaling governor file not found for CPU core {core_id}")
                return False

            try:
                with open(gov_path, encoding="utf-8") as f:
                    if f.read().strip() != "performance":
                        LOGGER.debug(
                            f"CPU core {core_id} scaling governor is not set to performance mode"
                        )
                        return False
            except Exception as e:
                LOGGER.debug(
                    f"Failed to read scaling governor file for CPU core {core_id}: {e}"
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

    def sync(self, target_data: dict):
        if not self.execute():
            LOGGER.warning(f"CPU high performance strategy is disabled. It is recommended to enable it.")


class VirtualMachine(CollectStrategy):
    def __init__(self, name: str = "virtual_machine"):
        super().__init__(name)

    def execute(self):
        cpu_info_path = "/proc/cpuinfo"

        if not os.path.isfile(cpu_info_path):
            LOGGER.debug("/proc/cpuinfo file not found")
            return False

        try:
            with open(cpu_info_path) as f:
                return any("hypervisor" in line for line in f)
        except Exception as e:
            LOGGER.warning(f"Failed to read /proc/cpuinfo file: {e}")
            return False


class TransparentHugepage(CollectStrategy):
    def __init__(self, name: str = "transparent_hugepage"):
        super().__init__(name)

    def execute(self):
        transparent_hugepage_path = "/sys/kernel/mm/transparent_hugepage/enabled"

        if not os.path.isfile(transparent_hugepage_path):
            LOGGER.debug("Transparent hugepage configuration file not found")
            return None

        try:
            with open(transparent_hugepage_path) as f:
                return f.read().strip()
        except Exception as e:
            LOGGER.warning(f"Failed to read transparent hugepage configuration: {e}")
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
            LOGGER.warning(f"Failed to get system page size: {e}")
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
