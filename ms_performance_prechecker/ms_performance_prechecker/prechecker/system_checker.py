# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform
from ms_performance_prechecker.prechecker.register import register_checker, GroupRrechecker, RrecheckerBase
from ms_performance_prechecker.prechecker.register import show_check_result, record, CONTENT_PARTS, CheckResult
from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, SUGGESTION_TYPES
from ms_performance_prechecker.prechecker.utils import get_dict_value_by_pos, str_to_digit, logger

try:
    import acl
except ModuleNotFoundError:
    logger.warning("Import acl error, will skip getting NPU info. Install and source cann toolkit if needed")
    acl = None


DRIVER_VERSION_PATH = "/usr/local/Ascend/driver/version.info"
CPUINFO_PATH = "/proc/cpuinfo"
TRANSPARENT_HUGEPAGE_PATH = "/sys/kernel/mm/transparent_hugepage/enabled"
GOVERNOR_PATH_FORMATTER = "/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor"


def get_cpu_info():
    import subprocess
    from shutil import which

    lscpu_path = which("lscpu")
    if not lscpu_path:
        logger.error("lscpu command not exists, will skip getting cpu info.")
        return {}

    try:
        result = subprocess.run([lscpu_path], capture_output=True, text=True, check=True, shell=False)
    except Exception as err:
        logger.error("Failed calling lscpu, will skip getting cpu info.")

    cpu_info = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            cpu_info[key.strip()] = value.strip()
    return cpu_info


class SystemInfoCollect(RrecheckerBase):
    __checker_name__ = "SystemInfo"

    def collect_env(self, **kwargs):
        cpu_info = get_cpu_info()
        cpu_num = cpu_info.get("CPU(s)", None)
        cpu_model_name = cpu_info.get("Model name", None)
        record(f"0500 CPU 型号：{cpu_model_name}", part=CONTENT_PARTS.sys)
        record(f"0600 CPU 核心数：{cpu_num}", part=CONTENT_PARTS.sys)
        mindie_version = ""
        ascend_toolkit_version = ""

        page_size = os.sysconf("SC_PAGESIZE")
        record(f"0800 页表大小：{page_size}", part=CONTENT_PARTS.sys)

        ascend_toolkit_home = os.getenv("ASCEND_TOOLKIT_HOME")
        ascend_toolkit_version_file = os.path.join(ascend_toolkit_home, "version.cfg") if ascend_toolkit_home else None
        if ascend_toolkit_version_file and os.path.exists(ascend_toolkit_version_file):
            with open(ascend_toolkit_version_file) as ff:
                for line in ff.readlines():
                    if "=" in line:
                        ascend_toolkit_version = line.split("=")[-1].strip()
                        break
            record(f"0100 CANN 版本：{ascend_toolkit_version[1:-1]}", part=CONTENT_PARTS.sys)

        mies_install_path = os.getenv("MIES_INSTALL_PATH")
        mindie_version_file = os.path.join(mies_install_path, "version.info") if mies_install_path else None
        if mindie_version_file and os.path.exists(mindie_version_file):
            with open(mindie_version_file) as ff:
                for line in ff.readlines():
                    if "Ascend-mindie-service" in line and ":" in line:
                        mindie_version = line.split(":")[-1].strip()
                        break
            record(f"0200 MINDIE 版本：{mindie_version}", part=CONTENT_PARTS.sys)
        return dict(
            cpu_model_name=cpu_model_name,
            cpu_num=cpu_num,
            page_size=page_size,
            ascend_toolkit_version=ascend_toolkit_version,
            mindie_version=mindie_version,
        )


class KernelReleaseChecker(RrecheckerBase):
    __checker_name__ = "KernelRelease"

    def collect_env(self, **kwargs):
        kernel_release = platform.release()
        logger.debug(f"Got kernel_release: {kernel_release}")
        record(f"0400 Linux 内核版本：{kernel_release}", part=CONTENT_PARTS.sys)

        kernel_release_split = kernel_release.split(".")
        if len(kernel_release_split) < 2:
            logger.warning(f"failed parsing kernel release version: {kernel_release}")
            return None

        major_version, minor_version = str_to_digit(kernel_release_split[0]), str_to_digit(kernel_release_split[1])
        if major_version is None or minor_version is None:
            logger.warning(f"failed parsing kernel release version: {kernel_release}")
            return None
        return (major_version, minor_version)

    def do_precheck(self, release_version, **kwargs):
        if release_version is None:
            return
        target_major_version, target_minor_version = 5, 10
        target_version = ".".join([str(ii) for ii in [target_major_version, target_minor_version]])
        logger.debug(f"kernel_release suggested is {target_version}")

        major_version, minor_version = release_version

        answer_kwargs = dict(
            domain="system",
            checker="内核版本",
            result=CheckResult.ERROR,
            action=f"升级到 {target_version} 以上",
            reason="建议升级到 5.10 以上，cpu下发加快，减少 host bound",
        )
        if major_version < target_major_version:
            show_check_result(**answer_kwargs)
        elif major_version == target_major_version and minor_version < target_minor_version:
            show_check_result(**answer_kwargs)
        else:
            show_check_result("system", "内核版本", CheckResult.OK)


class DriverVersionChecker(RrecheckerBase):
    __checker_name__ = "DriverVersion"

    def collect_env(self, **kwargs):
        if not os.path.exists(DRIVER_VERSION_PATH) or not os.access(DRIVER_VERSION_PATH, os.R_OK):
            logger.warning(f"{DRIVER_VERSION_PATH} not accessible")
            return None

        version = ""
        with open(DRIVER_VERSION_PATH) as ff:
            for line in ff.readlines():
                if "Version=" in line:
                    version = line.strip().split("=")[-1]
                    break
        logger.debug(f"Got driver version: {version}")
        record(f"0300 驱动版本：{version}", part=CONTENT_PARTS.sys)

        version_split = version.split(".")
        if len(version_split) < 3:
            logger.warning(f"failed parsing Ascend driver version: {version}")
            return None
        major_version, minor_version = str_to_digit(version_split[0]), str_to_digit(version_split[1])
        mini_version = str_to_digit(version_split[2], default_value=-1)  # value like "rc1" convert to -1
        if major_version is None or minor_version is None:
            logger.warning(f"failed parsing Ascend driver version: {version}")
            return None
        return dict(major_version=major_version, minor_version=minor_version, mini_version=mini_version)

    def do_precheck(self, driver_version, **kwargs):
        if driver_version is None:
            return
        target_major_version, target_minor_version, target_mini_version = 24, 1, 0
        target_version = ".".join([str(ii) for ii in [target_major_version, target_minor_version, target_mini_version]])
        logger.debug(f"suggested is {target_version}")

        major_version = driver_version.get("major_version")
        minor_version = driver_version.get("minor_version")
        mini_version = driver_version.get("mini_version")

        answer_kwargs = dict(
            domain="system",
            checker="驱动版本",
            result=CheckResult.ERROR,
            action=f"升级到 {target_version} 以上",
            reason="建议升级到最新的版本的驱动，性能会有提升",
        )
        if major_version < target_major_version:
            show_check_result(**answer_kwargs)
        elif major_version == target_major_version and minor_version < target_minor_version:
            show_check_result(**answer_kwargs)
        elif (
            major_version == target_major_version
            and minor_version == target_minor_version
            and mini_version < target_mini_version
        ):
            show_check_result(**answer_kwargs)
        else:
            show_check_result("system", "驱动版本", CheckResult.OK)


class VirtualMachineChecker(RrecheckerBase):
    __checker_name__ = "VirtualMachine"

    def collect_env(self, **kwargs):
        if not os.path.exists(CPUINFO_PATH) or not os.access(CPUINFO_PATH, os.R_OK):
            logger.warning(f"{CPUINFO_PATH} not accessible")
            return None

        is_virtual_machine = False
        with open(CPUINFO_PATH) as ff:
            for line in ff.readlines():
                if "hypervisor" in line:
                    is_virtual_machine = True
                    logger.info(f"Got hypervisor info from: {CPUINFO_PATH}")
                    break
        record(f"1000 是否虚拟机：{'是' if is_virtual_machine else '否'}", part=CONTENT_PARTS.sys)
        return is_virtual_machine

    def do_precheck(self, is_virtual_machine, **kwargs):
        if is_virtual_machine is None:
            return
        if is_virtual_machine:
            vmware_action = (
                "启用 CPU/MMU Virtualization（ESXi 高级设置）、禁用 CPU 限制（cpuid.coresPerSocket 配置为物理核心数）"
            )
            kvm_action = "配置 host-passthrough 模式（暴露完整 CPU 指令集）、启用多队列 virtio-net（减少网络延迟）"
            show_check_result(
                "system",
                "可能是虚拟机",
                CheckResult.ERROR,
                action=f"确定分配的 cpu 是完全体，如 VMware 中 {vmware_action}；KVM 中 {kvm_action}",
                reason="虚拟机和物理机的 cpu 核数、频率有差异会导致性能下降，如果是虚拟机环境，建议检查 cpu 情况",
            )


class TransparentHugepageChecker(RrecheckerBase):
    __checker_name__ = "TransparentHugepage"

    def collect_env(self, **kwargs):
        if not os.path.exists(TRANSPARENT_HUGEPAGE_PATH) or not os.access(TRANSPARENT_HUGEPAGE_PATH, os.R_OK):
            logger.warning(f"{TRANSPARENT_HUGEPAGE_PATH} not accessible")
            return None

        is_transparent_hugepage_enable = False
        with open(TRANSPARENT_HUGEPAGE_PATH) as ff:
            for line in ff.readlines():
                if "always" in line:
                    is_transparent_hugepage_enable = True
                    logger.debug(f"Got 'always' from: {TRANSPARENT_HUGEPAGE_PATH}")
                    break
        record(f"0900 是否开启透明大页：{'是' if is_transparent_hugepage_enable else '否'}", part=CONTENT_PARTS.sys)
        return is_transparent_hugepage_enable

    def do_precheck(self, is_transparent_hugepage_enable, **kwargs):
        if is_transparent_hugepage_enable is None:
            return
        if not is_transparent_hugepage_enable:
            show_check_result(
                "system",
                "透明大页",
                CheckResult.ERROR,
                action=f"设置为 always：echo always > {TRANSPARENT_HUGEPAGE_PATH}",
                reason="开启透明大页，吞吐率结果会更稳定",
            )


class CpuHighPerformanceChecker(RrecheckerBase):
    __checker_name__ = "CpuHighPerformance"

    def collect_env(self, **kwargs):
        cpu_count = os.cpu_count()
        is_performances = []
        for core in range(cpu_count):
            cur_governor_path = GOVERNOR_PATH_FORMATTER.format(core=core)
            if not os.path.exists(cur_governor_path) or not os.access(cur_governor_path, os.R_OK):
                continue

            with open(cur_governor_path, "r") as ff:
                for line in ff.readlines():
                    if line.strip() == "performance":
                        is_performances.append(True)
                        break
        performance_count = len(is_performances)
        record(f"0700 CPU 是否高性能模式：{'是' if cpu_count == performance_count else '否'}", part=CONTENT_PARTS.sys)
        return dict(cpu_count=cpu_count, performance_count=performance_count)

    def do_precheck(self, cpu_info, **kwargs):
        if cpu_info is None:
            return
        cpu_count = cpu_info.get("cpu_count")
        performance_count = cpu_info.get("performance_count")

        if performance_count != cpu_count:
            yum_cmd = "EulerOS/CentOS: yum install kernel-tools"
            apt_cmd = "Ubuntu：apt install cpufrequtils"
            run_cmd = "cpupower -c all frequency-set -g performance"
            fail_info = "如果失败可能需要在 BIOS 中开启"
            undo_cmd = "cpupower -c all frequency-set -g powersave"
            show_check_result(
                "system",
                "CPU高性能模式",
                CheckResult.ERROR,
                action=f"开启 CPU 高性能模式：{run_cmd}；\n        "
                f"如果没有 cpupower 命令可以通过 {yum_cmd} 或 {apt_cmd} 安装；\n        "
                f"{fail_info}\n        "
                f"如果需要回退，可以使用命令：{undo_cmd}",
                reason="使 CPU 运行在最大频率下，可以提升CPU性能，但是会提高能耗",
            )


class SystemChecker(GroupRrechecker):
    __checker_name__ = "System"

    def init_sub_checkers(self):
        return [
            SystemInfoCollect(),
            KernelReleaseChecker(),
            DriverVersionChecker(),
            VirtualMachineChecker(),
            TransparentHugepageChecker(),
            CpuHighPerformanceChecker(),
        ]


system_checker = SystemChecker()
