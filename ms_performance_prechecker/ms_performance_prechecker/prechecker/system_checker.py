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
import numpy as np
from ms_performance_prechecker.prechecker.register import register_checker, cached, answer
from ms_performance_prechecker.prechecker.utils import CHECK_TYPES, SUGGESTION_TYPES, get_dict_value_by_pos, str_to_digit

DRIVER_VERSION_PATH = "/usr/local/Ascend/driver/version.info"
CPUINFO_PATH = "/proc/cpuinfo"
TRANSPARENT_HUGEPAGE_PATH = "/sys/kernel/mm/transparent_hugepage/enabled"
GOVERNOR_PATH_FORMATTER = "/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor"

@register_checker()
def linux_kernel_release_checker(mindie_service_config, check_type):
    target_major_version, target_minor_version = 5, 10
    target_version = ".".join([str(ii) for ii in [target_major_version, target_minor_version]])


    kernel_release = platform.release()
    kernel_release_split = kernel_release.split(".")
    if len(kernel_release_split) < 2:
        print(f"[ERROR] failed parsing kernel release version: {kernel_release}")
        return

    major_version, minor_version = str_to_digit(kernel_release_split[0]), str_to_digit(kernel_release_split[1])
    if major_version is None or minor_version is None:
        print(f"[ERROR] failed parsing kernel release version: {kernel_release}")
        return

    if major_version < target_major_version:
        answer(suggesion_type=SUGGESTION_TYPES.system, suggesion_item="内核版本", action=f"升级到 {target_version} 以上", reason="内核版本升级后以上 host bound 时性能有提升")
    if major_version == target_major_version and minor_version < target_minor_version:
        answer(suggesion_type=SUGGESTION_TYPES.system, suggesion_item="内核版本", action=f"升级到 {target_version} 以上", reason="内核版本升级后以上 host bound 时性能有提升")


@register_checker()
def driver_version_checker(mindie_service_config, check_type):
    target_major_version, target_minor_version, target_mini_version = 24, 1, 0
    target_version = ".".join([str(ii) for ii in [target_major_version, target_minor_version, target_mini_version]])

    if not os.path.exists(DRIVER_VERSION_PATH) or not os.access(DRIVER_VERSION_PATH, os.R_OK):
        print(f"[ERROR] {DRIVER_VERSION_PATH} not accessible")
        return

    version = ""
    with open(DRIVER_VERSION_PATH) as ff:
        for line in ff.readlines():
            if "Version=" in line:
                version = line.strip().split("=")[-1]
                break
    version_split = version.split(".")
    if len(version_split) < 3:
        print(f"[ERROR] failed parsing Ascend driver version: {version}")
        return
    major_version, minor_version = str_to_digit(version_split[0]), str_to_digit(version_split[1])
    mini_version = str_to_digit(version_split[2], default_value=-1)  # value like "rc1" convert to -1
    if major_version is None or minor_version is None:
        print(f"[ERROR] failed parsing Ascend driver version: {version}")
        return

    if major_version < target_major_version:
        answer(suggesion_type=SUGGESTION_TYPES.system, suggesion_item="驱动版本", action=f"升级到 {target_version} 以上", reason="驱动版本升级后性能有提升")
    if major_version == target_major_version and minor_version < target_minor_version:
        answer(suggesion_type=SUGGESTION_TYPES.system, suggesion_item="内核版本", action=f"升级到 {target_version} 以上", reason="驱动版本升级后性能有提升")
    if major_version == target_major_version and minor_version == target_minor_version and mini_version < target_mini_version:
        answer(suggesion_type=SUGGESTION_TYPES.system, suggesion_item="内核版本", action=f"升级到 {target_version} 以上", reason="驱动版本升级后性能有提升")


@register_checker()
def virtual_machine_checker(mindie_service_config, check_type):
    if not os.path.exists(CPUINFO_PATH) or not os.access(CPUINFO_PATH, os.R_OK):
        print(f"[ERROR] {CPUINFO_PATH} not accessible")
        return

    is_virtual_machine = False
    with open(CPUINFO_PATH) as ff:
        for line in ff.readlines():
            if "hypervisor" in line or "vmx" in line or "svm" in line:
                is_virtual_machine = True
                break
    if is_virtual_machine:
        answer(suggesion_type=SUGGESTION_TYPES.system, suggesion_item="虚拟机", action="确定分配的 cpu 是完全体", reason="虚拟机和物理机的 cpu 核数、频率有差异会导致性能下降")



@register_checker()
def transparent_hugepage_checker(mindie_service_config, check_type):
    if not os.path.exists(TRANSPARENT_HUGEPAGE_PATH) or not os.access(TRANSPARENT_HUGEPAGE_PATH, os.R_OK):
        print(f"[ERROR] {TRANSPARENT_HUGEPAGE_PATH} not accessible")
        return

    is_transparent_hugepage_enable = False
    with open(TRANSPARENT_HUGEPAGE_PATH) as ff:
        for line in ff.readlines():
            if "always " in line:
                is_transparent_hugepage_enable = True
                break
    if not is_transparent_hugepage_enable:
        answer(suggesion_type=SUGGESTION_TYPES.system, suggesion_item="透明大页", action=f"设置为 always：echo always > {TRANSPARENT_HUGEPAGE_PATH}", reason="开启透明大页，多次实验的吞吐率结果会更稳定")


@register_checker()
def cpu_high_performance_checker(mindie_service_config, check_type):
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
    if len(is_performances) != cpu_count:
        answer(suggesion_type=SUGGESTION_TYPES.system, suggesion_item="CPU高性能模式", action="开启 CPU 高性能模式：cpupower -c all frequency-set -g performance", reason="在相同时延约束下，TPS会有~3%的提升")
