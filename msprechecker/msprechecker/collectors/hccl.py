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

# pylint: disable=duplicate-code

import ipaddress
import os
import shlex
import shutil
import subprocess  # nosec B404
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Union

from packaging.version import Version

from ..utils import get_npu_count
from ..utils import is_in_container
from .base import BaseCollector

HCCN_TOOL_CMD = Path("/usr/local/Ascend/driver/tools/hccn_tool").resolve()
_HCCN_TOOL_AVAILABLE = HCCN_TOOL_CMD.is_file() and os.access(HCCN_TOOL_CMD, os.X_OK)


class HCCNCollector(BaseCollector):
    CMD_NAME = ""

    def __init__(self, error_handler=None):
        super().__init__(error_handler)
        self.npu_count = get_npu_count()

    @abstractmethod
    def _generate_cmd(self):
        pass

    def _run_cmd(self, cmd: str):
        output = None
        try:
            output = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL, text=True)  # nosec B603
        except Exception:
            output = "100% packet loss"

        return output

    def _collect_data(self):
        if not _HCCN_TOOL_AVAILABLE:
            working_place = "宿主机" if not is_in_container() else "容器"
            self.error_handler.add_error(
                filename=__file__,
                function="_collect_data",
                lineno=55,
                what=f"{working_place}上没有找到 'hccn_tool' 命令或不可执行",
                reason=f"{working_place}上没有找到 'hccn_tool' 命令或不可执行",
            )
            return {}

        cmds = list(self._generate_cmd())
        max_workers = min(len(cmds), os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(self._run_cmd, cmd) for cmd in cmds]

        return [future.result() for future in futures]


class VnicCollector(HCCNCollector):
    CMD_NAME = "vnic"

    def _generate_cmd(self):
        for device_id in range(self.npu_count):
            yield f"{HCCN_TOOL_CMD} -i {device_id} -{self.CMD_NAME} -g"


class LinkCollector(HCCNCollector):
    CMD_NAME = "link"

    def _generate_cmd(self):
        for device_id in range(self.npu_count):
            yield f"{HCCN_TOOL_CMD} -i {device_id} -{self.CMD_NAME} -g"


class TlsCollector(HCCNCollector):
    CMD_NAME = "tls"

    def _generate_cmd(self):
        for device_id in range(self.npu_count):
            yield f"{HCCN_TOOL_CMD} -i {device_id} -{self.CMD_NAME} -g"


class HCCLCollector(BaseCollector):
    CMD_NAME = "ping"

    def __init__(self, rank_table, npu_count=None):
        super().__init__()
        self.rank_table = rank_table
        self.npu_count = npu_count or get_npu_count()
        self.option = "-hccs_ping" if getattr(rank_table, "version", Version("1.0")) >= Version("1.2") else "-ping"

    def _run_cmd(self, device_id: int, device_ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]):
        if device_ip.version == 4:
            cmd = f"{HCCN_TOOL_CMD} -i {device_id} {self.option} -g address {device_ip}"
        elif device_ip.version == 6:
            cmd = f"{HCCN_TOOL_CMD} -i {device_id} {self.option} -inet6 -g ipv6_address {device_ip}"
        else:
            raise ValueError(f"Invalid IP version: {device_ip.version}")

        with subprocess.Popen(  # nosec B603
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as proc:
            output = proc.stdout.read()
            ret = proc.wait()
        return cmd, ret, output

    def _run_per_device(self, device_id, device_ips):
        """Each device has max concurrency 1"""
        result = {}
        for device_ip in device_ips:
            cmd, ret, output = self._run_cmd(device_id, device_ip)
            result[cmd] = ret, output

        return result

    def _collect_data(self):
        if not shutil.which(HCCN_TOOL_CMD):
            working_place = "宿主机" if not is_in_container() else "容器"
            self.error_handler.add_error(
                filename=__file__,
                function="_collect_data",
                lineno=63,
                what=f"{working_place}上没有找到 'hccn_tool' 命令",
                reason=f"[Errno 2] No such file or directory: '{HCCN_TOOL_CMD}'",
            )
            return {}

        max_workers = min(self.npu_count, os.cpu_count() or 1)  # each device can proceed parallel

        all_device_ips = (
            device_info.device_ip
            for device_info_list in self.rank_table.host_to_devices.values()
            for device_info in device_info_list
        )

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [
                executor.submit(self._run_per_device, device_id, all_device_ips) for device_id in range(self.npu_count)
            ]

            results = {}
            for future in as_completed(futures):
                results.update(future.result())

        return results
