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
import logging
import shlex
import shutil
import subprocess
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Any, Dict

from .base_strategy import CollectStrategyGroup, CollectStrategy
from ..utils import RankTable, get_npu_count, Utils, LOGGER


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
            if LOGGER.isEnabledFor(logging.WARNING):
                LOGGER.exception("Failed to execute command: %s", cmd)
            return None

    @abstractmethod
    def execute(self) -> List[Any]:
        """Execute the hccn_tool command for each device and return a list of results."""


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


class HcclPing(_OptionIp):
    option = "-ping"
    default_name = "hccl_ping"


class HccsPing(_OptionIp):
    option = "-hccs_ping"
    default_name = "hccs_ping"


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


class Tls(_SingleOption):
    option = "-tls"
    default_name = "tls"


class Ping(CollectStrategy):
    def __init__(self, name: str = "ping", *, ip: str) -> None:
        if not isinstance(ip, str):
            raise TypeError(f"IP address must be a string: {ip!r}")
        if not Utils.is_valid_ip(ip):
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
            LOGGER.warning(
                "Failed to execute ping command for IP %s", self._ip, exc_info=True
            )
            return None

    def execute(self) -> Optional[str]:
        if self._ping_path is None:
            LOGGER.warning("ping command not found in system PATH")
            return None

        return self._ping_ip()


class Vnic(_SingleOption):
    option = "-vnic"
    default_name = "vnic"


class Link(_SingleOption):
    option = "-link"
    default_name = "link"