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
import itertools
import json
import os
import re
import shlex
import stat
import subprocess  # nosec B404
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

from packaging.version import InvalidVersion, Version

from .log import global_logger
from .path_io import to_user_path


class NpuType(Enum):
    TP_300 = "d100"
    TP_300I_DUO = "d500"
    TP_A1 = "d801"
    TP_A2 = "d802"
    TP_A3 = "d803"

    @property
    def display(self):
        return {
            NpuType.TP_300: "300",
            NpuType.TP_300I_DUO: "300I Duo",
            NpuType.TP_A1: "800I A1",
            NpuType.TP_A2: "800I A2",
            NpuType.TP_A3: "800I A3",
        }[self]


def get_npu_count():
    davinci_path_template = "/dev/davinci{}"
    device_id = -1

    for device_id in itertools.count(0):
        device_path = davinci_path_template.format(device_id)
        try:
            f_mode = os.stat(device_path).st_mode
        except Exception:
            break

        if not stat.S_ISCHR(f_mode):
            break

    return device_id


def get_npu_type():
    try:
        output = subprocess.check_output(["/usr/bin/lspci"], stderr=subprocess.DEVNULL, text=True)  # nosec B603
    except Exception:
        return None, None

    device_pattern = re.compile(r"device\s*(\w+)", re.IGNORECASE)
    device_list = []
    zip_engine_nums = 0
    for line in output.split("\n"):
        if "accelerator" in line:
            m = device_pattern.search(line)
            if not m:
                zip_engine_nums += 1  # d803 contains extra zip engines
                continue
            device_list.append(m.group(1))

    if not device_list:
        return None, None

    if any(device != device_list[0] for device in device_list):
        return None, None

    npu_type = device_list[0]
    if npu_type not in (member.value for member in NpuType):
        return None, None

    return NpuType(npu_type), len(device_list) - zip_engine_nums


def get_conn_mode():
    lldp_cmd = "hccn_tool -i 0 -lldp -g"
    try:
        output = subprocess.check_output(shlex.split(lldp_cmd), stderr=subprocess.DEVNULL, text=True)  # nosec B603
    except Exception:
        return None

    if not output:
        return None

    fields = output.split("\n")
    tlv_desc = "System Description TLV"
    if tlv_desc not in fields:
        return None

    desc_idx = fields.index(tlv_desc)
    if desc_idx >= len(fields):
        return None

    route_fields = "Routing"
    fiber_fields = "AscendNPU"
    if route_fields in fields[desc_idx + 1]:
        return "route"

    if fiber_fields in fields[desc_idx + 1]:
        return "fiber"

    return None


class Framework(Enum):
    MINDIE = "mindie"
    VLLM = "vllm"
    SGLANG = "sglang"
    UNKNOWN = "unknown"


@dataclass
class DeviceInfo:
    device_ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
    device_id: int
    rank_id: int


@dataclass
class RankTable:
    host_to_devices: dict[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], list[DeviceInfo]]
    server_count: int
    version: Version


class RankTableParseError(ValueError):
    """Raised when a rank table file exists but cannot be parsed correctly."""


class WeightDirNotFoundError(FileNotFoundError):
    """Raised when the weight directory cannot be located from config/script."""


_HOST_LIMIT = 1000
_DEVICE_LIMIT_PER_HOST = 32


def _load_json(path: str) -> dict:
    """Load and return JSON from *path*; raise RankTableParseError on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise RankTableParseError(f"Failed to load JSON from {to_user_path(path)}") from exc


def _parse_server_count(server_count: Union[int, str]) -> int:
    """Parse server_count from rank table."""
    if isinstance(server_count, int):
        return server_count
    if isinstance(server_count, str) and server_count.isdigit():
        return int(server_count)
    global_logger.warning("Unexpected server_count %r; defaulting to 0", server_count)
    return 0


def _parse_mindie_host_to_devices(
    server_list: list[dict[str, Union[str, list[dict[str, str]]]]],
) -> dict[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], list[DeviceInfo]]:
    """Parse host_to_devices from mindie rank table."""
    host_to_devices: dict[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], list[DeviceInfo]] = {}

    if not server_list:
        global_logger.warning("Expected server_list in rank table but not found")
        return host_to_devices

    for host_num, server_info in enumerate(server_list):
        if host_num >= _HOST_LIMIT:
            raise RankTableParseError(f"Host count exceeds limit {_HOST_LIMIT}")

        host_ip_str = server_info.get("server_id", "")
        device_list = server_info.get("device", [])

        if not host_ip_str:
            global_logger.warning("Expected server_id in server_list but not found, skipping")
            continue

        if not device_list:
            global_logger.warning("Expected list of devices in server_list but not found, skipping")
            continue

        try:
            host_ip = ipaddress.ip_address(host_ip_str)
        except ValueError:
            global_logger.warning("Invalid server_id %r found in server_list, skipping", host_ip_str)
            continue

        if host_ip not in host_to_devices:
            host_to_devices[host_ip] = []

        for dev_num, dev_info in enumerate(device_list):
            if dev_num >= _DEVICE_LIMIT_PER_HOST:
                raise RankTableParseError(
                    f"Device count for host {host_ip_str!r} exceeds limit {_DEVICE_LIMIT_PER_HOST}"
                )

            device_ip_str = dev_info.get("device_ip", "")
            device_id_str = dev_info.get("device_id", "")
            rank_id_str = dev_info.get("rank_id", "")

            try:
                device_ip = ipaddress.ip_address(device_ip_str)
            except ValueError:
                global_logger.warning("Invalid device_ip %r for %r; skipping", device_ip_str, host_ip_str)
                continue

            try:
                device_id = int(device_id_str)
            except ValueError:
                global_logger.warning("Invalid device_id %r for %r; skipping", device_id_str, host_ip_str)
                continue

            try:
                rank_id = int(rank_id_str)
            except ValueError:
                global_logger.warning("Invalid rank_id %r for %r; skipping", rank_id_str, host_ip_str)
                continue

            host_to_devices[host_ip].append(
                DeviceInfo(
                    device_ip=device_ip,
                    device_id=device_id,
                    rank_id=rank_id,
                )
            )

    return host_to_devices


def _parse_vllm_host_to_devices(
    prefill_device_list, decode_device_list
) -> dict[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], list[DeviceInfo]]:
    """Parse host_to_devices from vllm rank table."""
    host_to_devices: dict[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], list[DeviceInfo]] = {}

    for list_name, device_list in (
        ("prefill_device_list", prefill_device_list),
        ("decode_device_list", decode_device_list),
    ):
        if device_list is None:
            global_logger.warning("Expected %r in rank table but not found", list_name)
            continue

        if len(device_list) > _HOST_LIMIT * _DEVICE_LIMIT_PER_HOST:
            raise RankTableParseError(f"{list_name!r} length exceeds limit {_HOST_LIMIT * _DEVICE_LIMIT_PER_HOST}")

        for dev in device_list:
            host_ip_str = dev.get("server_id", "")
            try:
                host_ip = ipaddress.ip_address(host_ip_str)
            except ValueError:
                global_logger.warning("Invalid server_id %r; skipping", host_ip_str)
                continue

            if host_ip not in host_to_devices:
                if len(host_to_devices) >= _HOST_LIMIT:
                    raise RankTableParseError(f"Host count exceeds limit {_HOST_LIMIT}")
                host_to_devices[host_ip] = []

            if len(host_to_devices[host_ip]) >= _DEVICE_LIMIT_PER_HOST:
                raise RankTableParseError(
                    f"Device count for host {host_ip_str!r} exceeds limit {_DEVICE_LIMIT_PER_HOST}"
                )

            device_ip_str = dev.get("device_ip", "")
            device_id_str = dev.get("device_id", "")
            cluster_id_str = dev.get("cluster_id", "")

            try:
                device_ip = ipaddress.ip_address(device_ip_str)
            except ValueError:
                global_logger.warning("Invalid device_ip %r for %r; skipping", device_ip_str, host_ip_str)
                continue

            try:
                device_id = int(device_id_str)
            except ValueError:
                global_logger.warning("Invalid device_id %r for %r; skipping", device_id_str, host_ip_str)
                continue

            try:
                cluster_id = int(cluster_id_str)
            except ValueError:
                global_logger.warning(
                    "Invalid cluster_id %r for %r; skipping",
                    cluster_id_str,
                    host_ip_str,
                )
                continue

            host_to_devices[host_ip].append(
                DeviceInfo(
                    device_ip=device_ip,
                    device_id=device_id,
                    rank_id=cluster_id - 1,  # vllm cluster_id is 1-based
                )
            )
    return host_to_devices


def _parse_mindie(path: str) -> RankTable:
    """Parse rank table in MindIE format."""
    data = _load_json(path)

    if "server_list" not in data:
        raise RankTableParseError(f"'server_list' not found in rank table: {to_user_path(path)}")

    if "server_count" not in data:
        raise RankTableParseError(f"'server_count' not found in rank table: {to_user_path(path)}")

    host_to_devices = _parse_mindie_host_to_devices(data["server_list"])

    if not host_to_devices:
        raise RankTableParseError(f"No devices found in rank table: {to_user_path(path)}")

    server_count = _parse_server_count(data["server_count"])
    version_str = data.get("version", "1.0")  # version is optional
    try:
        version = Version(version_str)
    except InvalidVersion as e:
        raise RankTableParseError(f"Invalid version {version_str!r} found in {to_user_path(path)}") from e

    return RankTable(
        host_to_devices=host_to_devices,
        server_count=server_count,
        version=version,
    )


def _parse_vllm(path: str) -> RankTable:
    """Parse rank table in VLLM format."""
    data = _load_json(path)

    if "prefill_device_list" not in data or "decode_device_list" not in data:
        raise RankTableParseError(
            f"Expected 'prefill_device_list' and 'decode_device_list' in rank table: {to_user_path(path)}"
        )

    host_to_devices = _parse_vllm_host_to_devices(data["prefill_device_list"], data["decode_device_list"])

    if not host_to_devices:
        raise RankTableParseError(f"No devices found in rank table: {to_user_path(path)}")

    server_count = _parse_server_count(data.get("server_count"))

    version_str = data.get("version", "1.0")  # version is optional
    try:
        version = Version(version_str)
    except InvalidVersion as e:
        raise RankTableParseError(f"Invalid version {version_str!r} found in {to_user_path(path)}") from e

    return RankTable(
        host_to_devices=host_to_devices,
        server_count=server_count,
        version=version,
    )


_RANK_TABLE_PARSERS: dict[Framework, Callable[[str], RankTable]] = {
    Framework.MINDIE: _parse_mindie,
    Framework.VLLM: _parse_vllm,
}


def parse_rank_table(path: str, framework: Framework) -> RankTable:
    """Parse a rank table file for the given framework.

    Currently supported frameworks: MINDIE, VLLM.
    SGLang does not define a rank table format and is intentionally unsupported.

    Args:
        path (str): Path to the rank table JSON file.
        framework (Framework): Determines the parse strategy.

    Returns:
        Parsed RankTable.

    Raises:
        RankTableParseError: File exists but cannot be parsed.
        ValueError: Framework is not supported.
    """
    parser = _RANK_TABLE_PARSERS.get(framework)
    if parser is None:
        raise ValueError(f"No rank table parser for {framework!r}. Supported: {list(_RANK_TABLE_PARSERS)}")
    return parser(path)


model_type = None


def update_model_type(args):
    weight_dir = None
    if getattr(args, "weight_dir", None):
        weight_dir = args.weight_dir
    elif getattr(args, "mies_config_path", None):
        with Path(args.mies_config_path).open(encoding="utf-8") as f:
            data = json.load(f)
        try:
            weight_dir = data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelWeightPath"]
        except Exception:
            weight_dir = None

    if not weight_dir:
        return

    model_config_path = os.path.join(weight_dir, "config.json")

    global model_type
    try:
        with Path(model_config_path).open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        model_type = None
    else:
        model_type = data.get("model_type")


def get_model_type():
    return model_type
