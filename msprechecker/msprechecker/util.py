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

import dataclasses
import ipaddress
import itertools
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess  # nosec B404
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Optional, Union

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version
import socket

from packaging.version import InvalidVersion, Version

from msprechecker.utils.path_io import to_user_path

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
LOG_FORMAT = "%(levelname)-5s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"


logger = logging.getLogger(__name__)


class RankTableParseError(ValueError):
    """Raised when a rank table file exists but cannot be parsed correctly."""


class WeightDirNotFoundError(FileNotFoundError):
    """Raised when the weight directory cannot be located from config/script."""


class NpuType(Enum):
    d100 = "300"
    d500 = "300I_DUO"
    d801 = "800I A1"
    d802 = "800I A2"
    d803 = "800I A3"
    UNKNOWN = "unknown"


class ConnMode(Enum):
    ROUTE = "route"
    FIBER = "fiber"
    UNKNOWN = "unknown"


class Framework(Enum):
    MINDIE = "mindie"
    VLLM = "vllm"
    SGLANG = "sglang"
    UNKNOWN = "unknown"


class DeployMode(Enum):
    PD_MIX = "pd_mix"
    PD_DISAGGREGATION = "pd_disaggregation"
    PD_DISAGGREGATION_SINGLE_CONTAINER = "pd_disaggregation_single_container"
    EP = "ep"
    LWD = "lwd"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    npu_count: int
    npu_type: NpuType
    npu_memory_mb: Optional[int]
    conn_mode: ConnMode


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    device_ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
    device_id: int
    rank_id: int


@dataclasses.dataclass(frozen=True)
class RankTable:
    host_to_devices: dict[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], list[DeviceInfo]]
    server_count: int
    version: Version


def deploy_mode_from_precheck_scene(scene: Optional[str]) -> DeployMode:
    """Map ``msprechecker precheck --scene`` to :class:`DeployMode`.

    Comma-separated values are supported (e.g. ``mindie,ep`` selects EP).

    Args:
        scene: Raw ``--scene`` string, or ``None`` for default mixed mode.

    Returns:
        :data:`DeployMode.PD_MIX` when *scene* is empty; :data:`DeployMode.EP`
        when ``ep`` appears as a comma-separated token; otherwise the enum
        member matching the trimmed string, or :data:`DeployMode.UNKNOWN`.
    """
    if not scene:
        return DeployMode.PD_MIX
    parts = [p.strip().lower() for p in scene.split(",") if p.strip()]
    if "ep" in parts:
        return DeployMode.EP
    try:
        return DeployMode(scene.strip())
    except ValueError:
        return DeployMode.UNKNOWN


def get_pkg_version(pkg_name: str) -> Optional[Version]:
    try:
        # package version if exists must obey PEP 440
        return Version(version(pkg_name))
    except PackageNotFoundError:
        return None


def parse_version_heuristic(version_str: str) -> Version:
    """Parse kernel version or ascend driver version to Version object.

    The version string is expected to be in the format of "X.Y.Z" or "X.Y".
    If the version string is not in the format of "X.Y.Z" or "X.Y", raise InvalidVersion.
    """
    mo = re.search(r"\d+(?:\.\d+){0,2}", version_str)
    if not mo:
        raise InvalidVersion(f"Invalid version: {version_str!r}") from None
    return Version(mo.group(0))


def get_npu_count() -> int:
    """Count Davinci NPU character devices present on the machine."""
    template = "/dev/davinci{}"
    for device_id in itertools.count(0):
        try:
            mode = os.stat(template.format(device_id)).st_mode
        except OSError:
            # Device path doesn't exist: we've counted all devices.
            return device_id
        if not stat.S_ISCHR(mode):
            # Path exists but is not a character device: stop here.
            return device_id
    return 0  # unreachable; satisfies type checkers


def get_npu_type() -> NpuType:
    """Detect NPU type via lspci accelerator entries."""
    lspci = shutil.which("lspci")
    if not lspci:
        return NpuType.UNKNOWN

    try:
        output = subprocess.check_output([lspci], stderr=subprocess.DEVNULL, text=True)  # nosec B603
    except Exception:
        logger.exception("Failed to execute lspci")
        return NpuType.UNKNOWN

    device_pattern = re.compile(r"device\s*(\w+)", re.IGNORECASE)
    devices = []
    for line in output.splitlines():
        if "accelerator" not in line.lower():
            continue
        m = device_pattern.search(line)
        if m:
            devices.append(m.group(1))

    if not devices:
        return NpuType.UNKNOWN

    first = devices[0]
    if first not in NpuType.__members__ or any(d != first for d in devices):
        logger.debug("Inconsistent or unrecognised device types: %s", devices)
        return NpuType.UNKNOWN

    return NpuType[first]


def get_npu_memory() -> Optional[int]:
    """Return High-Bandwidth Memory capacity (MB) for device 0, or None if unavailable."""
    npu_smi = shutil.which("npu-smi")
    if not npu_smi:
        return None

    try:
        output = subprocess.check_output(  # nosec B603
            [npu_smi, "info", "-i", "0", "-t", "memory"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        logger.exception("Failed to execute npu-smi")
        return None

    for line in output.splitlines():
        if "HB" + "M Capacity" not in line or ":" not in line:
            continue
        value = line.split(":")[-1].strip()
        if value.isdigit():
            return int(value)

    return None


def get_conn_mode() -> ConnMode:
    """Detect NPU connection mode via hccn_tool LLDP output."""
    hccn_tool = shutil.which("hccn_tool")
    if not hccn_tool:
        return ConnMode.UNKNOWN

    try:
        output = subprocess.check_output(  # nosec B603
            shlex.split(f"{hccn_tool} -i 0 -lldp -g"),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        logger.exception("Failed to execute hccn_tool")
        return ConnMode.UNKNOWN

    if not output:
        return ConnMode.UNKNOWN

    lines = output.splitlines()
    try:
        idx = lines.index("System Description TLV")
    except ValueError:
        return ConnMode.UNKNOWN

    if idx + 1 >= len(lines):
        return ConnMode.UNKNOWN

    next_line = lines[idx + 1]
    if "Routing" in next_line:
        return ConnMode.ROUTE
    if "AscendNPU" in next_line:
        return ConnMode.FIBER
    return ConnMode.UNKNOWN


def probe_hardware() -> HardwareProfile:
    """
    Probe all hardware characteristics in one call.

    The caller is responsible for caching the result if repeated
    subprocess calls are undesirable.
    """
    return HardwareProfile(
        npu_count=get_npu_count(),
        npu_type=get_npu_type(),
        npu_memory_mb=get_npu_memory(),
        conn_mode=get_conn_mode(),
    )


def is_in_container():
    def check_docker_env_file():
        docker_env_file = "/.dockerenv"
        return os.path.exists(docker_env_file)

    def check_first_process():
        first_proc = "/proc/1"
        schedule_file = os.path.join(first_proc, "sched")

        try:
            with open(schedule_file, encoding="utf-8", errors="replace") as f:
                first_line = f.readlines(1)
        except Exception:
            return True

        if first_line and first_line[0] and first_line[0].startswith("systemd"):
            return False

        return True

    return check_docker_env_file() or check_first_process()


# Shown when CPU high-performance checks all fail inside a container: host may still
# be tuned correctly but dmidecode/tools are missing or sysfs is misleading.
CONTAINER_CPU_HIGH_PERF_AMBIGUITY_HINT = (
    "容器内五项 CPU 高性能检测均未通过，无法判断宿主机是否已开启高性能；若在宿主机已开启，"
    "请执行 df -h 查看挂载路径，将宿主机的 dmidecode 拷贝到容器可访问的目录，必要时将该目录加入 "
    "PATH 后重试。"
)


def get_current_ip_and_addr():
    import psutil

    for interface, addrs in psutil.net_if_addrs().items():
        if any(interface.startswith(prefix) for prefix in ("docker", "lo")):
            continue
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith("127"):
                return interface, addr.address
    return "", ""


# ---------------------------------------------------------------------------
# Framework detection — Protocol-based, stateless, composable
# ---------------------------------------------------------------------------


@runtime_checkable
class FrameworkProbe(Protocol):  # pylint: disable=unnecessary-ellipsis
    """Pluggable strategy for detecting a single inference framework."""

    def probe(self) -> Optional[Framework]:
        """Return the detected Framework, or None if not detected."""
        ...  # pylint: disable=unnecessary-ellipsis


class MindIEProbe:
    def probe(self) -> Optional[Framework]:
        if os.path.isdir("/usr/local/Ascend/mindie"):
            logger.debug("MindIE detected via install directory")
            return Framework.MINDIE
        if any("MINDIE" in k for k in os.environ):
            logger.debug("MindIE detected via environment variable")
            return Framework.MINDIE
        return None


class VLLMProbe:
    def probe(self) -> Optional[Framework]:
        if os.path.isdir("/vllm-workspace/vllm-ascend"):
            logger.debug("vLLM detected via workspace directory")
            return Framework.VLLM
        try:
            import vllm_ascend  # noqa: F401

            logger.debug("vLLM detected via import")
            return Framework.VLLM
        except ImportError:
            return None


class SGLangProbe:
    def probe(self) -> Optional[Framework]:
        try:
            import sglang  # noqa: F401

            logger.debug("SGLang detected via import")
            return Framework.SGLANG
        except ImportError:
            return None


DEFAULT_PROBES: tuple[FrameworkProbe, ...] = (
    MindIEProbe(),
    VLLMProbe(),
    SGLangProbe(),
)


def detect_framework(
    probes: tuple[FrameworkProbe, ...] = DEFAULT_PROBES,
) -> Framework:
    """
    Run each probe in order; return the first match.

    This function is stateless. The caller decides whether to cache the
    result (e.g. in a module-level variable or application context object).

    Args:
        probes: Ordered tuple of FrameworkProbe instances.

    Returns:
        Detected Framework, or Framework.UNKNOWN if none matched.
    """
    for probe in probes:
        result = probe.probe()
        if result is not None:
            return result
    logger.debug("No framework detected, returning UNKNOWN")
    return Framework.UNKNOWN


# Rank table parsing

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
    logger.warning("Unexpected server_count %r; defaulting to 0", server_count)
    return 0


def _parse_mindie_host_to_devices(
    server_list: list[dict[str, Union[str, list[dict[str, str]]]]],
) -> dict[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], list[DeviceInfo]]:
    """Parse host_to_devices from mindie rank table."""
    host_to_devices: dict[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], list[DeviceInfo]] = {}

    if not server_list:
        logger.warning("Expected server_list in rank table but not found")
        return host_to_devices

    for host_num, server_info in enumerate(server_list):
        if host_num >= _HOST_LIMIT:
            raise RankTableParseError(f"Host count exceeds limit {_HOST_LIMIT}")

        host_ip_str = server_info.get("server_id", "")
        device_list = server_info.get("device", [])

        if not host_ip_str:
            logger.warning("Expected server_id in server_list but not found, skipping")
            continue

        if not device_list:
            logger.warning("Expected list of devices in server_list but not found, skipping")
            continue

        try:
            host_ip = ipaddress.ip_address(host_ip_str)
        except ValueError:
            logger.warning("Invalid server_id %r found in server_list, skipping", host_ip_str)
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
                logger.warning("Invalid device_ip %r for %r; skipping", device_ip_str, host_ip_str)
                continue

            try:
                device_id = int(device_id_str)
            except ValueError:
                logger.warning("Invalid device_id %r for %r; skipping", device_id_str, host_ip_str)
                continue

            try:
                rank_id = int(rank_id_str)
            except ValueError:
                logger.warning("Invalid rank_id %r for %r; skipping", rank_id_str, host_ip_str)
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
            logger.warning("Expected %r in rank table but not found", list_name)
            continue

        if len(device_list) > _HOST_LIMIT * _DEVICE_LIMIT_PER_HOST:
            raise RankTableParseError(f"{list_name!r} length exceeds limit {_HOST_LIMIT * _DEVICE_LIMIT_PER_HOST}")

        for dev in device_list:
            host_ip_str = dev.get("server_id", "")
            try:
                host_ip = ipaddress.ip_address(host_ip_str)
            except ValueError:
                logger.warning("Invalid server_id %r; skipping", host_ip_str)
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
                logger.warning("Invalid device_ip %r for %r; skipping", device_ip_str, host_ip_str)
                continue

            try:
                device_id = int(device_id_str)
            except ValueError:
                logger.warning("Invalid device_id %r for %r; skipping", device_id_str, host_ip_str)
                continue

            try:
                cluster_id = int(cluster_id_str)
            except ValueError:
                logger.warning(
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
        rank_version = Version(version_str)
    except InvalidVersion as e:
        raise RankTableParseError(f"Invalid version {version_str!r} found in {to_user_path(path)}") from e

    return RankTable(
        host_to_devices=host_to_devices,
        server_count=server_count,
        version=rank_version,
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
        rank_version = Version(version_str)
    except InvalidVersion as e:
        raise RankTableParseError(f"Invalid version {version_str!r} found in {to_user_path(path)}") from e

    return RankTable(
        host_to_devices=host_to_devices,
        server_count=server_count,
        version=rank_version,
    )


_RANK_TABLE_PARSERS: dict[Framework, Callable[[str], RankTable]] = {
    Framework.MINDIE: _parse_mindie,
    Framework.VLLM: _parse_vllm,
}


def parse_rank_table(path: str, framework: Framework) -> RankTable:
    """
    Parse a rank table file for the given framework.

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


# Weight directory resolution

_VLLM_MODEL_RE = re.compile(
    r"""vllm\s+serve\s+(?:"([^"]+)"|'([^']+)'|([^\s"']+))|"""
    r"""--model[=\s](?:"([^"]+)"|'([^']+)'|([^\s"']+))"""
)
_SGLANG_MODEL_RE = re.compile(r"""(?:sglang|python.*sglang).*--model[=\s]+["']?([^\s"']+)""")


def _default_mindie_config_path() -> Optional[Path]:
    base = os.environ.get("MIES_INSTALL_PATH") or "/usr/local/Ascend/mindie/latest/mindie-service"
    candidate = Path(base) / "conf" / "config.json"
    resolved = candidate.resolve()
    return resolved if resolved.is_file() else None


def _weight_dir_from_mindie_config(config_path: Optional[Path]) -> str:
    path = config_path or _default_mindie_config_path()
    if path is None or not path.is_file():
        location = to_user_path(path) if path is not None else "None"
        raise WeightDirNotFoundError(f"MindIE config not found at {location}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        weight_dir = data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelWeightPath"]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RankTableParseError(f"Could not extract modelWeightPath from {to_user_path(path)}") from exc

    logger.info("Resolved weight_dir from MindIE config: %r", weight_dir)
    return weight_dir


def _weight_dir_from_script(
    script_path: Path,
    pattern: "re.Pattern[str]",  # quoted: re.Pattern[X] subscript requires 3.9+
) -> str:
    if not script_path.is_file():
        raise WeightDirNotFoundError(f"Launch script not found: {to_user_path(script_path)}")

    try:
        with open(script_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as exc:
        raise WeightDirNotFoundError(f"Failed to read script {to_user_path(script_path)}") from exc

    m = pattern.search(content)
    if not m:
        raise WeightDirNotFoundError(f"No model path pattern found in {to_user_path(script_path)}")

    # Return the first non-None capture group.
    weight_dir = next(g for g in m.groups() if g is not None)
    logger.info("Resolved weight_dir from %r: %r", script_path, weight_dir)
    return weight_dir


def _weight_dir_vllm(config_path: Optional[Path]) -> str:
    if config_path is None:
        raise ValueError("config_path (launch script) is required for VLLM")
    return _weight_dir_from_script(config_path, _VLLM_MODEL_RE)


def _weight_dir_sglang(config_path: Optional[Path]) -> str:
    if config_path is None:
        raise ValueError("config_path (launch script) is required for SGLANG")
    return _weight_dir_from_script(config_path, _SGLANG_MODEL_RE)


# Dispatch table replaces `match` statement (3.10+).
_WEIGHT_DIR_RESOLVERS: dict[Framework, Callable[[Optional[Path]], str]] = {
    Framework.MINDIE: _weight_dir_from_mindie_config,
    Framework.VLLM: _weight_dir_vllm,
    Framework.SGLANG: _weight_dir_sglang,
}


def resolve_weight_dir(
    framework: Framework,
    config_path: Optional[Path] = None,
) -> str:
    """
    Resolve the model weight directory for the given framework.

    For MINDIE, *config_path* points to config.json (None = use default).
    For VLLM and SGLANG, *config_path* is the launch shell script and is required.

    The caller is responsible for caching the result.

    Args:
        framework: The active inference framework.
        config_path: Framework-specific config file or launch script.

    Returns:
        Weight directory path string extracted from the config/script.

    Raises:
        WeightDirNotFoundError: File missing or model pattern not found.
        RankTableParseError: JSON structure unexpected (MINDIE only).
        ValueError: Framework unsupported, or required config_path missing.
    """
    resolver = _WEIGHT_DIR_RESOLVERS.get(framework)
    if resolver is None:
        raise ValueError(
            f"Weight dir resolution not supported for {framework!r}. Supported: {list(_WEIGHT_DIR_RESOLVERS)}"
        )
    return resolver(config_path)
