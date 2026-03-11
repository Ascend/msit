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

import itertools
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Protocol is available in typing since 3.8; fall back to typing_extensions for 3.7.
try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version
import socket

import psutil
from packaging.version import Version


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
LOG_FORMAT = "[%(levelname)s] [%(name)s] %(message)s"


logger = logging.getLogger(__name__)


def is_in_container():
    def check_docker_env_file():
        docker_env_file = "/.dockerenv"
        return os.path.exists(docker_env_file)

    def check_first_process():
        first_proc = "/proc/1"
        schedule_file = os.path.join(first_proc, "sched")

        try:
            with open(schedule_file) as f:
                first_line = f.readlines(1)
        except Exception:
            return True

        if first_line and first_line[0] and first_line[0].startswith("systemd"):
            return False

        return True

    return check_docker_env_file() or check_first_process()


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def is_valid_ip(ip: str):
    single_address = "(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
    ip_pattern = re.compile(rf"^{single_address}(?:\.{single_address}){{3}}$")

    return bool(ip_pattern.match(ip))


def get_pkg_version(pkg_name: str):
    try:
        return Version(version(pkg_name))
    except PackageNotFoundError:
        return None


def get_current_ip_and_addr():
    for interface, addrs in psutil.net_if_addrs().items():
        if any(interface.startswith(prefix) for prefix in ("docker", "lo")):
            continue
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith("127"):
                return interface, addr.address
    return "", ""


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


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    device_ip: str
    device_id: int
    rank_id: int


@dataclass(frozen=True)
class RankTable:
    host_to_devices: Dict[str, List[DeviceInfo]]
    server_count: int
    version: str


@dataclass(frozen=True)
class HardwareProfile:
    npu_count: int
    npu_type: NpuType
    npu_memory_mb: Optional[int]
    conn_mode: Optional[ConnMode]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class RankTableParseError(ValueError):
    """Raised when a rank table file exists but cannot be parsed correctly."""


class WeightDirNotFoundError(FileNotFoundError):
    """Raised when the weight directory cannot be located from config/script."""


# ---------------------------------------------------------------------------
# Hardware probing — pure functions, caller decides caching
# ---------------------------------------------------------------------------


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
        output = subprocess.check_output([lspci], stderr=subprocess.DEVNULL, text=True)
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
    """Return HBM capacity (MB) for device 0, or None if unavailable."""
    npu_smi = shutil.which("npu-smi")
    if not npu_smi:
        return None

    try:
        output = subprocess.check_output(
            [npu_smi, "info", "-i", "0", "-t", "memory"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        logger.exception("Failed to execute npu-smi")
        return None

    for line in output.splitlines():
        if "HBM Capacity" not in line or ":" not in line:
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
        output = subprocess.check_output(
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


# ---------------------------------------------------------------------------
# Framework detection — Protocol-based, stateless, composable
# ---------------------------------------------------------------------------


@runtime_checkable
class FrameworkProbe(Protocol):
    """Pluggable strategy for detecting a single inference framework."""

    def probe(self) -> Optional[Framework]:
        """Return the detected Framework, or None if not detected."""
        ...


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


DEFAULT_PROBES: Tuple[FrameworkProbe, ...] = (
    MindIEProbe(),
    VLLMProbe(),
    SGLangProbe(),
)


def detect_framework(
    probes: Tuple[FrameworkProbe, ...] = DEFAULT_PROBES,
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


_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
_IP_RE = re.compile(rf"^{_OCTET}(?:\.{_OCTET}){{3}}$")


def is_valid_ip_strict(value: str) -> bool:
    # ^ and $ anchors in _IP_RE already guarantee a full-string match.
    return bool(_IP_RE.match(value))


# Rank table parsing

_HOST_LIMIT = 1000
_DEVICE_LIMIT_PER_HOST = 32


def _load_json(path: Path) -> dict:
    """Load and return JSON from *path*; raise RankTableParseError on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise RankTableParseError(f"Failed to load JSON from {path!r}") from exc


def _parse_server_count(raw: object) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    logger.warning("Unexpected server_count %r; defaulting to 0", raw)
    return 0


def _parse_mindie(path: Path) -> RankTable:
    data = _load_json(path)
    host_to_devices: Dict[str, List[DeviceInfo]] = {}

    for host_num, server in enumerate(data.get("server_list", [])):
        if host_num >= _HOST_LIMIT:
            raise RankTableParseError("Host count exceeds limit")

        host_ip = server.get("server_id", "")
        if not is_valid_ip_strict(host_ip):
            logger.warning("Invalid server_id %r; skipping", host_ip)
            continue

        if host_ip not in host_to_devices:
            host_to_devices[host_ip] = []

        for dev_num, dev in enumerate(server.get("device", [])):
            if dev_num >= _DEVICE_LIMIT_PER_HOST:
                raise RankTableParseError(
                    f"Device count for host {host_ip!r} exceeds limit"
                )

            device_ip = dev.get("device_ip", "")
            if not is_valid_ip_strict(device_ip):
                logger.warning(
                    "Invalid device_ip %r for %r; skipping", device_ip, host_ip
                )
                continue

            raw_device_id = str(dev.get("device_id", ""))
            if not raw_device_id.isdigit():
                logger.warning(
                    "Invalid device_id %r for %r; skipping", raw_device_id, host_ip
                )
                continue

            raw_rank_id = str(dev.get("rank_id", ""))
            if not raw_rank_id.isdigit():
                logger.warning(
                    "Invalid rank_id %r for %r; skipping", raw_rank_id, host_ip
                )
                continue

            host_to_devices[host_ip].append(
                DeviceInfo(
                    device_ip=device_ip,
                    device_id=int(raw_device_id),
                    rank_id=int(raw_rank_id),
                )
            )

    return RankTable(
        host_to_devices=host_to_devices,
        server_count=_parse_server_count(data.get("server_count")),
        version=data.get("version", "1.0"),
    )


def _parse_vllm(path: Path) -> RankTable:
    data = _load_json(path)
    host_to_devices: Dict[str, List[DeviceInfo]] = {}

    for list_name in ("prefill_device_list", "decode_device_list"):
        device_list = data.get(list_name)
        if device_list is None:
            logger.warning("Expected %r in rank table but not found", list_name)
            continue

        if len(device_list) > _HOST_LIMIT * _DEVICE_LIMIT_PER_HOST:
            raise RankTableParseError(f"{list_name!r} length exceeds limit")

        for dev in device_list:
            host_ip = dev.get("server_id", "")
            if not is_valid_ip_strict(host_ip):
                logger.warning("Invalid server_id %r; skipping", host_ip)
                continue

            if host_ip not in host_to_devices:
                if len(host_to_devices) >= _HOST_LIMIT:
                    raise RankTableParseError("Host count exceeds limit")
                host_to_devices[host_ip] = []

            if len(host_to_devices[host_ip]) >= _DEVICE_LIMIT_PER_HOST:
                raise RankTableParseError(
                    f"Device count for host {host_ip!r} exceeds limit"
                )

            device_ip = dev.get("device_ip", "")
            if not is_valid_ip_strict(device_ip):
                logger.warning(
                    "Invalid device_ip %r for %r; skipping", device_ip, host_ip
                )
                continue

            raw_device_id = str(dev.get("device_id", ""))
            if not raw_device_id.isdigit():
                logger.warning(
                    "Invalid device_id %r for %r; skipping", raw_device_id, host_ip
                )
                continue

            raw_cluster_id = str(dev.get("cluster_id", ""))
            if not raw_cluster_id.isdigit():
                logger.warning(
                    "Invalid cluster_id %r for %r; skipping", raw_cluster_id, host_ip
                )
                continue

            host_to_devices[host_ip].append(
                DeviceInfo(
                    device_ip=device_ip,
                    device_id=int(raw_device_id),
                    rank_id=int(raw_cluster_id) - 1,  # vllm cluster_id is 1-based
                )
            )

    return RankTable(
        host_to_devices=host_to_devices,
        server_count=_parse_server_count(data.get("server_count")),
        version=data.get("version", "1.0"),
    )


_RANK_TABLE_PARSERS: Dict[Framework, Callable[[Path], RankTable]] = {
    Framework.MINDIE: _parse_mindie,
    Framework.VLLM: _parse_vllm,
}


def parse_rank_table(path: Path, framework: Framework) -> RankTable:
    """
    Parse a rank table file for the given framework.

    Currently supported frameworks: MINDIE, VLLM.
    SGLang does not define a rank table format and is intentionally unsupported.

    Args:
        path: Path to the rank table JSON file.
        framework: Determines the parse strategy.

    Returns:
        Parsed RankTable.

    Raises:
        RankTableParseError: File exists but cannot be parsed.
        ValueError: Framework is not supported.
    """
    parser = _RANK_TABLE_PARSERS.get(framework)
    if parser is None:
        raise ValueError(
            f"No rank table parser for {framework!r}. Supported: {list(_RANK_TABLE_PARSERS)}"
        )
    return parser(path)


# Weight directory resolution

_VLLM_MODEL_RE = re.compile(
    r"""vllm\s+serve\s+(?:"([^"]+)"|'([^']+)'|([^\s"']+))|"""
    r"""--model[=\s](?:"([^"]+)"|'([^']+)'|([^\s"']+))"""
)
_SGLANG_MODEL_RE = re.compile(
    r"""(?:sglang|python.*sglang).*--model[=\s]+["']?([^\s"']+)"""
)


def _default_mindie_config_path() -> Optional[Path]:
    base = (
        os.environ.get("MIES_INSTALL_PATH")
        or "/usr/local/Ascend/mindie/latest/mindie-service"
    )
    candidate = Path(base) / "conf" / "config.json"
    resolved = candidate.resolve()
    return resolved if resolved.is_file() else None


def _weight_dir_from_mindie_config(config_path: Optional[Path]) -> str:
    path = config_path or _default_mindie_config_path()
    if path is None or not path.is_file():
        raise WeightDirNotFoundError(f"MindIE config not found at {path!r}")

    try:
        with open(path) as f:
            data = json.load(f)
        weight_dir = data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
            "modelWeightPath"
        ]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RankTableParseError(
            f"Could not extract modelWeightPath from {path!r}"
        ) from exc

    logger.info("Resolved weight_dir from MindIE config: %r", weight_dir)
    return weight_dir


def _weight_dir_from_script(
    script_path: Path,
    pattern: "re.Pattern[str]",  # quoted: re.Pattern[X] subscript requires 3.9+
) -> str:
    if not script_path.is_file():
        raise WeightDirNotFoundError(f"Launch script not found: {script_path!r}")

    try:
        with open(script_path) as f:
            content = f.read()
    except Exception as exc:
        raise WeightDirNotFoundError(f"Failed to read script {script_path!r}") from exc

    m = pattern.search(content)
    if not m:
        raise WeightDirNotFoundError(f"No model path pattern found in {script_path!r}")

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
_WEIGHT_DIR_RESOLVERS: Dict[Framework, Callable[[Optional[Path]], str]] = {
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
