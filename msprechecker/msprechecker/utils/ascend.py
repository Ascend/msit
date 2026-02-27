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
import stat
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Type
from functools import lru_cache

from msguard.security import open_s


logger = logging.getLogger(__name__)


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
        output = subprocess.check_output(
            ["/usr/bin/lspci"], stderr=subprocess.DEVNULL, text=True
        )
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
        output = subprocess.check_output(
            shlex.split(lldp_cmd), stderr=subprocess.DEVNULL, text=True
        )
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


# --- rank table ---
class Framework(Enum):
    MINDIE = "mindie"
    VLLM = "vllm"
    SGLANG = 'sglang'
    UNKNOWN = 'unknown'


@dataclass
class DeviceInfo:
    device_ip: str
    device_id: int
    rank_id: int


@dataclass
class RankTable:
    host_to_devices: Dict[str, List[DeviceInfo]]
    server_count: int
    version: str


class ParserRegistry:
    """Rank Table Parser Registry. Dynamically register and get rank table parser."""

    _registry = {}

    @classmethod
    def register(cls, framework: Framework):
        """Register a parser using decorator. Example usage: @ParserRegistry.register(xx)"""

        def wrapper(parser_cls):
            cls._registry[framework] = parser_cls
            return parser_cls

        return wrapper

    @classmethod
    def get(cls, framework: Framework) -> Type["RankTableParser"]:
        """Get a RankTableParser"""
        if framework not in cls._registry:
            raise ValueError(
                f"Not registered framework: {framework}. Registered framework: {set(cls._registry)}"
            )

        return cls._registry[framework]


class RankTableParser(ABC):
    def __init__(self):
        single_address = "(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
        self._ip_pattern = re.compile(
            rf"\b{single_address}(?:\.{single_address}){{3}}\b"
        )
        self.host_limits = 1000  # max 1000 hosts in rank table
        self.device_limits_per_host = 32  # max 32 devices per host

    @abstractmethod
    def parse(self, rank_table_path: str) -> RankTable:
        pass


class JsonParser(RankTableParser):
    @staticmethod
    def _load_json(path: str):
        try:
            with open_s(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Error occured while loading json data: %s", str(e))
            return {}

    @abstractmethod
    def _parse_devices(self, data: dict) -> Dict[str, DeviceInfo]:
        pass

    def parse(self, rank_table_path: str):
        data = self._load_json(rank_table_path)
        if not data:
            return RankTable([], 0, "'")

        server_count = data.get("server_count")

        if not server_count or not server_count.isdigit():
            logger.warning(
                "Expected 'server_count' to be a digit str. Got %r instead.",
                server_count,
            )
            server_count = "0"
        else:
            server_count = int(server_count)

        version = data.get("version", "1.0")
        host_to_devices = self._parse_devices(data)

        return RankTable(
            host_to_devices=host_to_devices, server_count=server_count, version=version
        )


@ParserRegistry.register(Framework.MINDIE)
class MindIEParser(JsonParser):
    def _parse_devices(self, data: dict) -> Dict[str, DeviceInfo]:
        host_to_devices = {}

        for host_num, server_info in enumerate(data.get("server_list", {})):
            if host_num > self.host_limits:
                raise RuntimeError(
                    "Number of hosts found in rank table exceeds the limit"
                )

            host_ip = server_info.get("server_id", "")
            if not self._ip_pattern.match(host_ip):
                logger.warning("Invalid 'server_id' from rank table: %r", host_ip)
                continue

            if host_ip not in host_to_devices:
                host_to_devices[host_ip] = []

            for device_num, device_info in enumerate(server_info.get("device", {})):
                if device_num > self.device_limits_per_host:
                    raise RuntimeError(
                        f"Number of devices for host {host_ip!r} found in rank table exceeds the limit"
                    )

                device_ip = device_info.get("device_ip")
                if not device_ip or not self._ip_pattern.match(device_ip):
                    logger.warning(
                        "Invalid 'device_ip' for 'server_id' %r from rank table: %r",
                        device_ip,
                        host_ip,
                    )
                    continue

                device_id = device_info.get("device_id")
                if not device_id or not device_id.isdigit():
                    logger.warning(
                        "Expected 'device_id' for 'server_id' %r to be a digit str. Got %r instead.",
                        device_id,
                        host_ip,
                    )
                    continue
                device_id = int(device_id)

                rank_id = device_info.get("rank_id")
                if not rank_id or not rank_id.isdigit():
                    logger.warning(
                        "Expected 'rank_id' for host %r to be a digit str. Got %r instead.",
                        rank_id,
                        host_ip,
                    )
                    continue
                rank_id = int(rank_id)

                host_to_devices[host_ip].append(
                    DeviceInfo(
                        device_ip=device_ip, device_id=device_id, rank_id=rank_id
                    )
                )

        return host_to_devices


@ParserRegistry.register(Framework.VLLM)
class VLLMParser(JsonParser):
    def _parse_devices(self, data: dict) -> Dict[str, DeviceInfo]:
        host_to_devices = {}

        for device_list_name in ("prefill_device_list", "decode_device_list"):
            if device_list_name not in data:
                logger.warning(
                    "Expected %r in rank table, but not found.", device_list_name
                )
                continue

            if (
                len(data[device_list_name])
                > self.host_limits * self.device_limits_per_host
            ):
                raise RuntimeError(
                    "Number of items found in rank table exceeds the limit"
                )

            for device_info in data[device_list_name]:
                host_ip = device_info.get("server_id")
                if not host_ip or not self._ip_pattern.match(host_ip):
                    logger.warning("Invalid host_ip from rank table: %r", host_ip)
                    continue

                if host_ip not in host_to_devices:
                    host_to_devices[host_ip] = []

                if len(host_to_devices) > self.host_limits:
                    raise RuntimeError(
                        "Number of hosts found in rank table exceeds the limit"
                    )

                if len(host_to_devices[host_ip]) > self.device_limits_per_host:
                    raise RuntimeError(
                        f"Number of devices for host {host_ip!r} found in rank table exceeds the limit"
                    )

                device_ip = device_info.get("device_ip")
                if not device_ip or not self._ip_pattern.match(device_ip):
                    logger.warning(
                        "Invalid 'device_ip' for 'server_id' %r from rank table: %r",
                        device_ip,
                        host_ip,
                    )
                    continue

                device_id = device_info.get("device_id")
                if not device_id or not device_id.isdigit():
                    logger.warning(
                        "Expected 'device_id' for 'server_id' %r to be a digit str. Got %r instead.",
                        device_id,
                        host_ip,
                    )
                    continue

                device_id = int(device_id)
                rank_id = device_info.get("cluster_id")
                if not rank_id or not rank_id.isdigit():
                    logger.warning(
                        "Expected 'cluster_id' for host %r to be a digit str. Got %r instead.",
                        rank_id,
                        host_ip,
                    )
                    continue

                rank_id = int(rank_id) - 1  # vllm cluster_id starts from 1
                host_to_devices[host_ip].append(
                    DeviceInfo(
                        device_ip=device_ip, device_id=device_id, rank_id=rank_id
                    )
                )

        return host_to_devices


weight_dir = None
framework = None


def search_weight_dir_mindie(mies_config_path=''):
    global weight_dir

    if not mies_config_path:
        mies_install_path = os.environ.get('MIES_INSTALL_PATH')
        if mies_install_path is None or not os.path.isdir(mies_install_path):
            logger.debug('MIES_INSTALL_PATH environment variable not set or invalid, using default path')
            mies_install_path = '/usr/local/Ascend/mindie/latest/mindie-service'

        mies_config_path = os.path.realpath(os.path.join(mies_install_path, 'conf', 'config.json'))
        if not os.path.isfile(mies_config_path):
            logger.warning('MIES configuration file not found at: %s', mies_config_path)
            return weight_dir
    
    try:
        with open_s(mies_config_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning('Failed to parse MIES configuration file: %s', str(e))
        return weight_dir

    try:
        weight_dir = data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
            "modelWeightPath"
        ]
    except KeyError as e:
        logger.warning('Missing required configuration key in MIES config file: %s', str(e))
    
    return weight_dir


def search_weight_dir_vllm(script_path=''):
    global weight_dir

    if not script_path:
        return weight_dir

    if not os.path.isfile(script_path):
        return weight_dir
    
    try:
        with open(script_path) as f:
            content = f.read()
    except Exception:
        return weight_dir
    
    weight_pattern = re.compile(r"""vllm\s+serve\s+(?:["']?)([^\s"']+)|--model(?:["']?\s*[=\s]?\s*(?:["']?)([^\s"']+))""")

    mo = weight_pattern.search(content)
    if mo:
        weight_dir = mo.group(1) or mo.group(2)
    
    return weight_dir


def search_weight_dir_sglang(script_path=''):
    pass



def get_weight_dir():
    global weight_dir

    if weight_dir is not None:
        return weight_dir

    global framework
    framework_to_search_method = {
        Framework.MINDIE: search_weight_dir_mindie,
        Framework.VLLM: search_weight_dir_vllm,
        Framework.SGLANG: search_weight_dir_sglang
    }
        
    if framework not in framework_to_search_method:
        logger.warning('Unsupported framework for weight directory search, defaulting to MINDIE search method')
        return search_weight_dir_mindie()
    
    return framework_to_search_method[framework]()


def get_framework():
    global framework

    if framework is not None:
        return framework
    
    try:
        import vllm_ascend
        framework = Framework.VLLM
        return framework
    except ImportError:
        vllm_workspace = '/vllm-workspace/vllm-ascend'
        if os.path.isdir(vllm_workspace):
            framework = Framework.VLLM
            return framework

    try:
        import sglang
        framework = Framework.SGLANG
        return framework
    except ImportError:
        pass

    mindie_dir = '/usr/local/Ascend/mindie'
    if os.path.isdir(mindie_dir) or any('MINDIE' in env_name for env_name in os.environ):
        framework = Framework.MINDIE
        return framework

    return Framework.UNKNOWN
