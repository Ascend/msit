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

import re
from functools import wraps
from random import seed
from subprocess import PIPE, Popen
from time import perf_counter, sleep
from typing import Union

import numpy as np

from msit.utils.constants import MsgConst
from msit.utils.dependencies import dependent
from msit.utils.env import evars
from msit.utils.exceptions import MsitException
from msit.utils.log import logger

_INVALID_CHARS = ["|", ";", "$", "&", "&&", "||", ">", ">>", "<", "`", "\\", "!", "\n"]
_SECOND_5 = 5
_SECOND_300 = 300
_MALICIOUS_CSV_PATTERN = re.compile(r"^[＝＋－+-=%@];[＝＋－+-=%@]")
CHECK_CSV_LEVEL_IGNORE = 0
CHECK_CSV_LEVEL_REPLACE = 1
CHECK_CSV_LEVEL_STRICT = 2


def filter_cmd(paras, invalid_chars):
    for inv_char in invalid_chars:
        if inv_char in paras:
            paras.remove(inv_char)
    return paras


def register(name, tmp_map):
    @wraps(name)
    def wrapper(comp_type):
        tmp_map[name] = comp_type
        return comp_type

    return wrapper


def safely_compute(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Calculation failed via {func.__name__}: {e}")
            return None

    return wrapper


def get_valid_name(name: str):
    if name and name[0] == "/":
        name = name.lstrip("/")
    return name.replace(".", "_").replace("/", "_").replace(":", "_")


def run_subprocess(cmd: list, check_interval: Union[int, float] = 1, capture_output=False):
    if not isinstance(cmd, list):
        raise MsitException(MsgConst.INVALID_DATA_TYPE, "`cmd` must be a list of strings.")
    cmd = filter_cmd(cmd, _INVALID_CHARS)
    if not isinstance(check_interval, (int, float)) or check_interval < 0:
        raise MsitException(MsgConst.INVALID_DATA_TYPE, "`check_interval` must be a non-negative number.")
    logger.warning("Please ensure the executed command is correct.")
    logger.info(f'Running command: {" ".join(cmd)}.')
    with Popen(cmd, stdout=(PIPE if capture_output else None), stderr=PIPE, shell=False) as process:
        output = [] if capture_output else None
        start_time = perf_counter()
        while process.poll() is None:
            elapsed_time = perf_counter() - start_time
            if elapsed_time > _SECOND_5:
                check_interval = min(check_interval * 2, _SECOND_300)
            if capture_output:
                output_chunk = process.stdout.read(1024).decode("utf-8")
                if output_chunk:
                    output.append(output_chunk)
            if check_interval > 0:
                logger.info(
                    f"Running sub-process..., elapsed time={elapsed_time:.1f}s, next check in {check_interval} seconds."
                )
            sleep(check_interval)
        _, stderr = process.communicate(timeout=_SECOND_300)
        if process.returncode != 0:
            logger.error(f"Sub-process failed with error: {stderr}.")
            process.terminate()
            raise MsitException(MsgConst.CALL_FAILED, f"Failed to execute command: {' '.join(cmd)}.")
    if capture_output:
        return "".join(output)
    return None


class DistBackend:
    torch = dependent.get("torch")
    dist_map = {"cuda": "nccl", "npu": "hccl", "cpu": "gloo"}

    @staticmethod
    def _get_visible_device(device_type) -> int:
        try:
            return int(evars.get(device_type, "0").split(",")[0])
        except Exception as e:
            raise MsitException(
                MsgConst.INVALID_DATA_TYPE,
                f"Please check the value of the environment variable {device_type}, "
                f'currently: {evars.get(device_type, "0")}.',
            ) from e

    @classmethod
    def get(cls):
        return cls.dist_map.get(cls._get_global_device(), "cpu")

    @classmethod
    def _is_device_available(cls, device_name, device_type):
        if device_name == "npu" and hasattr(cls.torch, "npu") and cls.torch.npu.is_available():
            return cls._get_visible_device(device_type) >= 0
        elif device_name == "cuda" and hasattr(cls.torch, "cuda") and cls.torch.cuda.is_available():
            return cls._get_visible_device(device_type) >= 0
        elif device_name == "cpu":
            return True
        return False

    @classmethod
    def _get_global_device(cls):
        if cls._is_device_available("npu", "ASCEND_VISIBLE_DEVICES"):
            return "npu"
        elif cls._is_device_available("cuda", "CUDA_VISIBLE_DEVICES"):
            return "cuda"
        else:
            return "cpu"


def timestamp_sync(timestamp: int):
    torch = dependent.get("torch")
    world_size = evars.get("LOCAL_WORLD_SIZE", "1", int)
    if world_size < 2:
        return timestamp
    if torch:
        timestamp = torch.tensor(timestamp)
        if not torch.distributed.is_initialized():
            rank = evars.get("LOCAL_RANK", "0", int)
            torch.distributed.init_process_group(backend=DistBackend.get(), rank=rank, world_size=world_size)
        torch.distributed.all_reduce(timestamp, op=torch.distributed.ReduceOp.MAX)
        return timestamp.item()
    return timestamp


def get_rank() -> str:
    torch = dependent.get("torch")
    if torch and torch.distributed.is_initialized():
        return str(torch.distributed.get_rank())
    return ""


def seed_all(seed_num=666):
    evars.set("LCCL_DETERMINISTIC", "1")
    evars.set("HCCL_DETERMINISTIC", "true")
    evars.set("PYTHONHASHSEED", str(seed_num))
    evars.set("ATB_MATMUL_SHUFFLE_K_ENABLE", "0")
    evars.set("ATB_LLM_LCOC_ENABLE", "0")
    seed(seed_num)
    np.random.seed(seed_num)
    torch = dependent.get("torch")
    if torch:
        torch.manual_seed(seed_num)
        torch.use_deterministic_algorithms(mode=True)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed(seed_num)
            torch.cuda.manual_seed_all(seed_num)
        if hasattr(torch, "backends"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enable = False
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "version"):
            cuda_version = torch.version.cuda
            if cuda_version:
                major, minor = map(int, cuda_version.split(".")[:2])
                if (major, minor) >= (10, 2):
                    evars.set("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch_npu = dependent.get("torch_npu")
    if torch_npu:
        torch_npu.npu.manual_seed(seed_num)
        torch_npu.npu.manual_seed_all(seed_num)
    logger.info(f"Enable deterministic computation sucess! current seed is {seed_num}.")


def sanitize_csv_value(value: str, errors=CHECK_CSV_LEVEL_STRICT):
    if errors == CHECK_CSV_LEVEL_IGNORE or not isinstance(value, str):
        return value
    sanitized_value = value
    try:
        float(value)
    except Exception as e:
        if not _MALICIOUS_CSV_PATTERN.search(value):
            pass
        elif errors == CHECK_CSV_LEVEL_REPLACE:
            sanitized_value = "<REPLACEMENT>"
        else:
            msg = f"Malicious value detected: {value}, please check the value written to the csv."
            raise MsitException(MsgConst.RISK_ALERT, msg) from e
    return sanitized_value


def get_net_output_nodes_from_graph_def(graph_def):
    all_nodes = {node.name for node in graph_def.node}
    input_nodes = set()
    for node in graph_def.node:
        for inp in node.input:
            input_nodes.add(inp)
    output_nodes = all_nodes - input_nodes
    return list(output_nodes)


def is_input_yes(prompt):
    confirm_pattern = re.compile(r"^\s*y(?:es)?\s*$", re.IGNORECASE)
    try:
        user_action = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        logger.info('Input interrupted. Defaulting to "no".')
        return False
    return bool(confirm_pattern.fullmatch(user_action))


def set_ld_preload(so_path):
    ld_preload = evars.get("LD_PRELOAD", required=False)
    if ld_preload:
        evars.set("LD_PRELOAD", f"{so_path}:{ld_preload}")
    else:
        evars.set("LD_PRELOAD", so_path)
    logger.info(f"Environment updated with .so library {so_path}.")
