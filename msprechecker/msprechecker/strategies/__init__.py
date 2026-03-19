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

__all__ = [
    "CollectStrategyGroup",
    "CollectStrategy",
    "Image",
    "Network",
    "Ascend",
    "Weight",
    "Configs",
    "Config",
    "Sys",
    "Lscpu",
    "Env",
    "Driver",
    "CPUHighPerformance",
    "VirtualMachine",
    "TransparentHugepage",
    "Kernel",
    "JeMalloc",
    "CPU",
    "NPU",
]

from .base_strategy import CollectStrategyGroup, CollectStrategy
from .image_strategy import Image
from .network_strategy import Network
from .ascend_strategy import Ascend
from .weight_strategy import Weight
from .config_strategy import Configs, Config
from .sys_strategy import Sys, Lscpu
from .env_strategy import Env
from .ascend_strategy import Driver
from .sys_strategy import CPUHighPerformance, VirtualMachine, TransparentHugepage, Kernel, JeMalloc
from .stress_strategy import CPU, NPU
