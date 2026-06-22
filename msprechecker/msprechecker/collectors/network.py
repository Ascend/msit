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

import shlex
import subprocess  # nosec B404
from pathlib import Path

from ..utils.path_io import validate_trusted_executable
from .base import BaseCollector

PING_PATH = Path("/usr/bin/ping")
PING_CMD = validate_trusted_executable(PING_PATH)
_PING_AVAILABLE = PING_CMD is not None


class PingCollector(BaseCollector):
    def __init__(self, error_handler=None, *, rank_table=None):
        super().__init__(error_handler)
        self.rank_table = rank_table

        if not _PING_AVAILABLE:
            self._ping_cmd = None
        else:
            self._ping_cmd = f"{PING_CMD} -c 3 -q -W 2 {{}}"

    def _collect_data(self):
        result = {}

        if not self._ping_cmd:
            self.error_handler.add_error(
                filename=__file__,
                function="_collect_data",
                lineno=36,
                what="当前环境没有 'ping' 命令或不可执行",
                reason=f"路径 '{PING_PATH}' 不存在、不可执行或未通过安全校验",
            )
            return result

        host_to_devices = self.rank_table.host_to_devices
        if not host_to_devices:
            self.error_handler.add_error(
                filename=__file__,
                function="_collect_data",
                lineno=47,
                what="传入的 'rank table' 没有解析出任何信息",
                reason="请检查 'rank table' 是否符合格式规范",
            )
            return result

        for host in host_to_devices:
            try:
                output = subprocess.check_output(  # nosec B603
                    shlex.split(self._ping_cmd.format(host)), stderr=subprocess.STDOUT, text=True, timeout=5
                )
            except Exception:
                output = "ping failed"
            result[host] = output

        return result
