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

import os
import time

import psutil

from ...utils import SimpleProgressBar

from .base import BaseStressCollector


class CPUStressCollector(BaseStressCollector):
    def __init__(self, error_handler=None):
        super().__init__(error_handler)
        self.error_handler.type = "cpu stress"

    def _collect_data(self):
        if not self.torch:
            return []

        cpu_ids = os.cpu_count()
        output = dict.fromkeys(range(cpu_ids), 0)

        self.torch.set_num_threads(cpu_ids)

        for cpu_id in SimpleProgressBar(range(cpu_ids)):
            start_time = time.time()
            self._matmul_stress_test("cpu", cpu_id)
            end_time = time.time()
            cpu_time = (end_time - start_time) * 1000

            output[cpu_id] = cpu_time
        return output

    def _get_free_memory(self, device):
        return psutil.virtual_memory().available
