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

from ...utils import SimpleProgressBar
from .base import BaseStressCollector


class NPUStressCollector(BaseStressCollector):
    def __init__(self, error_handler=None):
        super().__init__(error_handler)
        self.error_handler.type = "npu stress"

        try:
            import torch_npu
        except ImportError as e:
            self.error_handler.add_error(
                filename=__file__,
                function="__init__",
                lineno=25,
                what="当前环境没有安装 torch_npu",
                reason=str(e),
            )
            self.torch_npu = None
        else:
            self.torch_npu = torch_npu

    def _collect_data(self):
        if not self.torch_npu:
            return []

        device_ids = self.torch_npu.npu.device_count()
        output = dict.fromkeys(range(device_ids), 0)

        for device_id in SimpleProgressBar(range(device_ids)):
            # 创建事件对象，用于记录运算的开始和结束时间
            start_event = self.torch_npu.npu.Event(enable_timing=True)
            end_event = self.torch_npu.npu.Event(enable_timing=True)

            start_event.record()
            self._matmul_stress_test("npu", device_id)
            end_event.record()

            # 同步当前流，确保全部运算均已完成
            self.torch_npu.npu.current_stream().synchronize()
            npu_time = start_event.elapsed_time(end_event)
            output[device_id] = npu_time

        return output

    def _get_free_memory(self, device):
        if self.torch.npu.is_available():
            total_memory = self.torch.npu.get_device_properties(device).total_memory
            used_memory = self.torch.npu.memory_allocated(device)
            return total_memory - used_memory
        else:
            self.error_handler.add_error(
                filename=__file__,
                function="_get_free_memory",
                lineno=62,
                what=f"尝试获取 device '{device}' 上剩余的内存失败",
                reason="'self.torch.npu.is_available()' 返回了 False",
            )
            return 0
