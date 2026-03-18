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
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_strategy import CollectStrategy
from ..utils import LOGGER, Utils


class Stress(CollectStrategy):
    def __init__(
            self, name, *, batch_size, seq_len, hidden_size, intermediate_size, epochs=5
    ):
        self._torch = None
        try:
            import torch

            self._torch = torch
        except ImportError:
            LOGGER.warning("Failed to import torch")

        for param_name, param_value in [
            ("batch_size", batch_size),
            ("seq_len", seq_len),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("epochs", epochs),
        ]:
            if not isinstance(param_value, int) or param_value <= 0:
                raise ValueError(
                    f"{param_name} must be a positive integer, got {param_value!r}"
                )

        super().__init__(name)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._epochs = epochs

    @property
    @abstractmethod
    def device_type(self) -> str:
        pass

    @abstractmethod
    def _get_free_memory(self, device) -> float:
        pass

    def _calculate_tensor_memory(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        import operator

        # Use functools.reduce for Python 3.7 compatibility (math.prod added in 3.8)
        from functools import reduce

        return reduce(operator.mul, shape, 1) * 4  # float32 = 4 bytes

    def _check_memory_for_matmul(self, device_pos):
        mat_a_mem = self._calculate_tensor_memory(
            (self._batch_size, self._seq_len, self._hidden_size)
        )
        mat_b_mem = self._calculate_tensor_memory(
            (self._batch_size, self._hidden_size, self._intermediate_size)
        )
        # addbmm output shape is (seq_len, intermediate_size); unchanged
        mat_c_mem = self._calculate_tensor_memory(
            (self._seq_len, self._intermediate_size)
        )
        total_required = mat_a_mem + mat_b_mem + mat_c_mem

        free_memory = self._get_free_memory(device_pos)
        safety_margin = 0.2
        available_with_margin = free_memory * (1 - safety_margin)
        has_enough_mem = total_required <= available_with_margin
        LOGGER.debug(
            "Device %s - Required memory: %d bytes, Free memory: %d bytes, "
            "Available with margin: %d bytes",
            device_pos,
            total_required,
            free_memory,
            available_with_margin,
        )

        if not has_enough_mem:
            LOGGER.warning(
                "Insufficient memory on device %s for matmul operation", device_pos
            )
            return False

        return True

    def _matmul_stress_test(self, device_id):
        """Run matrix-multiply stress on one device and return elapsed ms."""
        device_pos = f"{self.device_type}:{device_id}"

        if not self._check_memory_for_matmul(device_pos):
            return 0.0

        start_time = time.perf_counter()
        for _ in range(self._epochs):
            mat_a = self._torch.randn(
                self._batch_size, self._seq_len, self._hidden_size
            ).to(device_pos)
            mat_b = self._torch.randn(
                self._batch_size, self._hidden_size, self._intermediate_size
            ).to(device_pos)
            mat_c = self._torch.zeros(self._seq_len, self._intermediate_size).to(
                device_pos
            )
            self._torch.addbmm(mat_c, mat_a, mat_b)

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000

    def execute(self):
        if not self._torch:
            LOGGER.error("torch is not available, skip the stress test")
            return None

        output = {}
        cpu_count = os.cpu_count() or 1
        self._torch.set_num_threads(cpu_count)
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            future_to_id = {
                executor.submit(self._matmul_stress_test, cpu_id): cpu_id
                for cpu_id in range(cpu_count)
            }
            for future in as_completed(future_to_id):
                cpu_id = future_to_id[future]
                try:
                    elapsed_ms = future.result()
                    LOGGER.debug(
                        "Stress test completed on device %s:%s in %.2f ms",
                        self.device_type,
                        cpu_id,
                        elapsed_ms,
                    )
                    output[cpu_id] = elapsed_ms
                except Exception:
                    Utils.log_error_and_exit(
                        "Stress test failed on device %s:%s",
                        self.device_type,
                        cpu_id,
                    )
                    output[cpu_id] = None

        return output


class CPU(Stress):
    def __init__(
            self,
            name: str = "cpu",
            *,
            batch_size=1,
            seq_len=512,
            hidden_size=1024,
            intermediate_size=64,
            epochs=5,
    ):
        super().__init__(
            name,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            epochs=epochs,
        )

    @property
    def device_type(self) -> str:
        return "cpu"

    def _get_free_memory(self, device) -> float:
        import psutil

        memory_available = psutil.virtual_memory().available
        LOGGER.debug("Available CPU memory: %d bytes", memory_available)
        return memory_available


class NPU(Stress):
    def __init__(
            self,
            name: str = "npu",
            *,
            batch_size=1,
            seq_len=4096,
            hidden_size=8192,
            intermediate_size=3584,
            epochs=5,
    ):
        super().__init__(
            name,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            epochs=epochs,
        )

        self._torch_npu = None