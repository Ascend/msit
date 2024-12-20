# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import random

import torch
import torch_npu

import numpy as np
from msmodelslim import logger


DEV = torch.device('cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def cleanup_memory(verbose=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        if torch.cuda.is_available():
            return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))
        elif torch.npu.is_available():
            return sum(torch.npu.memory_reserved(device=i) for i in range(torch.npu.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbose:
            logger.debug(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
    if torch.npu.is_available():
        torch.npu.empty_cache()
        memory_after = total_reserved_mem()
        if verbose:
            logger.debug(
                f"NPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
